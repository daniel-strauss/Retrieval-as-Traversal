# Spatial Memory Retrieval — Implementation Plan

**Author:** Daniel Strauss \
**Date:** 2026-04-04 \
**Status:** Planning

---

## Table of Contents
1. [Theory — Design Options & Trade-offs](#1-theory--design-options--trade-offs)
2. [Architecture Overview](#2-architecture-overview)
3. [New Features Required](#3-new-features-required)
4. [Detailed Class Changes](#4-detailed-class-changes)
5. [Milestones](#5-milestones)

---

## 1. Theory — Design Options & Trade-offs

### 1.1 The Problem

The current agent uses a **temporal sliding window** to retrieve memories: at each step, TemporalMemory
selects the most recent M env-steps as keys for the transformer's cross-attention. This means the agent
can only attend to memories ordered by *when* they were created, not *where* they were created. If an
observation at a spatially important location (e.g. a hint tile) slides out of the window, it is lost.

**Goal:** Add a spatial retrieval head that lets the agent request memories by *map position*, so that it
can recall observations from spatially important locations regardless of how many steps have passed.

### 1.2 Coordinate Frame

Positions must be **relative to the agent's starting position** (spawn), not absolute grid coordinates.
This avoids the agent "cheating" by memorising the absolute layout of a fixed map. On episode reset, the
spatial index is cleared and positions are recorded as offsets from the initial `agent_pos`.

In MiniGrid, `env.agent_pos` gives the absolute `(x, y)`. On reset, we store `spawn_pos = agent_pos`
and compute `relative_pos = agent_pos - spawn_pos` at each step.

### 1.3 Pretraining Options

We compared four approaches for training the spatial retrieval head:

| # | Approach | Signal | Pros | Cons |
|---|----------|--------|------|------|
| A | **RF-scaled prediction** | Supervised: predict the last-layer RF-scaled field from the spatial head | Cheap (no extra forward pass); clean regression target | Attention ≠ importance — the agent may attend to irrelevant positions; doesn't teach the head to request *unvisited* positions |
| B | **Advantage-weighted RF** | Supervised: same as (A) but weight the loss by  | Filters out unimportant attention patterns | Still attention-based → same core limitation; depends on advantage quality |
| C | **Counterfactual value ablation** | Causal: for each spatial-retrieval position p, compute `|V(s,m) - V(s, m\p)|` and train the head to predict high-importance positions | Direct causal signal; decorrelates importance from attention; ablates at position level (not token level) | Requires extra forward passes every k-th minibatch; needs critic to be reasonably trained (warmup); stop-gradient on importance target |
| D | **Counterfactual action KL** | Causal: replace value shift with KL divergence of the action distribution | Also causal | Noisier for discrete actions (small KL even for important memories); more expensive than value ablation |

**Decision: Option C (Counterfactual value ablation)** — strongest causal signal, feasible cost.

### 1.4 Counterfactual Value Ablation — Details

Every k-th minibatch during PPO training:

1. Normal forward pass already gives V(s, m).
2. For each sample in the minibatch, ablate the K spatial tokens (one position at a time):
   zero out all memory frames associated with position p and set their mask to False.
3. Batch all ablated versions into one forward pass through transformer + critic → V(s, m\p).
4. Importance: I(p) = |V(s, m) - V(s, m\p)| (stop-gradient, treat as fixed target).
5. Normalize I(p) over positions → importance weights.
6. Train spatial Gaussian head with importance-weighted log-prob loss against these targets.

**Cost:** One extra forward pass of size `B * K_spatial` every k minibatches. With K=2, k=4, that's
~50% of one normal minibatch every 4 minibatches ≈ 12.5% overhead.

**Warmup:** Skip the counterfactual loss for the first N iterations until the critic stabilises (e.g.
when the value loss drops below a threshold or after a fixed iteration count).

### 1.5 PPO on the Internal Action

After (or alongside) pretraining, the spatial retrieval head is treated as a second action head:
- It gets its own PPO clip loss with separate `internal_log_probs`.
- The spatial retrieval affects the value function through the memories it selects.
- Advantages flow back through the spatial action naturally.
- The counterfactual pretraining loss can be phased out or downweighted over time.

**Caveat:** Use separate clip ratios for external and internal actions. A joint ratio
`r = r_ext * r_spatial` couples the two heads — separate losses are more stable.

### 1.6 Exploration via Spatial Retrieval

The spatial head can drive exploration:
- When the head requests a position the agent hasn't visited, the spatial index returns no memory
  (null/zero frames, mask = False).
- The transformer sees these null tokens differently from real memories → the critic may assign lower
  value to this state, creating an advantage signal that pushes the agent to visit that position.
- This is a natural "curiosity" mechanism: the head predicts that information *should* exist at a
  position, but it doesn't yet, creating implicit exploration pressure.

**Optional extension (future):** Add an explicit intrinsic reward for spatial retrieval misses:
`r_explore = α * confidence(head, pos) * is_miss(pos)`.

### 1.7 Caveats

1. **Position space size**: For a 9×9 MiniGrid, the position space is 81 cells. A categorical
   distribution (9 logits per axis) would destroy spatial structure — positions 3 and 8 appear
   as unrelated as 3 and 4. We use **Gaussian heads** instead (see §1.7b): `Normal(μ, σ)` per
   axis provides smooth gradients proportional to distance and generalises to any grid size.

2. **Non-stationarity**: The spatial index content changes as the policy improves. A position that
   was informative early in training may become irrelevant later. The head must adapt continuously.

3. **Memory staleness**: A memory from position (x,y) made 50 steps ago may be outdated. Consider
   recency weighting in future work.

4. **Episode boundary**: The spatial index MUST be reset when an episode ends (done=True). The
   spawn-relative coordinate frame means indices from a previous episode are meaningless.

5. **Multiple memories per position**: A position may have multiple env-steps associated with it
   (the agent visits the same cell multiple times). The spatial index stores all of them.
   Retrieval strategy: return the most recent one, or all (with a capacity limit K_spatial).

6. **Random seed**: Adding new nn.Parameters (spatial head linears) changes the parameter
   initialisation order, which shifts the random seed for all subsequent parameters. This means
   results are not directly comparable to runs without the spatial head unless seeds are carefully
   managed.

### 1.7b Gaussian Spatial Retrieval Head

The spatial head outputs a **continuous 2D position** via two independent Gaussian distributions:
`Normal(μ_x, σ_x)` and `Normal(μ_y, σ_y)`. This is preferred over a categorical (one logit per
grid cell) because:

1. **Spatial inductive bias**: The Gaussian density encodes proximity — nearby cells get similar
   probability mass. A categorical treats every cell as unrelated.
2. **Gradient quality**: If the target is position 3 but the agent sampled 5, the Gaussian gradient
   smoothly pushes μ toward 3 with magnitude proportional to the error. A categorical only says
   "5 was wrong, 3 was right" — no distance signal.
3. **Compact parameterisation**: 4 outputs (μ_x, μ_y, log_σ_x, log_σ_y) vs 2×W logits. Fewer
   parameters, works identically for any grid size.
4. **Natural exploration**: σ controls exploration width. Early training with large σ covers the
   grid; the head narrows to precise locations as it learns. Entropy regularisation is analytical:
   H = ½ ln(2πeσ²).

**Implementation details:**
- Two linear layers: `mu_head: Linear(trxl_dim, 2)` → (μ_x, μ_y),
  `log_sigma_head: Linear(trxl_dim, 2)` → (log_σ_x, log_σ_y).
- σ is clamped: `sigma = clamp(exp(log_sigma), min=0.1, max=grid_size)` to prevent collapse.
- Sampling: `pos = Normal(mu, sigma).rsample()` (reparameterised for gradient flow).
- **Discretisation for SpatialMemory lookup**: `grid_pos = round(pos).int().clamp(min_bound, max_bound)`.
- **Boundary handling**: Positions are in spawn-relative coordinates, which can be negative.
  The Gaussian naturally covers this range. For SpatialMemory lookup, out-of-bounds samples
  produce misses (zero frames, mask=False), which is correct.
- **log_σ initialisation**: Init `log_sigma_head` bias to ~1.0 so early σ ≈ 2.7, covering most
  of a 9×9 grid from any position.

### 1.8 2D Spatial Positional Encoding

In addition to the existing *temporal* positional encoding (which tells the transformer *when* a
memory was created), we add a *spatial* positional encoding that tells the transformer *where* on
the grid a memory was created. This encoding is applied **uniformly to all memory tokens** — both
temporal and spatial — inside the Transformer, exactly like the temporal positional encoding.

Since every memory frame in the rollout buffer has an associated `perceived_position` (the agent's
spawn-relative position when the observation was made), the Transformer can look up the 2D spatial
embedding for every token it receives. There is no need to distinguish spatial tokens from temporal
tokens: `MemoryWindow` carries `perceived_positions` for all tokens, and the Transformer applies
the spatial encoding to all of them uniformly.

**Biological motivation — Grid cells.** In the mammalian hippocampal formation, grid cells in the
medial entorhinal cortex fire at regular spatial intervals, forming a hexagonal lattice that tiles
the environment. These cells provide a metric spatial code that the hippocampus combines with
contextual/temporal signals to form episodic memories. Our 2D spatial positional encoding serves
an analogous role: it provides a learned spatial coordinate signal that the transformer can combine
with the temporal positional encoding and the memory content itself. Just as grid cells give the
hippocampus a "where" signal independent of "what" was observed, the spatial embedding gives the
transformer a position tag that is independent of the observation content stored in the memory frame.

The analogy extends further: in the hippocampal formation, grid cells (spatial) and time cells
(temporal) provide two orthogonal coordinate systems that are combined in the hippocampus to
form a joint spatiotemporal index into episodic memory. Similarly, our Transformer receives two
additive positional signals for each memory token — temporal (from `memory_indices`) and spatial
(from `perceived_positions`) — forming a joint spatiotemporal representation.



Two implementation options:

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
|**Just stay with temporal encoding**|
| **Sinusoidal 2D** | Fixed sin/cos encoding with different frequencies for x and y, analogous to 2D extensions of Vaswani et al. | Zero parameters; generalises to any grid size; closer to periodic grid-cell firing patterns | Slightly less expressive for a fixed grid |
| **Sinusoidal 2D 60 degrees** | A grid like a gridcell sheets | |  |



**Decision:** Use haparams to switch 2D embedding on and of and find the best. 

---

## 2. Architecture Overview

### 2.1 High-Level Design

```
                 ┌──────────────────────────────────────────────────┐
                 │                    Agent                         │
                 │                                                  │
                 │  AgentModule                                     │
                 │  ├── encoder                                     │
                 │  ├── transformer (accepts temporal + spatial KV)  │
                 │  ├── actor_branches (external action)             │
                 │  ├── critic                                       │
                 │  ├── mu_head     (spatial: μ_x, μ_y)  ← NEW     │
                 │  └── log_sigma_head (spatial: log σ)  ← NEW     │
                 │                                                  │
                 │  TemporalMemory (sliding window indices/masks)   │
                 └──────────┬──────────────────────┬────────────────┘
                            │                      │
                    temporal retrieval     spatial request (x,y)
                    plan (indices, masks)  + perceived position
                            │                      │
                            ▼                      ▼
                 ┌──────────────────────────────────────────────────┐
                 │              MemoryHandler                        │
                 │                                                  │
                 │  MemoryRolloutBuffer (ring buffer, stores all)   │
                 │  SpatialMemory (per-env spatial index)    ← NEW  │
                 │                                                  │
                 │  get_memory_window_rollout()  → MemoryWindow     │
                 │  get_memory_window_minibatch() → MemoryWindow    │
                 │  get_spatial_memory_window()  → MemoryWindow NEW │
                 └──────────────────────────────────────────────────┘
```

### 2.2 Data Flow — Rollout Phase

```
For each rollout step:
  1. Trainer extracts perceived_position = agent_pos - spawn_pos (from env)
  2. Trainer calls memory_handler.get_memory_window_rollout(step-1) → temporal MemoryWindow
  3. Trainer calls memory_handler.get_spatial_memory_window(requested_positions, env_ids)
     → spatial MemoryWindow  (requested_positions from PREVIOUS step's write record)
  4. Agent.sample_action(obs, envs_t, fov, temporal_window, spatial_window)
     - Transformer: concat temporal + spatial tokens as keys
     - Actor branches: sample external action
     - Spatial heads: sample (x, y) = requested_position for NEXT step
  5. Agent returns MemoryWriteRecord containing:
     - pure_memory_frame, receptive_fields, temporal indices/masks (as before)
     - perceived_position (N, 2)        ← NEW
     - requested_position (N, 2)        ← NEW
  6. Trainer calls memory_handler.add_memory_write_record(...)
     - MemoryRolloutBuffer stores perceived_position and requested_position
     - SpatialMemory indexes: spatial_index[env][pos].append(env_step)
```

### 2.3 Data Flow — Training Phase

```
For each minibatch:
  1. Trainer calls memory_handler.get_memory_window_minibatch(env_ids, global_steps) → temporal window
  2. Trainer calls memory_handler.get_spatial_memory_window_minibatch(env_ids, global_steps) → spatial window
     (looks up the stored requested_position at those global_steps, then retrieves)
  3. Both windows are passed to agent.evaluate_actions()
  4. Every k-th minibatch: run counterfactual ablation (M5+)
```

### 2.4 Rendering — Spatial Retrieval Visualisation

The tiles currently being accessed by the spatial memory retrieval head should be highlighted in the
video overlay with a **distinct colour** (e.g. green or cyan, to distinguish from the existing red/purple
RF overlays).

- The Agent provides the `requested_position (N, 2)` each step.
- Trainer passes `requested_position[0]` (env 0) to `capture_frame()`.
- `OverlayRender.add_image_overlay()` gains an optional `spatial_tiles` argument to draw a coloured
  overlay on the requested cells.

---

## 3. New Features Required

### 3.1 Neural Network

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| F1 | Spatial retrieval heads | `AgentModule` | Two linear heads: `mu_head: Linear(trxl_dim, 2)` → (μ_x, μ_y), `log_sigma_head: Linear(trxl_dim, 2)` → (log σ_x, log σ_y). Output: `Normal(μ, σ)` distributions over continuous spawn-relative positions. |
| F2 | `Action.internal_action` populated | `AgentModule.get_action_and_value` | Sample/evaluate (x, y) from Gaussian heads via `Normal.rsample()`. Store log-probs in `Action.internal_log_probs`. Discretise via `round().int()` for SpatialMemory lookup. |

### 3.2 Data Types

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| F3 | `perceived_position` field | `MemoryWriteRecord` | `(N, 2)` int tensor: agent-relative position where observation was made. |
| F4 | `requested_position` field | `MemoryWriteRecord` | `(N, 2)` **float** tensor: spatial head's continuous output position (spawn-relative). Discretised to int for SpatialMemory lookup. |
| F5 | Spatial columns in buffer | `Memory` dataclass, `MemoryRolloutBuffer` | Two new `(N, buffersize, 2)` columns: `perceived_positions`, `requested_positions`. |

### 3.3 Memory System

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| F6 | `SpatialMemory` index | `src_new/model/retrieval_decision/spatial_memory.py` | Per-env dict: `(x,y) → list[env_step]`. Write, read, reset-on-done. |
| F7 | `get_spatial_memory_window()` | `MemoryHandler` | Given requested positions + env_ids, look up spatial index, fetch frames from rollout buffer, return `MemoryWindow`. |
| F8 | `get_spatial_memory_window_minibatch()` | `MemoryHandler` | Same as F7 but for training minibatches (reads stored `requested_position`). |

### 3.4 Agent

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| F9  | Accept spatial window | `Agent.sample_action`, `evaluate_actions` | New parameter `spatial_window: MemoryWindow | None`. |
| F10 | Concat temporal + spatial tokens | `Agent._policy_value_forward` | Before calling `module.get_action_and_value`, concatenate temporal and spatial frames/masks/indices. |
| F11 | `requested_position` in write record | `Agent.sample_action` | Populate from `Action.internal_action`. |

### 3.5 Transformer & Spatial Positional Encoding

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| F12 | 2D spatial positional embedding | `Transformer` | Indexed by `perceived_positions (N, M, 2)`. |
| F13 | `perceived_positions` in forward | `Transformer.forward` | New parameter `perceived_positions: (N, M, 2)`. Passed through from `MemoryWindow`. |
| F14 | Combined mask | `Transformer.forward` | Concatenated `(N, M_temporal + K_spatial)` mask. |

### 3.6 Trainer / Rendering

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| F15 | Extract spawn-relative position | `Trainer.run()` rollout loop | Compute `perceived_position = agent_pos - spawn_pos` from env. |
| F16 | Pass / store spatial data | `Trainer.run()` rollout + training | Thread spatial windows through rollout and minibatch loops. |
| F17 | Render spatial tiles | `capture_frame`, `OverlayRender` | New overlay colour for tiles being spatially retrieved. |
| F18 | Counterfactual loss | `Trainer.run()` training loop | Every k-th minibatch: ablate spatial positions, compute importance, train head. |
| F19 | PPO internal loss | `Trainer.run()` training loop | Separate clip loss for `internal_log_probs`. |

### 3.7 Config

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| F20 | `spatial_memory_k` | `TrainConfig` | Number of spatial tokens to retrieve (K_spatial). Default: 2. |
| F21 | `counterfactual_every_k` | `TrainConfig` | Run counterfactual ablation every k-th minibatch. Default: 4. |
| F22 | `counterfactual_warmup_iters` | `TrainConfig` | Skip counterfactual loss for first N iterations. Default: 50. |
| F23 | `spatial_loss_coef` | `TrainConfig` | Coefficient for the spatial pretraining loss. Default: 0.1. |
| F24 | `internal_action_coef` | `TrainConfig` | Coefficient for internal action PPO loss. Default: 0.5. |

---

## 4. Detailed Class Changes

### 4.1 `TrainConfig` (src_new/config.py)

Add fields:
```python
# Spatial memory
spatial_memory_k: int = 2                  # number of spatial tokens to retrieve
counterfactual_every_k: int = 4            # counterfactual ablation frequency
counterfactual_warmup_iters: int = 50      # skip counterfactual for first N iterations
spatial_loss_coef: float = 0.1             # weight for spatial pretraining loss
internal_action_coef: float = 0.5          # weight for PPO loss on internal action
```

### 4.2 `MemoryWriteRecord` (src_new/memory/types.py)

Add two new fields:
```python
perceived_position: torch.Tensor  # (N, 2) int — agent-relative (x, y)
requested_position: torch.Tensor  # (N, 2) int — spatial head output (x, y) for next-step retrieval
```

**`MemoryWindow`** — add field for spatial positional encoding (populated for **all** tokens,
both temporal and spatial):
```python
perceived_positions: torch.Tensor  # (B, M, 2) int — spawn-relative (x, y) of each memory token.
                                   # Used by the Transformer to apply 2D spatial positional
                                   # encoding (§1.8). Every memory frame has a known position
                                   # because perceived_position is stored in the rollout buffer
                                   # for every step.
```

### 4.3 `Memory` dataclass & `MemoryRolloutBuffer` (src_new/memory/rollout_buffer.py)

**Memory** — add two columns:
```python
perceived_positions: torch.Tensor  # (N, buffersize, 2)
requested_positions: torch.Tensor  # (N, buffersize, 2)
```

**MemoryRolloutBuffer.__init__** — allocate those tensors.

**MemoryRolloutBuffer.add_step** — write `memory_write_record.perceived_position` and
`memory_write_record.requested_position` into the ring buffer at the current entry.

**MemoryRolloutBuffer.get_window_by_pairs** — when constructing the MemoryWindow, also retrieve
`perceived_positions` and `requested_positions` for the queried global steps. The
`perceived_positions` are included in the returned `MemoryWindow` so the Transformer can apply
the 2D spatial positional encoding (§1.8) to all tokens. The `requested_positions` are needed
for minibatch spatial lookup.

### 4.4 `SpatialMemory` (src_new/model/retrieval_decision/spatial_memory.py)

Full implementation:
```python
class SpatialMemory:
    def __init__(self, num_envs, max_episode_steps, grid_w, grid_h):
        self.num_envs = num_envs
        self.grid_w = grid_w
        self.grid_h = grid_h
        # Per-env spatial index: (x, y) → list of env_steps
        # Stored as a tensor: (num_envs, grid_w, grid_h, max_entries) with a count tensor
        # Or simply as a list of dicts for clarity, then tensorise on read
        self.index: list[dict[tuple[int,int], list[int]]] = [dict() for _ in range(num_envs)]

    def write(self, env_id: int, position: tuple[int,int], env_step: int):
        key = (position[0], position[1])
        self.index[env_id].setdefault(key, []).append(env_step)

    def write_batch(self, positions: Tensor, env_steps: Tensor):
        # Vectorised write for all envs at once

    def read(self, env_id: int, position: tuple[int,int], k: int = 1) -> list[int]:
        # Return up to k most recent env_steps at that position
        key = (position[0], position[1])
        entries = self.index[env_id].get(key, [])
        return entries[-k:]  # most recent k

    def read_batch(self, env_ids: Tensor, positions: Tensor, k: int) -> Tensor:
        # Batch read: returns (B, k) tensor of env_steps, padded with -1 for misses

    def reset_env(self, env_id: int):
        self.index[env_id].clear()
```

### 4.5 `MemoryHandler` (src_new/memory/handler.py)

**Modified `get_memory_window_rollout` and `get_memory_window_minibatch`:** The returned
`MemoryWindow` must now include `perceived_positions: (B, M, 2)` for the temporal tokens.
These are read from the rollout buffer alongside the existing fields. This is needed so the
Transformer can apply 2D spatial positional encoding to all tokens (§4.7).

**New attributes:**
```python
self.spatial_memory = SpatialMemory(num_envs, max_episode_steps, grid_w, grid_h)
```

**Modified `add_memory_write_record`:** After writing to rollout buffer, also write to spatial index:
```python
self.spatial_memory.write_batch(
    positions=write_record.perceived_position,
    env_steps=envs_t,
)
# Reset spatial index for done envs
for env_id in done_envs.nonzero():
    self.spatial_memory.reset_env(env_id)
```

**New `get_spatial_memory_window(requested_positions, env_ids)` method:**
1. For each (env_id, requested_position): look up `spatial_memory.read(env_id, pos, k=K_spatial)`
   → get env_steps.
2. Convert env_steps to global_steps (env_step + episode_start_global_step).
3. Fetch pure_memory_frames from `memory_rollout_buffer` at those global steps.
4. Look up the `perceived_positions` from the rollout buffer for those global steps → `(B, K_spatial, 2)`.
5. Construct and return a MemoryWindow with shape `(B, K_spatial, L, D)`, including
   `perceived_positions` for the 2D spatial positional encoding (§1.8).
6. For positions with no entries (misses), return red frames

**New `get_spatial_memory_window_minibatch(env_ids, global_steps)` method:**
1. Read stored `requested_positions` from rollout buffer at the given global steps.
2. Call `get_spatial_memory_window()` with those positions.

### 4.6 `AgentModule` (src_new/model/agent_module.py)

**Uncomment and activate spatial heads:**
```python
if use_spatial_memory:
    self.mu_head = layer_init(nn.Linear(trxl_dim, 2), np.sqrt(0.01))       # → (μ_x, μ_y)
    self.log_sigma_head = layer_init(nn.Linear(trxl_dim, 2), np.sqrt(0.01))  # → (log σ_x, log σ_y)
    # Init log_sigma bias so initial σ ≈ 2.7, covering most of a 9×9 grid
    nn.init.constant_(self.log_sigma_head.bias, 1.0)
```

Note: The 2D spatial positional embedding tables live in the `Transformer` (§4.7), not here.

**`get_action_and_value` modifications:**
- The method receives `memory_frames` that are already concatenated (temporal + spatial)
  by the Agent layer, so the transformer processes them transparently.
- After the transformer + actor, sample from Gaussian heads:
  ```python
  if self.use_spatial_memory:
      mu = self.mu_head(x)                                    # (N, 2)
      log_sigma = self.log_sigma_head(x)                      # (N, 2)
      sigma = torch.exp(log_sigma).clamp(min=0.1, max=grid_size)
      dist = Normal(mu, sigma)
      if internal_action_in is None:
          internal_action_in = dist.rsample()                  # (N, 2) continuous
      internal_log_probs = dist.log_prob(internal_action_in)   # (N, 2)
  ```

**`get_value` modifications:** Same concatenation approach — receives pre-concatenated memory_frames.

### 4.7 `Transformer` (src_new/model/trxl.py) — Add 2D spatial positional encoding

The Transformer gains a **new parameter** `perceived_positions: (N, M, 2)` in `forward()` and
two learned embedding tables for 2D spatial positional encoding. The encoding is applied to **all**
memory tokens uniformly, exactly like the existing temporal positional encoding.

**New `__init__` parameters** (gated by `use_spatial_memory`):
```python
if use_spatial_memory:
    self.spatial_pos_emb_x_y = sin_cosine_stuff
```

**Updated `forward()` signature:**
```python
def forward(self, x: torch.Tensor,
            memory_frames: torch.Tensor, memory_mask: torch.Tensor,
            memory_indices: torch.Tensor,
            perceived_positions: torch.Tensor | None = None):  # NEW (N, M, 2)
```

**Spatial encoding applied alongside temporal encoding:**
```python
# Existing temporal positional encoding
if self.positional_encoding == "absolute":
    pos_embedding = self.pos_embedding_a(self.max_episode_steps)[memory_indices]
    memory_frames = memory_frames + pos_embedding.unsqueeze(2)
elif self.positional_encoding == "learned":
    memory_frames = memory_frames + self.pos_embedding_l[memory_indices].unsqueeze(2)

# NEW: 2D spatial positional encoding (§1.8)
if self.use_spatial_memory:
    space_emb = self.spatial_pos_emb_x_y(perceived_positions[:, :, :])  # (N, M, D//2)
    memory_frames = memory_frames + space_emb.unsqueeze(2)   # broadcast over layers
```

Every memory token now receives two additive positional signals:
- **Temporal** (from `memory_indices`): *when* the memory was created.
- **Spatial** (from `perceived_positions`): *where* on the grid the agent's observation was.

This mirrors the joint spatiotemporal code in the hippocampal formation (see §1.8). The
Transformer does not need to know whether a token was temporally or spatially retrieved —
it treats all tokens identically.

### 4.8 `Agent` (src_new/model/agent.py)

**`__init__`:** Remove the `raise NotImplementedError("TODO")` for `use_spatial_memory`.

**`sample_action`:**
- Accept new parameter: `spatial_window: MemoryWindow | None`
- If spatial_window is provided, concatenate with temporal window before forward pass
- After forward pass, extract `internal_action` from Action → becomes `requested_position`
- Add `perceived_position` and `requested_position` to `MemoryWriteRecord`

**`evaluate_actions`:** Same spatial_window parameter, concatenate before forward.

**`bootstrap_value`:** Same spatial_window parameter, concatenate before forward.

**`_policy_value_forward`:** Concatenation logic:
```python
if spatial_window is not None:
    memory_frames = torch.cat([memory_frames, spatial_window.pure_memory_frames], dim=1)
    memory_masks = torch.cat([memory_masks, spatial_window.memory_masks], dim=1)
    memory_indices_env = torch.cat([memory_indices_env, spatial_window.memory_indices_env], dim=1)
```

**`_validate_memory_window`:** Must be updated to handle M_temporal + K_spatial total tokens.
In gourmet mode, validate temporal and spatial windows separately before concatenation.

### 4.9 `Trainer` (src_new/trainer.py)

**`__init__`:**
- Store `spawn_pos` tensor `(num_envs, 2)` — initialised on env reset.
- Pass `spatial_memory_k` to MemoryHandler (if config says `use_spatial_memory`).

**Rollout loop — per step:**
```python
# Extract positions
agent_positions = torch.tensor([env.unwrapped.agent_pos for env in self.envs.envs], device=device)
perceived_position = agent_positions - self.spawn_pos  # (N, 2) agent-relative

# Retrieve spatial window from PREVIOUS step's requested_position
if train_conf.use_spatial_memory:
    spatial_window = self.memory_handler.get_spatial_memory_window(
        requested_positions=prev_requested_position,
        env_ids=torch.arange(train_conf.num_envs, device=device),
    )
else:
    spatial_window = None

# Agent includes spatial data
action, value, memory_write_record = agent.sample_action(
    obs=current_obs,
    episode_steps_envs=envs_t.clone(),
    fov=fov,
    memory_window=memory_window,
    spatial_window=spatial_window,
)

# Store perceived_position into write record (set by trainer, not agent)
memory_write_record.perceived_position = perceived_position
prev_requested_position = memory_write_record.requested_position.clone()
```

**Rollout loop — capture_frame:**
```python
# Pass spatial retrieval tiles for rendering
spatial_tiles_env0 = prev_requested_position[0].cpu().numpy()
envs.envs[0].capture_frame(
    env_t=envs_t[0].item(),
    rf_scaled=rf_scaled_env0[-1].cpu().numpy(),
    rf_binary=rf_binary_env0[-1].cpu().numpy(),
    spatial_tiles=spatial_tiles_env0,  # NEW
)
```

**Rollout loop — episode reset:**
```python
for id, done_i in enumerate(done):
    if done_i:
        envs_t[id] = 0
        self.spawn_pos[id] = torch.tensor(self.envs.envs[id].unwrapped.agent_pos, device=device)
        # SpatialMemory reset happens inside memory_handler.add_memory_write_record
```

**Training loop:**
```python
# Retrieve spatial window for minibatch
if train_conf.use_spatial_memory:
    mb_spatial_window = self.memory_handler.get_spatial_memory_window_minibatch(
        env_ids=mb_envs,
        global_steps=mb.mem_retrieval_steps,
    )
else:
    mb_spatial_window = None

# Pass to evaluate_actions
result, newvalue, _, _ = agent.module.get_action_and_value(...)
# OR: through agent.evaluate_actions(..., spatial_window=mb_spatial_window)
```

**Counterfactual loss (M6):** Every k-th minibatch:
```python
if (minibatch_counter % train_conf.counterfactual_every_k == 0
    and iteration > train_conf.counterfactual_warmup_iters
    and train_conf.use_spatial_memory):

    with torch.no_grad():
        baseline_value = newvalue.detach()

        # Ablate each spatial position
        K = train_conf.spatial_memory_k
        importance = torch.zeros(B, K, device=device)
        for k in range(K):
            ablated_frames = mb_combined_frames.clone()
            ablated_masks = mb_combined_masks.clone()
            spatial_offset = M_temporal + k
            ablated_frames[:, spatial_offset] = 0
            ablated_masks[:, spatial_offset] = False
            _, ablated_value, _, _ = agent.module.get_action_and_value(
                x=mb.obs,
                memory_frames=ablated_frames,
                memory_mask=ablated_masks,
                memory_indices=mb_combined_indices,
                action=...,
            )
            importance[:, k] = (baseline_value - ablated_value.detach()).abs()

    # Map importance to positions → train spatial head
    # (details depend on how positions map to K tokens)
    importance_norm = F.softmax(importance, dim=1)  # (B, K) normalised importance
    # spatial_positions: (B, K, 2) — the positions the head chose for each slot
    # spatial_mu, spatial_sigma: (B, 2) — from the normal forward pass
    spatial_dist = Normal(spatial_mu, spatial_sigma)
    # Importance-weighted negative log-prob of the chosen positions
    slot_log_probs = spatial_dist.log_prob(spatial_positions)  # (B, K, 2)
    spatial_loss = -(importance_norm.unsqueeze(-1) * slot_log_probs).sum(dim=(1, 2)).mean()
    loss += train_conf.spatial_loss_coef * spatial_loss
```

### 4.10 `OverlayRender` (src_new/env/overlay_render.py)

**`add_image_overlay`:** Accept optional `spatial_tiles` argument:
```python
def add_image_overlay(self, base, rf_scaled, rf_binary, spatial_tiles=None):
    ...
    if spatial_tiles is not None:
        spatial_mask = np.zeros((self.grid_w, self.grid_h), dtype=np.float32)
        for (x, y) in spatial_tiles:
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                spatial_mask[x, y] = 1.0
        overlay_layers.append(
            self.add_shape(spatial_mask, shape="fill", color=(0, 255, 128), max_alpha=0.35)
        )
    ...
```

### 4.11 `CustomRecordVideo` (src_new/env/custom_env_wrappers.py)

**`capture_frame`:** Accept optional `spatial_tiles` parameter, pass through to `_capture_frame`
and `render_overlay_frame`.

---

## 5. Milestones

### M1 — Spatial head exists, output discarded
**Target:** April 5, 2026

**Goal:** Add the spatial retrieval heads to `AgentModule`, sample from them, return via `Action`.
The output is not used anywhere. Training runs identically except for the random seed shift.

**Changes:**

| File | Change |
|------|--------|
| `src_new/config.py` | Add `spatial_memory_k: int = 2` to `TrainConfig` |
| `src_new/model/agent_module.py` | Add `mu_head` and `log_sigma_head` (gated by `use_spatial_memory`). In `get_action_and_value`: sample from Gaussian heads via `Normal(mu, sigma).rsample()`, populate `Action.internal_action` (continuous float) and `Action.internal_log_probs`. |
| `src_new/model/agent.py` | Remove the `raise NotImplementedError("TODO")` in `__init__`. No other logic changes. |

**Validation:**
- Run with `use_spatial_memory=False` → identical to before (exact same random seed).
- Run with `use_spatial_memory=True` → `action.internal_action` has shape `(N, 2)`,
  continuous float values (spawn-relative). `action.internal_log_probs` has shape `(N, 2)`.
  Early samples should be spread due to large initial σ.
- Loss and training curves are sane (spatial head output is discarded, so no effect on convergence).

---

### M2 — Store agent position in rollout buffer
**Target:** April 5, 2026

**Goal:** Capture the agent's spawn-relative position at each step and store it in the rollout buffer.
Nothing consumes this data yet.

**Changes:**

| File | Change |
|------|--------|
| `src_new/memory/types.py` | Add `perceived_position: torch.Tensor  # (N, 2)` and `requested_position: torch.Tensor  # (N, 2)` to `MemoryWriteRecord`. |
| `src_new/memory/rollout_buffer.py` | Add `perceived_positions: torch.Tensor  # (N, buffersize, 2)` and `requested_positions: torch.Tensor  # (N, buffersize, 2)` to `Memory`. Allocate in `__init__`. Write in `add_step`. Include in validation. |
| `src_new/trainer.py` | On env reset: store `spawn_pos[env_id]`. Each step: compute `perceived_position = agent_pos - spawn_pos`. After `sample_action`: set `memory_write_record.perceived_position = perceived_position`. Set `memory_write_record.requested_position` from `action.internal_action` (continuous float, or zeros if spatial memory disabled). |
| `src_new/model/agent.py` | In `sample_action`: add `requested_position` (from `action.internal_action` — continuous float — or zeros) to the returned `MemoryWriteRecord`. `perceived_position` left as a placeholder — trainer fills it in. |

**Caveat:** `spawn_pos` must be initialised at the very first `envs.reset()` and updated whenever
a particular env resets (done=True). In MiniGrid, `env.agent_pos` is available after `reset()`.

**Validation:**
- After a rollout, check `memory_rollout_buffer.data.perceived_positions` has valid values.
- Positions should be zero when env_step=0 (agent is at spawn).
- Positions should change as the agent moves.
- No change to training curves or loss.

---

### M3 — SpatialMemory index (write + read)
**Target:** April 6, 2026

**Goal:** Implement the spatial index and integrate it into `MemoryHandler`. Writes happen during
rollout. Reads are tested but output is not consumed by the agent yet.

**Changes:**

| File | Change |
|------|--------|
| `src_new/model/retrieval_decision/spatial_memory.py` | Full implementation: `__init__`, `write_batch`, `read_batch`, `reset_env`. Per-env dict mapping `(x,y) → list[env_step]`. |
| `src_new/memory/handler.py` | Add `SpatialMemory` as an attribute. In `add_memory_write_record`: call `spatial_memory.write_batch(perceived_positions, envs_t)`. Reset on done. |
| `src_new/config.py` | No changes (already have `use_spatial_memory` and `spatial_memory_k`). |

**Validation:**
- Unit test: write 5 positions across 2 envs, query back, verify correct env_steps returned.
- Unit test: reset env 0, verify its index is empty, env 1 still has data.
- Unit test: write same position multiple times, read with k=2, verify most recent 2 returned.
- Integration: run a short rollout, inspect spatial index contents, verify they match agent positions.
- No change to training.

---

### M4 — Retrieve spatial memories, log shapes
**Target:** April 6, 2026

**Goal:** Wire up the full spatial retrieval pipeline. The spatial window is constructed and passed
through the system, but the transformer does NOT consume it yet (just shape assertions).

**Changes:**

| File | Change |
|------|--------|
| `src_new/memory/handler.py` | Implement `get_spatial_memory_window(requested_positions, env_ids)` and `get_spatial_memory_window_minibatch(env_ids, global_steps)`. Returns `MemoryWindow` with shape `(B, K_spatial, L, D)`. Misses → zero frames, mask=False. |
| `src_new/trainer.py` | **Rollout loop:** Track `prev_requested_position`. Retrieve spatial window. Pass to `agent.sample_action(spatial_window=...)`. **Training loop:** Retrieve spatial minibatch window. |
| `src_new/model/agent.py` | `sample_action`, `evaluate_actions`, `bootstrap_value`: accept `spatial_window: MemoryWindow | None`. For now, assert shapes only — don't pass to transformer. |

**Caveat:** The first rollout step has no previous `requested_position`. Use zeros (which will
produce a miss → all-zero spatial window → mask all False). This is correct: at step 0 the agent
hasn't requested anything yet.

**Caveat:** `requested_position` is a continuous float from the Gaussian head. Before SpatialMemory
lookup, discretise: `grid_pos = round(requested_position).int()`. Out-of-bounds → miss.

**Caveat:** `get_spatial_memory_window` needs to convert env_steps (from spatial index) to
global_steps (for rollout buffer lookup). This requires knowing the episode start global step.
MemoryRolloutBuffer already tracks `dones` — we can compute episode_start from that. Alternatively,
store `episode_start_global_step` per env in MemoryHandler.

**Validation:**
- Spatial window has shape `(B, K_spatial, L, D)`.
- Misses (unvisited positions) produce zero frames and False masks.
- Visited positions produce non-zero frames and True masks.
- Training still works unchanged.

---

### M5 — Feed spatial tokens to transformer
**Target:** April 7, 2026

**Goal:** The transformer now receives concatenated temporal + spatial tokens. This is the first
milestone where spatial retrieval actually affects the agent's forward pass.

**Changes:**

| File | Change |
|------|--------|
| `src_new/model/agent.py` | In `_policy_value_forward`: concatenate temporal and spatial memory_frames, masks, indices, and `perceived_positions` along dim=1. Total keys: `M_temporal + K_spatial`. Pass combined `perceived_positions` to transformer. |
| `src_new/model/agent_module.py` | `get_action_and_value` and `get_value`: accept arbitrary M (no hardcoded check on window size). Pass `perceived_positions` through to Transformer. |
| `src_new/model/trxl.py` | Add `spatial_pos_emb_x = nn.Embedding(grid_w, dim // 2)` and `spatial_pos_emb_y = nn.Embedding(grid_h, dim // 2)` (gated by `use_spatial_memory`). In `forward()`: accept new `perceived_positions` parameter, apply 2D spatial encoding to all memory frames alongside temporal encoding (see §4.7). |
| `src_new/memory/handler.py` | `get_memory_window_rollout` and `get_memory_window_minibatch`: populate `perceived_positions` in the returned `MemoryWindow` by reading from the rollout buffer. |
| `src_new/model/receptive_field_utils.py` | `compute_receptive_fields`: attention_weights now have shape `(N, H, 1, M_temporal + K_spatial)`. Only the first M_temporal columns should be used for RF computation (spatial tokens don't have a prior RF). Split attention before RF computation. |
| `src_new/model/agent.py` | `_validate_memory_window`: Update to handle combined window size. Separate validation for temporal and spatial parts. RF computation uses only temporal attention weights. |
| `src_new/model/agent.py` | `validate_attention_weights`: Update expected M to `M_temporal + K_spatial`. |

**Caveat — Receptive fields:** The attention weights returned by the transformer now cover
M_temporal + K_spatial keys. The RF computation should only use the temporal portion (first
M_temporal columns) since spatial tokens don't carry recursive RFs. The spatial columns of attention
are interesting for visualisation but should not feed into the RF recursion.

**Caveat — gourmet_mode:** The validation that checks `memory_masks == expected_temporal_masks` will
fail because the combined mask is longer. Update to validate only the temporal slice.

**Caveat — bootstrap_value:** Must also receive and concatenate spatial window.

**Validation:**
- Training runs successfully with `use_spatial_memory=True`.
- Attention weights have shape `(N, H, 1, M_total)` where `M_total = M_temporal + K_spatial`.
- Initially, spatial tokens should get near-zero attention (untrained head, random positions).
- Loss still converges (spatial tokens shouldn't disrupt learning if properly masked).

---

### M5.5 — Render spatial retrieval tiles
**Target:** April 7, 2026

**Goal:** The tiles being accessed by the spatial retrieval head are highlighted in the video overlay
with a distinct colour (green/cyan).

**Changes:**

| File | Change |
|------|--------|
| `src_new/env/overlay_render.py` | `add_image_overlay`: accept optional `spatial_tiles: np.ndarray | None` parameter (shape `(K, 2)` — list of `(x, y)` grid positions). Draw a green/cyan `"fill"` overlay on those cells. |
| `src_new/env/custom_env_wrappers.py` | `capture_frame`: accept optional `spatial_tiles: np.ndarray | None`. Pass through to `_capture_frame` → `render_overlay_frame` → `add_image_overlay`. |
| `src_new/trainer.py` | In rollout loop: pass `requested_position[0]` (env 0, spawn-relative) to `capture_frame`. Convert to absolute grid coords by adding `spawn_pos[0]` for rendering. |

**Rendering colour convention:**
- **Red border:** Binary RF (existing) — cells the agent has *ever* seen through attention.
- **Purple circle:** Scaled RF (existing) — attention intensity per cell.
- **Green/cyan fill:** Spatial retrieval tiles (NEW) — cells currently being spatial-retrieved.

**Caveat:** The requested position is in agent-relative coordinates. For rendering on the grid, it
must be converted back to absolute coordinates: `abs_pos = requested_pos + spawn_pos`.

**Caveat:** Out-of-bounds positions (agent requests a position outside the grid) should be silently
clipped or skipped in the overlay.

**Validation:**
- In recorded videos, green tiles appear at positions the spatial head is requesting.
- Early in training: green tiles should be random (untrained head).
- The existing RF overlays (red, purple) are unaffected.

---

### M6 — Counterfactual pretraining loss
**Target:** April 8–9, 2026

**Goal:** Train the spatial head to predict value-relevant positions using counterfactual ablation.

**Changes:**

| File | Change |
|------|--------|
| `src_new/config.py` | Add `counterfactual_every_k`, `counterfactual_warmup_iters`, `spatial_loss_coef`. |
| `src_new/trainer.py` | Every k-th minibatch (after warmup): ablate spatial positions → compute importance → train Gaussian head with importance-weighted log-prob loss. See section 4.9 for detailed pseudocode. |
| `src_new/trainer.py` | Add `spatial_loss` to `Metrics` dataclass and W&B logging. |

**Detailed counterfactual procedure:**

1. After the normal forward pass, we have `newvalue` (V(s, m)) and the combined memory window.
2. For each spatial slot k ∈ [0, K_spatial):
   a. Clone the combined frames and masks.
   b. Zero out slot `M_temporal + k` in frames, set mask to False.
   c. Run a forward pass (transformer + critic only, no actor needed) to get V(s, m\k).
      Use `agent.module.get_value()` — cheaper than full `get_action_and_value`.
   d. `importance[k] = |newvalue - V(s, m\k)|`
3. Normalise importance to a distribution over K slots.
4. The Gaussian head parameters (μ, σ) are already computed in the normal forward pass.
   The target is over *slots*, not positions. We need to map:
   - Slot k → position (x_k, y_k) (from the stored `requested_position`).
   - Gaussian loss: maximise log-prob of the important position under the head:
     `L = -sum_k I_norm[k] * log N(pos_k | μ, σ)` where pos_k is the grid position of slot k.
   
   **Alternative (simpler, preferred):** Since the spatial head already chose these positions,
   use importance-weighted log-prob regression. For each spatial slot k with position pos_k:
   `L = -importance_norm[k] * dist.log_prob(pos_k)` where `dist = Normal(μ, σ)` from the head.
   This naturally pushes μ toward the important position and tightens σ around it. No 2D map needed.

**Caveat:** Use `torch.no_grad()` for importance computation. Stop-gradient on the target.

**Caveat:** The counterfactual forward passes should use `agent.module.get_value()` (not 
`get_action_and_value`) to save the cost of actor computation.

**Caveat:** Batch the K ablation forward passes into a single call: reshape (B*K, M_total, L, D)
and run one batched `get_value`.

**Validation:**
- `spatial_loss` decreases over training.
- In videos: green tiles start concentrating on task-relevant positions (e.g. the hint tile in
  MiniGrid-MemoryS9).
- Overall training still converges; the spatial loss is an auxiliary that shouldn't destabilise PPO.

---

### M7 — PPO on internal action
**Target:** April 10, 2026

**Goal:** Treat the spatial retrieval as a second action and apply a PPO loss on it.

**Changes:**

| File | Change |
|------|--------|
| `src_new/config.py` | Add `internal_action_coef`. |
| `src_new/trajectory.py` | Store `internal_log_probs` in `RolloutItem` and `Minibatch`. |
| `src_new/trainer.py` | **Rollout loop:** Store `action.internal_log_probs` alongside `action.external_log_probs`. |
| `src_new/trainer.py` | **Training loop:** Compute ratio, clipped loss for internal action. Add to total loss scaled by `internal_action_coef`. |

**PPO internal loss:**
```python
internal_logratio = new_internal_logprobs - mb.internal_log_probs
internal_ratio = torch.exp(internal_logratio)
internal_loss1 = -mb_advantages * internal_ratio
internal_loss2 = -mb_advantages * torch.clamp(
    internal_ratio, 1 - clip_coef, 1 + clip_coef
)
internal_pg_loss = torch.max(internal_loss1, internal_loss2).mean()
loss += train_conf.internal_action_coef * internal_pg_loss
```

**Caveat:** The same advantages are used for both external and internal actions. This is correct if
both actions influence the return. Alternatively, use separate advantage estimates if the spatial
action has its own reward signal (future work).

**Caveat:** Consider downweighting or removing the counterfactual loss (M6) once PPO on internal
actions is active, to avoid conflicting gradients.

**Validation:**
- Internal action log-probs appear in trajectory.
- The internal PPO loss is logged to W&B.
- The spatial head learns meaningful positions (converges faster than M5 alone).
- Episode returns improve compared to temporal-only baseline.

---

### Summary Timeline

| Milestone | Target Date | Risk Level | Key Deliverable |
|-----------|------------|------------|----------------|
| M1 | April 5 | Low | Spatial head in AgentModule, output discarded |
| M2 | April 5 | Low | Agent position stored in rollout buffer |
| M3 | April 6 | Low | SpatialMemory index implemented and tested |
| M4 | April 6 | Medium | Spatial retrieval pipeline wired, shapes verified |
| M5 | April 7 | **High** | Transformer consumes spatial tokens — full integration |
| M5.5 | April 7 | Low | Spatial tiles visible in rendered videos |
| M6 | April 8–9 | **High** | Counterfactual pretraining loss active |
| M7 | April 10 | Medium | PPO on internal action, full system |

**Total: ~6 working days** (April 5 – April 10, 2026)