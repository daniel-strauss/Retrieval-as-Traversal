so the plan needs these changes:
SpatialMemory -> it interprets the spatial head.
SpatialIndex -> it does what the plan currently says about SpatialMemory.
SpatialIndex now belongs to agent. The interaction agent <-> trainer <-> memory_handler stays exactly the same during rollout.
The internal action will be updated in the exact same way as external action.

the milestones should be this:
## M1. AgentView Rendering + PerceivedPosition
Agent calculates its offset position from start (which will be needed for spatial computing)
To verify this works we add an agent view rendering to our environment wrapper. The wrapper gets a class. AgentViewRenderer.
The wrapper passes data such as the agent perceived position and a AgentViewRenderer returns a frame. Then the wrapper will render both fromaes, side by side into one video

## M2. SpatialIndex
spatial index is implemented and tested.
As we dont need counterfactuals yet SpatialIndex is envronment step specific and thus belongs to agent. Later memory rollout buffer - when we compute counter factuals- will manage spatialindexing. But for now it belongs to agent.
Agent passes env steps for sanity checks. Spatial index gets at each timestep for all agents the positions that have been observed and stoers that fast. and then if the agent teslls the positions to retrieve it retrieves that fast.

This class should be tested with a unit test.

## M3. Combied sliding window.
We do not yet let the agent head decide for the position, but we do it manually.
The retrieved positions shall be shown in AgentViewRenderer.
For this we need to expand TemporalMemory to combine both outputs (the current view and the retrieved view) into the same sliding window.

we verify it by looking at the plots. manually adapting the retrieval position

## M4. SpatialMemory

We implement spatial memory that get the head output from the agent at returns a position. We test it in the same way as M3, but thistime by manually adapptung the agent memory retrieval head.

## M5. Training Logic

in the training loop we add with internal action the exact same thing as with external action

M6. We take the wheights from a previously trained network, reduce and train the new retrieval head with PPO whilst slowly reducing M. goal. get it to work with a small M.

-----------------------------------
-----------------------------------
# BELOW AI GENERATED "DETAILS"


-----------------------------------
-----------------------------------

-----------------------------------
-----------------------------------

-----------------------------------
-----------------------------------
## Plan: Implement Spatial Memory Retrieval (PPO-First)

#### TL;DR

Add spatial memory retrieval to the PPO+TrXL agent in 6 incremental milestones.
Start from a pretrained temporal-only model, add rendering / indexing / retrieval infrastructure, then train the retrieval head with PPO (same loss structure as external action) while gradually reducing the temporal window M.
No counterfactuals — those come later as a separate extension.

---

### §1  Architecture Overview

########## Two New Classes — Clear Separation of Concerns

| Class | Role | Location | Owned by |
|---|---|---|---|
| **`SpatialIndex`** | Pure data structure. Per-env dict `{(x,y) → list[env_step]}` | `src_new/model/retrieval/spatial_index.py` (rename from existing `spatial_indexing.py`) | `Agent` |
| **`SpatialMemory`** | Decision logic. Interprets Gaussian head output → discrete position → queries `SpatialIndex` → returns env_steps | `src_new/model/retrieval/spatial_memory.py` (existing stub) | `Agent` |

> **Why Agent owns SpatialIndex:** Without counterfactuals the index is episode-scoped (reset on done).
> There is no need to access past-episode indices during training.
> When counterfactuals are added, SpatialIndex migrates to MemoryHandler / rollout buffer.

######## Gaussian Spatial Heads (on AgentModule)

Replace the old commented-out categorical heads:

```python
#### --- Current code (agent_module.py L100-104) ---
## outcommented to not affect random seed TODO debug
####self.x_head: nn.Module
##self.y_head: nn.Module
####if use_spatial_memory:
##    self.x_head = layer_init(nn.Linear(trxl_dim, grid_w), np.sqrt(0.01))
##    self.y_head = layer_init(nn.Linear(trxl_dim, grid_h), np.sqrt(0.01))
```

with Gaussian heads:

```python
## --- New code ---
if use_spatial_memory:
    self.mu_head = layer_init(nn.Linear(trxl_dim, 2), np.sqrt(0.01))     ## → (μ_x, μ_y)
    self.log_sigma_head = nn.Linear(trxl_dim, 2)                          ## → (log σ_x, log σ_y)
    nn.init.constant_(self.log_sigma_head.bias, 1.0)                      ## σ ≈ 2.7 at init
```

During forward pass (replacing old categorical sampling):

```python
## --- Current code (agent_module.py L176-188) ---
## if self.use_spatial_memory:
##     row_dist = Categorical(logits=self.y_head(x))
##     col_dist = Categorical(logits=self.x_head(x))
##     ...
```

```python
## --- New code ---
if self.use_spatial_memory:
    mu = self.mu_head(x)                                          ## (N, 2)
    log_sigma = self.log_sigma_head(x).clamp(-2.3, math.log(max(grid_w, grid_h)))
    sigma = log_sigma.exp()                                       ## σ ∈ [~0.1, grid_size]
    dist = torch.distributions.Normal(mu, sigma)
    if internal_action_in is None:
        internal_action_in = dist.rsample()                       ## (N, 2) — differentiable
    internal_log_probs = dist.log_prob(internal_action_in)        ## (N, 2)
```

**Why Gaussian over Categorical:**
- Spatial inductive bias: nearby positions get similar probability, unlike a flat categorical over 81 cells
- Smooth gradients: Normal log-prob gradient ∝ (x - μ)/σ², vs hard argmax for categorical
- Compact: 4 outputs (μ_x, μ_y, log_σ_x, log_σ_y) vs 9×9 = 81
- Natural exploration: σ controls exploration radius directly

#### Spawn-Relative Coordinates

```
perceived_position = agent_pos - spawn_pos
```

- `spawn_pos` captured per-env on episode reset
- Positions can be negative (agent moves left/up from spawn)
- Reset `spawn_pos` on done

### #AgentViewRenderer

New class for side-by-side video rendering. Left panel = existing RF overlay. Right panel = top-down schematic with agent position dot + retrieved spatial position markers.

---

### §2  Existing Code — What We Build On

###### Agent.__init__ — The NotImplementedError to replace

```python
## agent.py L30-31
if train_conf.use_spatial_memory:
    raise NotImplementedError("TODO")
```

and:

```python
## agent.py L66-68
if train_conf.use_spatial_memory:
    ## TODO this doesnt belong here, see todo above
    raise NotImplementedError("TODO")
```

###### Action dataclass — Already has internal_action fields

```python
## agent_module.py L14-31
@dataclass
class Action:
    external_action: torch.Tensor            ## (N, num_branches)
    external_log_probs: torch.Tensor         ## same shape
    external_entropy: torch.Tensor           ## (N,)
    internal_action: torch.Tensor | None = None      ## (N, 2) or None
    internal_log_probs: torch.Tensor | None = None   ## (N, 2) or None
```

→ No changes needed to Action. Currently internal fields are always `None`.

###### SpatialIndexing — Already partly implemented

```python
## src_new/model/retrieval/spatial_indexing.py
class SpatialIndexing:
    def __init__(self, num_agents, grid_w, grid_h):
        ## Pre-allocates {(x,y): set()} for all grid cells per agent
        self.spatial_index: dict[int, dict[tuple[int, int], set[int]]] = ...

    def add_fov(self, fov: torch.Tensor, env_steps: torch.Tensor):
        ## Writes ALL visible cells from fov (N, grid_w, grid_h) mask
        for agent, x, y in zip(*torch.where(fov)):
            self.spatial_index[agent.item()][(x.item(), y.item())].add(int(env_steps[agent]))

    def restet_envs(self, dones: torch.Tensor):  ## typo in original
        ## Clears and reinitialises dicts for done envs

    def get_memory_indices_and_mask(self, positions, env_steps):
        ## Returns (N, k) indices + mask for variable-k results
```

**Issues with current SpatialIndexing:**
1. Uses absolute grid coords (pre-allocates all `{(x,y): set()}` for `grid_w × grid_h`). We need spawn-relative coords which can be negative → switch to plain `dict`, no pre-allocation.
2. `add_fov` writes entire FOV (all visible cells). We need a simpler `write(positions, env_steps)` that writes the agent's perceived position.
3. Typo: `restet_envs` → `reset_envs`
4. The `get_memory_indices_and_mask` returns variable-k. We need fixed-k (= `spatial_memory_k`) with most-recent-k selection.

→ **Rename to `SpatialIndex`, rewrite with spawn-relative dict approach.**

###### TemporalMemory (MemorySlider) — Templates for sliding window

```python
## temporal_memory.py — _build_templates()
## memory_mask_template: lower-triangular (M, M), e.g. for M=6:
##   0 0 0 0 0 0
##   1 0 0 0 0 0
##   1 1 0 0 0 0
##   1 1 1 0 0 0
##   1 1 1 1 0 0
##   1 1 1 1 1 0

## memory_indices_template: sliding window (max_ep_steps, M), e.g. M=4, max=7:
##   0 1 2 3   (step 0, window not full yet)
##   0 1 2 3   (step 1)
##   0 1 2 3   (step 2)
##   0 1 2 3   (step 3, window just filled)
##   1 2 3 4   (step 4, sliding)
##   2 3 4 5
##   3 4 5 6
```

→ Need to add `get_combined_indices_and_mask()` that reserves S slots for spatial tokens.

###### Trainer rollout loop — Where perceived_position + writes go

```python
## trainer.py L262-271 (inside rollout_step loop)
action, value, memory_write_record = agent.sample_action(
    obs=current_obs,
    episode_steps_envs=envs_t.clone(),
    fov=fov,
    memory_window=memory_window,
)

external_action = action.external_action
external_logprob = action.external_log_probs
```

```python
## trainer.py L285 — capture_frame before env.step
envs.envs[0].capture_frame(
    env_t=envs_t[0].item(),
    rf_scaled=rf_scaled_env0[-1].cpu().numpy(),
    rf_binary=rf_binary_env0[-1].cpu().numpy(),
)
```

```python
## trainer.py L330-334 — done handling
envs_t += 1
for id, done_i in enumerate(done):
    if done_i:
        envs_t[id] = 0
```

→ Insert `spawn_pos` tracking, `SpatialIndex.write`, and `perceived_pos` passing to capture_frame here.

###### Trainer training loop — Where internal PPO loss goes

```python
## trainer.py L469-480 — currently only trains external action
result, newvalue, _, _ = agent.module.get_action_and_value(
    x=mb.obs,
    memory_frames=mb_memory_window.pure_memory_frames,
    memory_mask=mb_memory_window.memory_masks,
    memory_indices=mb_memory_window.memory_indices_env,
    action=Action(
        external_action=mb.actions,
        external_log_probs=torch.empty(0, device=device),
        external_entropy=torch.empty(0, device=device),
    ),
)

external_newlogprob = result.external_log_probs
entropy = result.external_entropy

## ... PPO loss computed only on external action ...
logratio = newlogprob - mb.log_probs
ratio = torch.exp(logratio)
pgloss1 = -mb_advantages * ratio
pgloss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
metrics.pg_loss = torch.max(pgloss1, pgloss2).mean()
```

→ The `Action` passed to `get_action_and_value` currently has no `internal_action`. Will need to pass
`mb.internal_actions` and compute a parallel PPO loss on `result.internal_log_probs`.

###### RolloutItem / Minibatch — Missing internal_action fields

```python
## trajectory.py — RolloutItem fields (currently):
self.rewards       ## (n, m)
self.actions       ## (n, m, len(action_space_shape))
self.dones         ## (n, m)
self.values        ## (n, m)
self.log_probs     ## (n, m, len(action_space_shape))
self.obs           ## (n, m, *obs_shape)
self.global_steps  ## (n, m)
self.mem_retrieval_steps  ## (n, m)
self.envs_t        ## (n, m)
## ← NO internal_action, NO internal_log_probs
```

→ Need to add `internal_actions: torch.Tensor | None` (n, m, 2) and `internal_log_probs: torch.Tensor | None` (n, m, 2).

###### Config — Spatial fields to add

```python
## config.py — current spatial config:
use_spatial_memory: bool = False
```

→ Need to add: `spatial_memory_k`, `internal_action_coef`, `spatial_warmup_steps`.

###### CustomRecordVideo.capture_frame — Current signature

```python
## custom_env_wrappers.py L243-259
def capture_frame(self, env_t: int, rf_scaled: np.ndarray, rf_binary: np.ndarray):
    ...
    self._capture_frame(rf_bin_for_render=rf_binary, rf_scaled_for_render=rf_scaled)
```

→ Extend to accept `perceived_position` and `retrieved_positions`, pass to AgentViewRenderer, compose side-by-side.

---

## §3  Milestones

### M1: AgentView Rendering + PerceivedPosition

**Goal:** Agent computes its spawn-relative position. Verify via a side-by-side video: left = existing RF overlay, right = top-down schematic showing perceived position.

#### Changes

**`src_new/trainer.py`** — Add `spawn_pos` tracking:

```python
## After envs.reset():
spawn_pos = torch.zeros((train_conf.num_envs, 2), dtype=torch.long, device=device)
for i, env in enumerate(envs.envs):
    pos = env.unwrapped.agent_pos  ## (x, y) tuple
    spawn_pos[i] = torch.tensor(pos, dtype=torch.long, device=device)

## Inside rollout loop, before sample_action:
current_agent_positions = torch.stack([
    torch.tensor(env.unwrapped.agent_pos, dtype=torch.long, device=device)
    for env in envs.envs
])
perceived_pos = current_agent_positions - spawn_pos  ## (N, 2), can be negative

## In done handling:
for id, done_i in enumerate(done):
    if done_i:
        envs_t[id] = 0
        pos = envs.envs[id].unwrapped.agent_pos
        spawn_pos[id] = torch.tensor(pos, dtype=torch.long, device=device)
```

**`src_new/env/agent_view_renderer.py`** — New file:

```python
class AgentViewRenderer:
    """Renders a top-down schematic of the grid with position annotations.

    Returns an RGB numpy frame showing:
    - Agent perceived position (green dot)
    - Retrieved spatial positions (orange dots), if provided
    """

    def __init__(self, grid_w: int, grid_h: int, tile_size: int = 28):
        ...

    def render(self,
               perceived_position: tuple[int, int],
               retrieved_positions: list[tuple[int, int]] | None = None,
               ) -> np.ndarray:
        """Return an RGB frame (H, W, 3) with position annotations."""
        ...
```

**`src_new/env/custom_env_wrappers.py`** — Extend `capture_frame` signature:

```python
## Before:
def capture_frame(self, env_t, rf_scaled, rf_binary):

## After:
def capture_frame(self, env_t, rf_scaled, rf_binary,
                  perceived_position=None, retrieved_positions=None):
    ...
    if self.agent_view_renderer is not None and perceived_position is not None:
        agent_view_frame = self.agent_view_renderer.render(perceived_position, retrieved_positions)
        overlay_frame = np.concatenate([overlay_frame, agent_view_frame], axis=1)
```

**`src_new/trainer.py`** — Pass position to capture_frame:

```python
envs.envs[0].capture_frame(
    env_t=envs_t[0].item(),
    rf_scaled=rf_scaled_env0[-1].cpu().numpy(),
    rf_binary=rf_binary_env0[-1].cpu().numpy(),
    perceived_position=tuple(perceived_pos[0].cpu().tolist()),
)
```

**Verification:** Run short training, inspect video. Right panel shows green dot at agent position. Position resets to (0,0) on episode start, moves correctly relative to spawn.

---

###### M2: SpatialIndex

**Goal:** `SpatialIndex` stores and retrieves `position→env_step` mappings. Fully unit-tested.

######## Changes

**`src_new/model/retrieval/spatial_index.py`** — New file (replaces `spatial_indexing.py`):

```python
class SpatialIndex:
    """Per-env dict mapping (x,y) → list[env_step].

    Uses spawn-relative coordinates (can be negative).
    No pre-allocation — positions are added dynamically.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._index: list[dict[tuple[int, int], list[int]]] = [
            {} for _ in range(num_envs)
        ]

    def write(self, env_ids: torch.Tensor, positions: torch.Tensor, env_steps: torch.Tensor):
        """Batch write: for each env, append env_step to dict[(x,y)].

        Args:
            env_ids: (K,) env indices
            positions: (K, 2) spawn-relative (x, y) — can be negative
            env_steps: (K,) current episode steps
        """
        for i in range(env_ids.shape[0]):
            eid = int(env_ids[i])
            pos = (int(positions[i, 0]), int(positions[i, 1]))
            if pos not in self._index[eid]:
                self._index[eid][pos] = []
            self._index[eid][pos].append(int(env_steps[i]))

    def read(self, env_id: int, position: tuple[int, int], k: int = 1) -> list[int]:
        """Return most recent k env_steps at position. Empty list if unvisited."""
        steps = self._index[env_id].get(position, [])
        return steps[-k:]  ## most recent k

    def reset(self, env_ids: torch.Tensor):
        """Clear dicts for done envs."""
        for eid in env_ids.tolist():
            self._index[eid] = {}
```

**`src_new/model/agent.py`** — Add SpatialIndex to Agent:

```python
## Replace NotImplementedError with:
if train_conf.use_spatial_memory:
    from src_new.model.retrieval.spatial_index import SpatialIndex
    self.spatial_index = SpatialIndex(num_envs=train_conf.num_envs)
```

**`src_new/trainer.py`** — Wire writes in rollout loop:

```python
## After computing perceived_pos (from M1):
if train_conf.use_spatial_memory:
    all_env_ids = torch.arange(train_conf.num_envs, device=device)
    agent.spatial_index.write(all_env_ids, perceived_pos, envs_t)

## In done handling (after envs_t reset):
if train_conf.use_spatial_memory:
    done_ids = torch.where(done)[0]
    if done_ids.numel() > 0:
        agent.spatial_index.reset(done_ids)
```

**`tests/tests_milestones_spatial_retrieval/test_spatial_index.py`** — Unit tests:

- Write + read single position → correct env_step returned
- Write multiple steps to same position → `read(k=1)` returns most recent, `read(k=3)` returns last 3
- Read unvisited position → empty list
- Batch write across multiple envs → each env's data isolated
- Reset clears only specified envs
- Negative coordinates (spawn-relative) work correctly

**Verification:** All unit tests pass.

---

###### M3: Combined Sliding Window

**Goal:** Spatial retrieval positions enter the transformer's memory window alongside temporal tokens. Manual position selection (no head yet). Verify via video.

######## Changes

**`src_new/model/retrieval/temporal_memory.py`** — Add combined window method:

```python
def get_combined_indices_and_mask(
    self,
    episode_steps_envs: torch.Tensor,     ## (N,)
    spatial_env_steps: torch.Tensor,       ## (N, S) — env steps from SpatialIndex
    spatial_valid_mask: torch.Tensor,      ## (N, S) — which spatial slots are valid
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a combined window: (M-S) temporal + S spatial slots.

    Returns:
        combined_indices: (N, M) — env_step indices into memory buffer
        combined_mask: (N, M) — attention mask (True = attend)

    Layout: [temporal_0, ..., temporal_{M-S-1}, spatial_0, ..., spatial_{S-1}]
    Temporal slots use the causal sliding window mask.
    Spatial slots use spatial_valid_mask (always attend if the read returned data).
    """
    S = spatial_env_steps.shape[1]
    M_temporal = self.trxl_memory_length - S

    ## Get temporal part using existing template, but only first M_temporal columns
    ## ... (truncate template to M-S columns)

    ## Concatenate: [temporal_indices, spatial_env_steps] → (N, M)
    ## Concatenate: [temporal_mask, spatial_valid_mask] → (N, M)
    ...
```

**`src_new/model/retrieval/spatial_memory.py`** — Add manual query method:

```python
class SpatialMemory:
    """Interprets head output → grid position → queries SpatialIndex."""

    def __init__(self, spatial_index: SpatialIndex, k: int = 1):
        self.spatial_index = spatial_index
        self.k = k

    def query_manual(
        self, env_ids: torch.Tensor, position: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Manual mode: hardcoded position for testing.

        Returns:
            spatial_env_steps: (N, k) env_step indices
            spatial_valid_mask: (N, k) bool mask
        """
        N = env_ids.shape[0]
        spatial_env_steps = torch.zeros(N, self.k, dtype=torch.long)
        spatial_valid_mask = torch.zeros(N, self.k, dtype=torch.bool)
        for i in range(N):
            steps = self.spatial_index.read(int(env_ids[i]), position, k=self.k)
            for j, s in enumerate(steps):
                spatial_env_steps[i, j] = s
                spatial_valid_mask[i, j] = True
        return spatial_env_steps, spatial_valid_mask
```

**`src_new/model/agent.py`** — Wire combined window in `sample_action`:

```python
## In sample_action, replace:
memory_indices_next, memory_masks_next = self.temporal_memory.get_memory_indices_and_mask(
    episode_steps_envs + 1
)

## With (when use_spatial_memory):
if self.args.use_spatial_memory:
    spatial_steps, spatial_mask = self.spatial_memory.query_manual(
        env_ids=..., position=self._debug_retrieval_position
    )
    memory_indices_next, memory_masks_next = self.temporal_memory.get_combined_indices_and_mask(
        episode_steps_envs + 1, spatial_steps, spatial_mask
    )
else:
    memory_indices_next, memory_masks_next = self.temporal_memory.get_memory_indices_and_mask(
        episode_steps_envs + 1
    )
```

**`src_new/env/agent_view_renderer.py`** — Show retrieved positions as orange dots.

**`src_new/trainer.py`** — Pass retrieved positions to `capture_frame`.

**Verification:** Run with `use_spatial_memory=True` and a manually chosen position. Video right panel shows green dot (agent) + orange dot (retrieval target). Manually vary the position, confirm different memories are attended to.

---

###### M4: Gaussian Retrieval Head (SpatialMemory)

**Goal:** SpatialMemory interprets Gaussian head output → position. Wire to actual neural network head.

######## Changes

**`src_new/model/agent_module.py`** — Implement Gaussian heads (see §1 for code).

Replace old commented-out code with `mu_head` + `log_sigma_head`.
In `get_action_and_value`, compute Normal distribution, sample, store in `Action.internal_action` and `Action.internal_log_probs`.

**`src_new/model/retrieval/spatial_memory.py`** — Add head interpretation:

```python
def interpret_head_output(
    self,
    env_ids: torch.Tensor,
    internal_action: torch.Tensor,       ## (N, 2) — continuous from Gaussian head
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretise head output → query SpatialIndex.

    Args:
        internal_action: continuous (x, y) in spawn-relative coords

    Returns:
        spatial_env_steps: (N, k) env_step indices
        spatial_valid_mask: (N, k) bool mask
    """
    grid_pos = internal_action.round().long()  ## (N, 2)

    N = env_ids.shape[0]
    spatial_env_steps = torch.zeros(N, self.k, dtype=torch.long)
    spatial_valid_mask = torch.zeros(N, self.k, dtype=torch.bool)
    for i in range(N):
        pos = (int(grid_pos[i, 0]), int(grid_pos[i, 1]))
        steps = self.spatial_index.read(int(env_ids[i]), pos, k=self.k)
        for j, s in enumerate(steps):
            spatial_env_steps[i, j] = s
            spatial_valid_mask[i, j] = True
    return spatial_env_steps, spatial_valid_mask
```

**`src_new/model/agent.py`** — Replace manual query (M3) with head-driven query:

```python
## In sample_action (when use_spatial_memory):
internal_action = action.internal_action  ## from module.get_action_and_value
spatial_steps, spatial_mask = self.spatial_memory.interpret_head_output(
    env_ids=all_env_ids, internal_action=internal_action
)
memory_indices_next, memory_masks_next = self.temporal_memory.get_combined_indices_and_mask(
    episode_steps_envs + 1, spatial_steps, spatial_mask
)
```

**`src_new/model/agent.py`** — Wire `evaluate_actions` to return `internal_log_probs`:

```python
## evaluate_actions already returns result.internal_log_probs (4th return value):
def evaluate_actions(...):
    ...
    return result.external_log_probs, result.external_entropy, value, result.internal_log_probs
```

→ Currently this returns `None` for `internal_log_probs`. After M4, it returns actual values when `use_spatial_memory=True`.

**Verification:** Run with head. Initially random positions (untrained). Video shows jumping dots. Log μ and σ to tensorboard — σ starts ≈2.7 (broad exploration). `Action.internal_action` and `.internal_log_probs` are non-None.

---

###### M5: PPO Training for Internal Action

**Goal:** Train the retrieval head with PPO using the same advantages as the external action.

######## Changes

**`src_new/trajectory.py`** — Add internal action fields:

```python
@dataclass
class RolloutItem:
    ...
    ## Add:
    internal_actions: torch.Tensor | None = None      ## (n, m, 2) or None
    internal_log_probs: torch.Tensor | None = None    ## (n, m, 2) or None

    @staticmethod
    def get_empty(n, m, action_space_shape, observation_space_shape, device,
                  use_spatial_memory=False):
        ...
        internal_actions = torch.empty((n, m, 2), dtype=torch.float, device=device) \
            if use_spatial_memory else None
        internal_log_probs = torch.empty((n, m, 2), dtype=torch.float, device=device) \
            if use_spatial_memory else None
        return RolloutItem(..., internal_actions=internal_actions,
                          internal_log_probs=internal_log_probs)

@dataclass
class Minibatch:
    ...
    ## Add:
    internal_actions: torch.Tensor | None = None      ## (B, 2) or None
    internal_log_probs: torch.Tensor | None = None    ## (B, 2) or None
```

**`src_new/trainer.py`** — Store internal action in rollout loop:

```python
## After sample_action:
trajy.store_rollout_step(
    ...,
    internal_action=action.internal_action,        ## NEW
    internal_log_prob=action.internal_log_probs,   ## NEW
)
```

**`src_new/trainer.py`** — Add internal PPO loss in training loop:

```python
## In the minibatch training loop, after external PPO loss:

## Pass internal_action for re-evaluation
action=Action(
    external_action=mb.actions,
    external_log_probs=torch.empty(0, device=device),
    external_entropy=torch.empty(0, device=device),
    internal_action=mb.internal_actions,     ## NEW — for re-evaluation
)

## After getting result:
if train_conf.use_spatial_memory and result.internal_log_probs is not None:
    internal_logratio = result.internal_log_probs - mb.internal_log_probs  ## (B, 2)
    internal_ratio = torch.exp(internal_logratio)

    ## Use same advantages (shared reward signal)
    mb_adv_internal = mb.advantages.unsqueeze(1).repeat(1, 2)

    internal_pgloss1 = -mb_adv_internal * internal_ratio
    internal_pgloss2 = -mb_adv_internal * torch.clamp(
        internal_ratio, 1.0 - train_conf.clip_coef, 1.0 + train_conf.clip_coef
    )
    internal_pg_loss = torch.max(internal_pgloss1, internal_pgloss2).mean()

    metrics.loss += train_conf.internal_action_coef * internal_pg_loss
```

**`src_new/config.py`** — Add config fields:

```python
spatial_memory_k: int = 1
"""number of spatial retrieval slots in the memory window"""
internal_action_coef: float = 1.0
"""weight of internal action PPO loss"""
```

**`src_new/trainer.py`** — Log internal metrics:

```python
if train_conf.use_spatial_memory:
    writer.add_scalar("losses/internal_pg_loss", internal_pg_loss.item(), global_step)
    ## Log μ and σ from the head
```

**Verification:**
- `internal_pg_loss` is logged and non-zero
- `internal_approx_kl` stays reasonable (no policy collapse)
- σ decreases over training (agent becomes more certain about retrieval positions)
- No NaN in any loss terms
- External returns are not degraded compared to baseline

---

###### M6: Pretrained Model + Reduced Window

**Goal:** Load pretrained temporal-only weights, reduce M, train spatial head to compensate.

######## Changes

**`src_new/utils.py` or `src_new/trainer.py`** — Weight loading:

```python
def load_pretrained_weights(agent_module: AgentModule, checkpoint_path: str):
    """Load pretrained checkpoint, skip spatial heads (randomly initialized)."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    ## Filter out spatial head keys that don't exist in checkpoint
    missing, unexpected = agent_module.load_state_dict(state_dict, strict=False)
    ## Expect missing: mu_head.weight, mu_head.bias, log_sigma_head.weight, log_sigma_head.bias
    ...
```

**`src_new/config.py`** — Add warmup config:

```python
spatial_warmup_steps: int = 0
"""global steps during which spatial slots are not used in the window (head still trains).
After warmup: M is reduced by spatial_memory_k and spatial slots take their place."""
pretrained_checkpoint: str = ""
"""path to pretrained temporal-only checkpoint for transfer learning"""
```

**`src_new/trainer.py`** — Gradual M reduction:

```python
## At training start:
if train_conf.pretrained_checkpoint:
    load_pretrained_weights(agent.module, train_conf.pretrained_checkpoint)

## In rollout loop, check warmup:
use_spatial_in_window = (
    train_conf.use_spatial_memory
    and global_count >= train_conf.spatial_warmup_steps
)
```

**Optional: Freeze encoder/transformer during warmup:**

```python
if global_count < train_conf.spatial_warmup_steps:
    for name, param in agent.module.named_parameters():
        if "mu_head" not in name and "log_sigma_head" not in name:
            param.requires_grad = False
else:
    for param in agent.module.parameters():
        param.requires_grad = True
```

**Verification:**
- Load pretrained model (temporal-only, high M), run with reduced M + spatial retrieval
- Compare returns vs temporal-only baseline at same reduced M → spatial should outperform
- σ converges to meaningful values (decreases)
- Video shows agent retrieving task-relevant positions
- Training is stable (no loss spikes from weight transfer)

---

#### §4  File Change Summary

| File | M1 | M2 | M3 | M4 | M5 | M6 | Change type |
|------|----|----|----|----|----|----|-------------|
| `src_new/trainer.py` | ✓ | ✓ | ✓ | | ✓ | ✓ | spawn_pos, index writes, internal loss, weight loading |
| `src_new/model/agent.py` | | ✓ | ✓ | ✓ | | | Remove NotImplementedError, own SpatialIndex, wire queries |
| `src_new/model/agent_module.py` | | | | ✓ | | | mu_head, log_sigma_head, Normal sampling |
| `src_new/model/retrieval/spatial_index.py` | | ✓ | | | | | **New.** Replaces spatial_indexing.py |
| `src_new/model/retrieval/spatial_memory.py` | | | ✓ | ✓ | | | Rewrite stub: manual query → head interpretation |
| `src_new/model/retrieval/temporal_memory.py` | | | ✓ | | | | get_combined_indices_and_mask() |
| `src_new/env/agent_view_renderer.py` | ✓ | | ✓ | | | | **New.** Top-down schematic renderer |
| `src_new/env/custom_env_wrappers.py` | ✓ | | | | | | Accept renderer, side-by-side frames |
| `src_new/trajectory.py` | | | | | ✓ | | internal_actions, internal_log_probs fields |
| `src_new/config.py` | | | | | ✓ | ✓ | spatial_memory_k, internal_action_coef, warmup |
| `tests/.../test_spatial_index.py` | | ✓ | | | | | **New.** Unit tests |

---

#### §5  Key Decisions

1. **SpatialIndex belongs to Agent** (not MemoryHandler) — episode-scoped, no cross-episode access needed without counterfactuals. Migrates to MemoryHandler when counterfactuals are added.
2. **No counterfactuals** in this plan — PPO-only training. Counterfactuals can be added as a later extension by rebuilding SpatialIndex snapshots from the rollout buffer.
3. **Same advantages for internal and external action** — shared reward signal. Both actions contribute to the same outcome. Separate advantage estimation is possible but unnecessarily complex at this stage.
4. **Gaussian heads over categorical** — spatial inductive bias, smooth gradients, 4 outputs vs 81.
5. **Spawn-relative coordinates** — `perceived_pos = agent_pos - spawn_pos`. Can be negative. SpatialIndex uses plain dict (no pre-allocation of grid).
6. **Combined sliding window** — S spatial tokens replace the S oldest temporal slots in the M-sized window. Spatial tokens are always unmasked (attended) when valid.
7. **`perceived_position` computed in Trainer** — Trainer has access to `env.unwrapped.agent_pos`. Passed to Agent for SpatialIndex writes.
8. **`spatial_indexing.py` renamed to `spatial_index.py`** — Cleaner naming, simpler interface (write position instead of entire FOV), spawn-relative coords.

#### §6  Open Questions

1. **How many spatial slots (S)?** Start with S=1. Config: `spatial_memory_k`.
2. **Should internal action entropy bonus be separate?** Start with shared `ent_coef`. Add `internal_ent_coef` later if needed.
3. **M3 manual position:** Use a config parameter `debug_spatial_retrieval_offset: tuple[int,int]` relative to perceived_pos, or hardcoded absolute position?
4. **What happens when SpatialIndex.read returns empty** (unvisited position)? The spatial slot mask is False → transformer ignores that slot → effectively one fewer memory token. This is fine and self-correcting as σ narrows.