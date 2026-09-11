"""
Test that the original and new implementations produce identical minibatches
in the first training iteration (rollout + GAE + minibatch construction).

This does NOT run training (backward passes). It only verifies that the
*inputs* to the first training loop are bit-for-bit identical.

Run:
    cd /home/dstrauss/Documents/uni/master_thesis/sandbox/ppo_trxl
    conda activate clean_rl
    python -m pytest src_new/tests/test_identical_minibatches.py -v -s
"""

print("Starting test_identical_minibatches.py...", flush=True)

import warnings

warnings.filterwarnings("ignore")
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameters (match the production config)
# ═══════════════════════════════════════════════════════════════════════════════
N = 16  # num_envs
T = 256  # num_rollout_steps (num_steps)
M = 64  # trxl_memory_length
L = 2  # trxl_num_layers
D = 256  # trxl_dim
H = 4  # trxl_num_heads
SEED = 1
MAX_EP = 96  # max episode steps for MiniGrid-MemoryS9-v0
DEVICE = torch.device("cpu")
NUM_MINIBATCHES = 8
UPDATE_EPOCHS = 3
CLIP_COEF = 0.2
VF_COEF = 0.5
GAMMA = 0.995
GAE_LAMBDA = 0.95
ANNEAL_STEPS = 4096000
INIT_LR = 2.75e-4
FINAL_LR = 1e-5
INIT_ENT_COEF = 0.0001
FINAL_ENT_COEF = 0.000001
MAX_GRAD_NORM = 0.25
BATCH_SIZE = N * T
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def batched_index_select(inp, dim, index):
    """Exact copy from original_implementation.py."""
    for ii in range(1, len(inp.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(inp, dim, index)


def build_templates(mem_len, max_ep):
    """Build memory mask and index templates (same as original)."""
    mask = torch.tril(torch.ones((mem_len, mem_len)), diagonal=-1)
    reps = torch.repeat_interleave(torch.arange(mem_len).unsqueeze(0), mem_len - 1, dim=0).long()
    idx = torch.stack([torch.arange(i, i + mem_len) for i in range(max_ep - mem_len + 1)]).long()
    idx = torch.cat((reps, idx))
    return mask, idx


# ═══════════════════════════════════════════════════════════════════════════════
# Run original: rollout → GAE → flatten → capture minibatch inputs
# ═══════════════════════════════════════════════════════════════════════════════


def run_original_first_iteration():
    """Run the original implementation for one full rollout + GAE.

    Returns all the flat batched arrays and the RNG state right before the
    training loop so we can reproduce the exact randperm sequence.
    """
    from original_imeplentation_adapted import Agent, Args, make_env

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    envs = gym.vector.SyncVectorEnv(
        [make_env("MiniGrid-MemoryS9-v0", i, False, "test") for i in range(N)]
    )
    asp = (envs.single_action_space.n,)

    args = Args()
    args.cuda = False
    args.num_envs = N
    args.num_steps = T
    args.trxl_num_layers = L
    args.trxl_dim = D
    args.trxl_num_heads = H
    args.trxl_memory_length = M
    args.trxl_positional_encoding = "absolute"
    args.reconstruction_coef = 0.0
    args.batch_size = BATCH_SIZE
    args.minibatch_size = MINIBATCH_SIZE
    args.num_minibatches = NUM_MINIBATCHES
    args.update_epochs = UPDATE_EPOCHS
    args.clip_coef = CLIP_COEF
    args.vf_coef = VF_COEF
    args.clip_vloss = True
    args.norm_adv = False
    args.max_grad_norm = MAX_GRAD_NORM
    args.init_ent_coef = INIT_ENT_COEF
    args.final_ent_coef = FINAL_ENT_COEF
    args.init_lr = INIT_LR
    args.final_lr = FINAL_LR
    args.gamma = GAMMA
    args.gae_lambda = GAE_LAMBDA
    args.anneal_steps = ANNEAL_STEPS

    agent = Agent(args, envs.single_observation_space, asp, MAX_EP)
    init_state = {k: v.clone() for k, v in agent.state_dict().items()}
    optimizer = torch.optim.AdamW(agent.parameters(), lr=INIT_LR)

    mask_tpl, idx_tpl = build_templates(M, MAX_EP)

    # ── Rollout storage ──────────────────────────────────────────────────
    env_step = torch.zeros(N, dtype=torch.long)
    global_step = 0
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(N)
    next_memory = torch.zeros(N, MAX_EP, L, D)

    rewards = torch.zeros(T, N)
    actions = torch.zeros(T, N, len(asp), dtype=torch.long)
    dones = torch.zeros(T, N)
    obs = torch.zeros((T, N) + envs.single_observation_space.shape)
    log_probs = torch.zeros(T, N, len(asp))
    values = torch.zeros(T, N)
    sm_masks = torch.zeros(T, N, M, dtype=torch.bool)
    sm_index = torch.zeros(T, N, dtype=torch.long)
    sm_indices = torch.zeros(T, N, M, dtype=torch.long)

    # ── Annealing (iteration 1) ──────────────────────────────────────────
    do_anneal = ANNEAL_STEPS > 0 and global_step < ANNEAL_STEPS
    frac = 1 - global_step / ANNEAL_STEPS if do_anneal else 0
    lr = (INIT_LR - FINAL_LR) * frac + FINAL_LR
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    ent_coef = (INIT_ENT_COEF - FINAL_ENT_COEF) * frac + FINAL_ENT_COEF

    # Init memory index
    stored_memories = [next_memory[e] for e in range(N)]
    for e in range(N):
        sm_index[:, e] = e

    # ── Rollout ──────────────────────────────────────────────────────────
    for step in range(T):
        global_step += N

        with torch.no_grad():
            obs[step] = next_obs
            dones[step] = next_done
            sm_masks[step] = mask_tpl[torch.clip(env_step, 0, M - 1)]
            sm_indices[step] = idx_tpl[env_step]
            mw = batched_index_select(next_memory, 1, sm_indices[step])

            action, lp, _, val, new_m = agent.get_action_and_value(
                next_obs, mw, sm_masks[step], sm_indices[step]
            )
            next_memory[range(N), env_step] = new_m
            actions[step], log_probs[step], values[step] = action, lp, val

        next_obs, rew, term, trunc, infos = envs.step(action.cpu().numpy())
        next_done = torch.Tensor(np.logical_or(term, trunc))
        rewards[step] = torch.tensor(rew).view(-1)
        next_obs = torch.Tensor(next_obs)

        for eid in range(N):
            if next_done[eid]:
                env_step[eid] = 0
                stored_memories[sm_index[step, eid]] = stored_memories[sm_index[step, eid]].clone()
                next_memory[eid] = torch.zeros(MAX_EP, L, D)
                if step < T - 1:
                    stored_memories.append(next_memory[eid])
                    sm_index[step + 1 :, eid] = len(stored_memories) - 1
            else:
                env_step[eid] += 1

    # ── Bootstrap + GAE ──────────────────────────────────────────────────
    with torch.no_grad():
        start = torch.clip(env_step - M, 0)
        end = torch.clip(env_step, M)
        bv_idx = torch.stack([torch.arange(start[b], end[b]) for b in range(N)]).long()
        bv_win = batched_index_select(next_memory, 1, bv_idx)
        nv = agent.get_value(
            next_obs,
            bv_win,
            mask_tpl[torch.clip(env_step, 0, M - 1)],
            sm_indices[-1],
        )
        adv = torch.zeros_like(rewards)
        lg = 0
        for t in reversed(range(T)):
            nnt = (1 - next_done) if t == T - 1 else (1 - dones[t + 1])
            nvt = nv if t == T - 1 else values[t + 1]
            d = rewards[t] + GAMMA * nvt * nnt - values[t]
            adv[t] = lg = d + GAMMA * GAE_LAMBDA * nnt * lg
        rets = adv + values

    # ── Flatten ──────────────────────────────────────────────────────────
    b_obs = obs.reshape(-1, *obs.shape[2:])
    b_lp = log_probs.reshape(-1, *log_probs.shape[2:])
    b_act = actions.reshape(-1, *actions.shape[2:])
    b_adv = adv.reshape(-1)
    b_ret = rets.reshape(-1)
    b_val = values.reshape(-1)
    b_mi = sm_index.reshape(-1)
    b_mindices = sm_indices.reshape(-1, M)
    b_mmask = sm_masks.reshape(-1, M)
    stored_memories_stacked = torch.stack(stored_memories, 0)

    actual_max = int((sm_indices * sm_masks).max().item()) + 1
    if actual_max < M:
        b_mindices = b_mindices[:, :actual_max]
        b_mmask = b_mmask[:, :actual_max]
        stored_memories_stacked = stored_memories_stacked[:, :actual_max]

    # ── Capture RNG state before training ────────────────────────────────
    rng_state = torch.random.get_rng_state()

    envs.close()

    return {
        "init_state": init_state,
        "b_obs": b_obs,
        "b_logprobs": b_lp,
        "b_actions": b_act,
        "b_advantages": b_adv,
        "b_returns": b_ret,
        "b_values": b_val,
        "b_memory_index": b_mi,
        "b_memory_indices": b_mindices,
        "b_memory_mask": b_mmask,
        "stored_memories": stored_memories_stacked,
        "actual_max": actual_max,
        "ent_coef": ent_coef,
        "rng_state": rng_state,
        "values_2d": values.clone(),
        "advantages_2d": adv.clone(),
        "returns_2d": rets.clone(),
        "bootstrap_value": nv.clone(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Run new: rollout → GAE → capture minibatch inputs
# ═══════════════════════════════════════════════════════════════════════════════


def run_new_first_iteration(orig_init_state):
    """Run the new implementation for one full rollout + GAE.

    Loads original weights so both start from the same parameters.
    Returns the Trajectory and MemoryHandler so we can reconstruct
    minibatches in the test, plus the RNG state before training.
    """
    from src_new.memory.handler import MemoryHandler
    from src_new.model.agent import Agent
    from src_new.trainer.trainer import make_env
    from src_new.trainer.trajectory import Trajectory
    from src_new.utils import get_visible_cells

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    envs = gym.vector.SyncVectorEnv(
        [make_env("MiniGrid-MemoryS9-v0", i, False, "test") for i in range(N)]
    )
    asp = (envs.single_action_space.n,)
    GW = GH = 9

    class FC:
        trxl_memory_length = M
        trxl_num_layers = L
        trxl_dim = D
        trxl_num_heads = H
        trxl_positional_encoding = "absolute"
        use_spatial_memory = False
        reconstruction_coef = 0.0
        num_envs = N
        num_rollout_steps = T

    agent = Agent(
        train_conf=FC(),
        observation_space=envs.single_observation_space,
        action_space_shape=asp,
        max_episode_steps=MAX_EP,
        grid_w=GW,
        grid_h=GH,
        device=DEVICE,
        plot_rf_env0=False,
    )
    agent.gourmet_mode = False

    # Load original weights
    sd = agent.module.state_dict()
    for k, v in orig_init_state.items():
        nk = (
            "transformer.pos_embedding_a.inv_freqs"
            if k == "transformer.pos_embedding.inv_freqs"
            else k
        )
        if nk in sd:
            sd[nk] = v.clone()
    agent.module.load_state_dict(sd)

    mh = MemoryHandler(
        num_envs=N,
        num_rollout_steps=T,
        max_episode_steps=MAX_EP,
        trxl_memory_length=M,
        trxl_num_layers=L,
        trxl_dim=D,
        grid_w=GW,
        grid_h=GH,
        device=DEVICE,
    )

    trajy = Trajectory(
        num_envs=N,
        rollout_steps=T,
        action_space_shape=asp,
        observation_space_shape=envs.single_observation_space.shape,
        device=DEVICE,
    )

    envs_t = torch.zeros(N, dtype=torch.long, device=DEVICE)
    gs = 0
    gc = 0
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.as_tensor(next_obs, device=DEVICE)
    done = torch.zeros(N, dtype=torch.bool, device=DEVICE)

    # Annealing (iteration 1, global_count = 0)
    do_anneal = ANNEAL_STEPS > 0 and gc < ANNEAL_STEPS
    frac = 1 - gc / ANNEAL_STEPS if do_anneal else 0
    ent_coef = (INIT_ENT_COEF - FINAL_ENT_COEF) * frac + FINAL_ENT_COEF

    trajy.new_rollout()

    # ── Rollout ──────────────────────────────────────────────────────────
    for rs in range(T):
        cur_obs = next_obs

        with torch.no_grad():
            fov = torch.stack([get_visible_cells(env.unwrapped) for env in envs.envs]).to(DEVICE)
            mr = gs - 1
            mw = mh.get_memory_window_rollout(global_step=mr)

            action, val, mwr = agent.sample_action(
                obs=cur_obs,
                episode_step=envs_t.clone(),
                fov=fov,
                memory_window=mw,
            )
            ea = action.external_action
            elp = action.external_log_probs

        next_obs, rew, term, trunc, infos = envs.step(ea.cpu().numpy())
        next_obs = torch.as_tensor(next_obs, device=DEVICE)
        reward_t = torch.as_tensor(rew, device=DEVICE)
        done = torch.as_tensor(np.logical_or(term, trunc), dtype=torch.bool, device=DEVICE)

        trajy.store_rollout_step(
            global_step=gs,
            mem_retrieval_step=mr,
            envs_t=envs_t.clone(),
            rollout_step=rs,
            action=ea,
            reward=reward_t,
            done=done,
            value=val,
            logprob=elp,
            obs=cur_obs,
        )

        with torch.no_grad():
            mh.add_memory_write_record(
                global_step=gs,
                write_record=mwr,
                done_envs=done,
                envs_t=envs_t.clone(),
            )

        envs_t += 1
        for eid in range(N):
            if done[eid]:
                envs_t[eid] = 0
        gc += N
        gs += 1

    # ── Bootstrap + GAE ──────────────────────────────────────────────────
    with torch.no_grad():
        bmw = mh.get_memory_window_rollout(global_step=gs - 1)
        bv = agent.bootstrap_value(next_obs, bmw, envs_t)
        trajy.calculate_advantages_and_returns(
            bootstrap_value=bv, gamma=GAMMA, gae_lambda=GAE_LAMBDA
        )

    _all_indices = mh.memory_rollout_buffer.data.memory_indices_next
    _all_masks = mh.memory_rollout_buffer.data.memory_masks_next
    actual_max = int((_all_indices * _all_masks).max().item())

    rng_state = torch.random.get_rng_state()

    envs.close()

    return {
        "trajy": trajy,
        "mh": mh,
        "actual_max": actual_max,
        "ent_coef": ent_coef,
        "rng_state": rng_state,
        "bootstrap_value": bv.clone(),
        "agent": agent,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: run once per session, shared across tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def orig_data():
    """Run original implementation rollout + GAE once."""
    print("\n[fixture] Running original implementation rollout...", flush=True)
    data = run_original_first_iteration()
    print(f"[fixture] Original done. actual_max={data['actual_max']}", flush=True)
    return data


@pytest.fixture(scope="session")
def new_data(orig_data):
    """Run new implementation rollout + GAE once (depends on orig for weights)."""
    print("[fixture] Running new implementation rollout...", flush=True)
    data = run_new_first_iteration(orig_data["init_state"])
    print(f"[fixture] New done. actual_max={data['actual_max']}", flush=True)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: reconstruct minibatch data from both implementations
# ═══════════════════════════════════════════════════════════════════════════════


def get_all_minibatches_original(orig):
    """Reproduce the exact minibatch sequence from the original implementation.

    Uses the captured RNG state to get the same randperm sequence.
    Yields (epoch, mb_idx, dict_of_tensors) for each minibatch.
    """
    torch.random.set_rng_state(orig["rng_state"].clone())

    b_obs = orig["b_obs"]
    b_lp = orig["b_logprobs"]
    b_act = orig["b_actions"]
    b_adv = orig["b_advantages"]
    b_ret = orig["b_returns"]
    b_val = orig["b_values"]
    b_mi = orig["b_memory_index"]
    b_mindices = orig["b_memory_indices"]
    b_mmask = orig["b_memory_mask"]
    stored_mems = orig["stored_memories"]

    results = []
    for epoch in range(UPDATE_EPOCHS):
        b_inds = torch.randperm(BATCH_SIZE)
        mb_idx = 0
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb = b_inds[start:end]

            mb_memories = stored_mems[b_mi[mb]]
            mb_memory_windows = batched_index_select(mb_memories, 1, b_mindices[mb])

            results.append(
                {
                    "epoch": epoch,
                    "mb_idx": mb_idx,
                    "flat_indices": mb.clone(),
                    "obs": b_obs[mb].clone(),
                    "actions": b_act[mb].clone(),
                    "log_probs": b_lp[mb].clone(),
                    "advantages": b_adv[mb].clone(),
                    "returns": b_ret[mb].clone(),
                    "values": b_val[mb].clone(),
                    "memory_windows": mb_memory_windows.clone(),
                    "memory_masks": b_mmask[mb].clone(),
                    "memory_indices": b_mindices[mb].clone(),
                }
            )
            mb_idx += 1

    return results


def get_all_minibatches_new(new_data, rng_state):
    """Reproduce the exact minibatch sequence from the new implementation.

    Uses the same RNG state as the original to get the same randperm.
    """
    torch.random.set_rng_state(rng_state.clone())

    trajy = new_data["trajy"]
    mh = new_data["mh"]
    actual_max = new_data["actual_max"]

    minibatches_grid = torch.cartesian_prod(torch.arange(T), torch.arange(N))

    results = []
    for epoch in range(UPDATE_EPOCHS):
        b_inds = torch.randperm(BATCH_SIZE)
        mb_idx = 0
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_flat = b_inds[start:end]

            mb_steps, mb_envs = minibatches_grid[mb_flat].T
            mb = trajy.get_minibatch(mb_envs=mb_envs, mb_steps=mb_steps)

            mb_mw = mh.get_memory_window_minibatch(
                env_ids=mb_envs, global_steps=mb.mem_retrieval_steps
            )

            _am = actual_max
            results.append(
                {
                    "epoch": epoch,
                    "mb_idx": mb_idx,
                    "flat_indices": mb_flat.clone(),
                    "obs": mb.obs.clone(),
                    "actions": mb.actions.clone(),
                    "log_probs": mb.log_probs.clone(),
                    "advantages": mb.advantages.clone(),
                    "returns": mb.returns.clone(),
                    "values": mb.values.clone(),
                    "memory_windows": mb_mw.pure_memory_frames[:, :_am].clone(),
                    "memory_masks": mb_mw.memory_masks[:, :_am].clone(),
                    "memory_indices": mb_mw.memory_indices_env[:, :_am].clone(),
                }
            )
            mb_idx += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRolloutMatch:
    """Verify that the rollout tensors (pre-flatten) are identical."""

    def test_values_mean_match(self, orig_data, new_data):
        """Mean of value estimates across rollout must match."""
        orig_mean = orig_data["values_2d"].mean().item()
        new_mean = new_data["trajy"].ri.values.mean().item()
        assert abs(orig_mean - new_mean) < 1e-4, (
            f"values mean diff: {abs(orig_mean - new_mean):.8f}"
        )

    def test_advantages_match(self, orig_data, new_data):
        """GAE advantages must match (original is (T,N), new is (N,T))."""
        orig_adv = orig_data["advantages_2d"]  # (T, N)
        new_adv = new_data["trajy"].pri.advantages  # (N, T)
        # Transpose new to (T, N) for comparison
        diff = (orig_adv - new_adv.permute(1, 0)).abs().max().item()
        assert diff < 1e-4, f"advantages max diff: {diff:.8f}"

    def test_returns_match(self, orig_data, new_data):
        """Returns must match."""
        orig_ret = orig_data["returns_2d"]
        new_ret = new_data["trajy"].pri.returns
        diff = (orig_ret - new_ret.permute(1, 0)).abs().max().item()
        assert diff < 1e-4, f"returns max diff: {diff:.8f}"

    def test_bootstrap_value_match(self, orig_data, new_data):
        """Bootstrap values must match."""
        diff = (orig_data["bootstrap_value"] - new_data["bootstrap_value"]).abs().max().item()
        assert diff < 1e-4, f"bootstrap value diff: {diff:.8f}"

    def test_ent_coef_match(self, orig_data, new_data):
        """Entropy coefficient must be identical."""
        assert orig_data["ent_coef"] == new_data["ent_coef"], (
            f"ent_coef: orig={orig_data['ent_coef']} new={new_data['ent_coef']}"
        )

    def test_actual_max_match(self, orig_data, new_data):
        """actual_max_episode_steps must match between implementations.

        The original computes actual_max = max(indices * masks) + 1 (using current-step indices).
        The new computes actual_max = max(indices * masks) (using next-step indices).
        Both should yield the same number.
        """
        assert orig_data["actual_max"] == new_data["actual_max"], (
            f"actual_max: orig={orig_data['actual_max']} new={new_data['actual_max']}"
        )


class TestMinibatchMatch:
    """Verify that every minibatch in the first training iteration
    is identical between original and new implementations."""

    @pytest.fixture(scope="class")
    def all_minibatches(self, orig_data, new_data):
        """Generate all minibatches for both implementations."""
        orig_mbs = get_all_minibatches_original(orig_data)
        new_mbs = get_all_minibatches_new(new_data, orig_data["rng_state"])
        return orig_mbs, new_mbs

    def test_same_number_of_minibatches(self, all_minibatches):
        orig_mbs, new_mbs = all_minibatches
        expected = UPDATE_EPOCHS * NUM_MINIBATCHES
        assert len(orig_mbs) == expected, (
            f"orig has {len(orig_mbs)} minibatches, expected {expected}"
        )
        assert len(new_mbs) == expected, f"new has {len(new_mbs)} minibatches, expected {expected}"

    def test_flat_indices_match(self, all_minibatches):
        """The flat indices (from randperm) must be identical given same RNG."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            assert torch.equal(o["flat_indices"], n["flat_indices"]), (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): flat_indices differ"
            )

    def test_obs_match(self, all_minibatches):
        """Observations must match for every minibatch."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            diff = (o["obs"] - n["obs"]).abs().max().item()
            assert diff < 1e-5, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): obs max diff = {diff:.8f}"
            )

    def test_actions_match(self, all_minibatches):
        """Actions must match."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            assert torch.equal(o["actions"], n["actions"]), (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): actions differ"
            )

    def test_log_probs_match(self, all_minibatches):
        """Log probabilities must match."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            diff = (o["log_probs"] - n["log_probs"]).abs().max().item()
            assert diff < 1e-5, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"log_probs max diff = {diff:.8f}"
            )

    def test_advantages_match(self, all_minibatches):
        """Advantages must match."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            diff = (o["advantages"] - n["advantages"]).abs().max().item()
            assert diff < 1e-5, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"advantages max diff = {diff:.8f}"
            )

    def test_returns_match(self, all_minibatches):
        """Returns must match."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            diff = (o["returns"] - n["returns"]).abs().max().item()
            assert diff < 1e-5, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"returns max diff = {diff:.8f}"
            )

    def test_values_match(self, all_minibatches):
        """Value predictions must match."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            diff = (o["values"] - n["values"]).abs().max().item()
            assert diff < 1e-5, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"values max diff = {diff:.8f}"
            )

    def test_memory_masks_match(self, all_minibatches):
        """Memory masks must match for every minibatch."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            om = o["memory_masks"]
            nm = n["memory_masks"]
            # Shapes must match first
            assert om.shape == nm.shape, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"memory_masks shape: orig={om.shape} new={nm.shape}"
            )
            assert torch.equal(om.bool(), nm.bool()), (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"memory_masks differ. "
                f"orig_true={om.bool().sum().item()} new_true={nm.bool().sum().item()}"
            )

    def test_memory_indices_match(self, all_minibatches):
        """Memory indices must match for every minibatch."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            oi = o["memory_indices"]
            ni = n["memory_indices"]
            assert oi.shape == ni.shape, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"memory_indices shape: orig={oi.shape} new={ni.shape}"
            )
            assert torch.equal(oi, ni), (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): memory_indices differ"
            )

    def test_memory_windows_match(self, all_minibatches):
        """Memory windows (the actual memory content fed to the transformer)
        must match for every minibatch."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            omw = o["memory_windows"]
            nmw = n["memory_windows"]
            assert omw.shape == nmw.shape, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"memory_windows shape: orig={omw.shape} new={nmw.shape}"
            )
            diff = (omw - nmw).abs().max().item()
            assert diff < 1e-5, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"memory_windows max diff = {diff:.8f}"
            )

    def test_memory_windows_unmasked_match(self, all_minibatches):
        """Memory windows must match at all positions where the mask is True
        (active memory slots). This catches issues even if masked positions
        differ due to zeroing conventions."""
        orig_mbs, new_mbs = all_minibatches
        for i, (o, n) in enumerate(zip(orig_mbs, new_mbs)):
            mask = o["memory_masks"].bool()
            if not mask.any():
                continue
            omw = o["memory_windows"][mask]
            nmw = n["memory_windows"][mask]
            diff = (omw - nmw).abs().max().item()
            assert diff < 1e-5, (
                f"Minibatch {i} (epoch={o['epoch']}, mb={o['mb_idx']}): "
                f"memory_windows (unmasked only) max diff = {diff:.8f}"
            )


class TestMinibatchForwardPass:
    """Verify that running the forward pass on each minibatch produces
    identical outputs (logprobs, entropy, values) from both models."""

    @pytest.fixture(scope="class")
    def forward_results(self, orig_data, new_data):
        """Run forward pass for the first minibatch in both implementations."""
        from original_imeplentation_adapted import Agent as OrigAgent
        from original_imeplentation_adapted import Args
        from src_new.model.agent_module import Action

        # Restore original agent
        args = Args()
        args.cuda = False
        args.num_envs = N
        args.num_steps = T
        args.trxl_num_layers = L
        args.trxl_dim = D
        args.trxl_num_heads = H
        args.trxl_memory_length = M
        args.trxl_positional_encoding = "absolute"
        args.reconstruction_coef = 0.0

        orig_agent = OrigAgent(
            args,
            # We need the observation space — reconstruct from shape
            type("FakeSpace", (), {"shape": orig_data["b_obs"].shape[1:]})(),
            (7,),  # action space shape for MiniGrid
            MAX_EP,
        )
        orig_agent.load_state_dict(orig_data["init_state"])
        orig_agent.eval()

        new_agent = new_data["agent"]

        # Get first minibatch from both
        torch.random.set_rng_state(orig_data["rng_state"].clone())
        b_inds = torch.randperm(BATCH_SIZE)
        mb = b_inds[:MINIBATCH_SIZE]

        # Original forward
        b_obs = orig_data["b_obs"]
        b_act = orig_data["b_actions"]
        b_mmask = orig_data["b_memory_mask"]
        b_mindices = orig_data["b_memory_indices"]
        b_mi = orig_data["b_memory_index"]
        stored_mems = orig_data["stored_memories"]

        mb_mems = stored_mems[b_mi[mb]]
        mb_mw = batched_index_select(mb_mems, 1, b_mindices[mb])

        with torch.no_grad():
            _, orig_lp, orig_ent, orig_val, _ = orig_agent.get_action_and_value(
                b_obs[mb], mb_mw, b_mmask[mb], b_mindices[mb], b_act[mb]
            )

        # New forward
        torch.random.set_rng_state(orig_data["rng_state"].clone())
        b_inds_new = torch.randperm(BATCH_SIZE)
        mb_new = b_inds_new[:MINIBATCH_SIZE]

        minibatches_grid = torch.cartesian_prod(torch.arange(T), torch.arange(N))
        mb_steps, mb_envs = minibatches_grid[mb_new].T
        trajy = new_data["trajy"]
        mh = new_data["mh"]
        mb_data = trajy.get_minibatch(mb_envs=mb_envs, mb_steps=mb_steps)
        mb_memory_window = mh.get_memory_window_minibatch(
            env_ids=mb_envs, global_steps=mb_data.mem_retrieval_steps
        )

        _am = new_data["actual_max"]
        act_in = Action(
            external_action=mb_data.actions,
            external_log_probs=torch.empty(0),
            external_entropy=torch.empty(0),
        )
        with torch.no_grad():
            new_result, new_val, _, _ = new_agent.module.get_action_and_value(
                mb_data.obs,
                mb_memory_window.pure_memory_frames[:, :_am],
                mb_memory_window.memory_masks[:, :_am],
                mb_memory_window.memory_indices_env[:, :_am],
                act_in,
            )

        return {
            "orig_lp": orig_lp,
            "orig_ent": orig_ent,
            "orig_val": orig_val,
            "new_lp": new_result.external_log_probs,
            "new_ent": new_result.external_entropy,
            "new_val": new_val,
        }

    def test_forward_logprobs_match(self, forward_results):
        r = forward_results
        diff = (r["orig_lp"] - r["new_lp"]).abs().max().item()
        assert diff < 1e-4, f"Forward pass log_probs diff: {diff:.8f}"

    def test_forward_entropy_match(self, forward_results):
        r = forward_results
        diff = (r["orig_ent"] - r["new_ent"]).abs().max().item()
        assert diff < 1e-4, f"Forward pass entropy diff: {diff:.8f}"

    def test_forward_values_match(self, forward_results):
        r = forward_results
        diff = (r["orig_val"] - r["new_val"]).abs().max().item()
        assert diff < 1e-4, f"Forward pass values diff: {diff:.8f}"
