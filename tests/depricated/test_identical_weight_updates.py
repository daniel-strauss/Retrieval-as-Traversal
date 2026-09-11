print("Starting test_identical_weight_updates.py...", flush=True)

"""
Test that the original implementation (original_implementation.py) and the new
implementation (src_new/) produce **identical** minibatches, losses, logprobs
and weight updates during the first training loop (PPO iteration 1).

The test:
  A) Runs one full rollout in both implementations with the same seed, then
     verifies that within the first training pass every minibatch contains
     the same (obs, actions, returns, advantages, memory windows, masks).
  B) After each minibatch gradient step, checks that logprobs, entropies,
     losses and values match.
  C) After the full first training loop (all epochs x minibatches), confirms
     that the network weights are identical.

Run with:  python -m src_new.tests.test_identical_weight_updates
      or:  python src_new/tests/test_identical_weight_updates.py
"""

import warnings

warnings.filterwarnings("ignore")
import os
import random
import sys

# Ensure project root is on the path so we can import original_implementation & src_new
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — must match the YAML / original_implementation.py defaults
# ═══════════════════════════════════════════════════════════════════════════════

N = 16  # num_envs
T = 256  # num_rollout_steps
M = 64  # trxl_memory_length
L = 2  # trxl_num_layers
D = 256  # trxl_dim
H = 4  # trxl_num_heads
SEED = 1
MAX_EP = 96
DEVICE = torch.device("cpu")
GAMMA = 0.995
LAM = 0.95
CLIP = 0.2
VF_COEF = 0.5
ENT0 = 0.0001
ENT1 = 0.000001
LR0 = 2.75e-4
LR1 = 1e-5
ANNEAL = 4096000
EPOCHS = 3
NUM_MB = 8
BS = N * T
MBS = BS // NUM_MB
MAX_GRAD = 0.25

# Tolerances
ATOL_TIGHT = 1e-5  # for single forward-pass quantities
ATOL_LOSS = 1e-3  # accumulated gradient steps may drift a bit (float32)
ATOL_WEIGHT = 1e-3  # weights after full training loop


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
    mask = torch.tril(torch.ones((mem_len, mem_len)), diagonal=-1)
    reps = torch.repeat_interleave(torch.arange(mem_len).unsqueeze(0), mem_len - 1, dim=0).long()
    idx = torch.stack([torch.arange(i, i + mem_len) for i in range(max_ep - mem_len + 1)]).long()
    idx = torch.cat((reps, idx))
    return mask, idx


def close(a, b, atol):
    """Check if two tensors are close."""
    if a.shape != b.shape:
        return False
    return (a.float() - b.float()).abs().max().item() <= atol


def report_diff(name, a, b, atol=ATOL_TIGHT):
    d = (a.float() - b.float()).abs().max().item()
    ok = d <= atol
    status = "OK" if ok else "MISMATCH"
    if not ok:
        print(f"    {name}: maxdiff={d:.10f}  [{status}]")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Run ORIGINAL implementation – one full iteration
# Returns all per-minibatch intermediate data
# ═══════════════════════════════════════════════════════════════════════════════


def run_original_iteration():
    """
    Run one PPO iteration of the original flat implementation.
    Returns:
        init_state:         initial weight dict
        rng_before_train:   torch RNG state right before the training loop
        mb_data:            list of dicts, one per minibatch step
        weights_after:      state_dict after training
        actual_max:         the actual_max_episode_steps value
        ent_coef:           computed entropy coefficient
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
    args.batch_size = BS
    args.minibatch_size = MBS
    args.num_minibatches = NUM_MB
    args.update_epochs = EPOCHS
    args.clip_coef = CLIP
    args.vf_coef = VF_COEF
    args.clip_vloss = True
    args.norm_adv = False
    args.max_grad_norm = MAX_GRAD
    args.init_ent_coef = ENT0
    args.final_ent_coef = ENT1
    args.init_lr = LR0
    args.final_lr = LR1
    args.gamma = GAMMA
    args.gae_lambda = LAM
    args.anneal_steps = ANNEAL

    agent = Agent(args, envs.single_observation_space, asp, MAX_EP)
    init_state = {k: v.clone() for k, v in agent.state_dict().items()}
    optimizer = torch.optim.AdamW(agent.parameters(), lr=LR0)

    mask_tpl, idx_tpl = build_templates(M, MAX_EP)

    # ── Annealing (iteration 1, global_step starts at 0) ────────────
    global_step = 0
    frac = 1 - global_step / ANNEAL
    lr = (LR0 - LR1) * frac + LR1
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    ent_coef = (ENT0 - ENT1) * frac + ENT1

    # ── Storage ──────────────────────────────────────────────────────
    rewards = torch.zeros(T, N)
    actions = torch.zeros(T, N, len(asp), dtype=torch.long)
    dones = torch.zeros(T, N)
    obs = torch.zeros((T, N) + envs.single_observation_space.shape)
    log_probs = torch.zeros(T, N, len(asp))
    values = torch.zeros(T, N)
    sm_masks = torch.zeros(T, N, M, dtype=torch.bool)
    sm_index = torch.zeros(T, N, dtype=torch.long)
    sm_indices = torch.zeros(T, N, M, dtype=torch.long)

    env_step = torch.zeros(N, dtype=torch.long)
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(N)
    next_memory = torch.zeros(N, MAX_EP, L, D)

    stored_memories = [next_memory[e] for e in range(N)]
    for e in range(N):
        sm_index[:, e] = e

    # ── Rollout ──────────────────────────────────────────────────────
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

    # ── Bootstrap + GAE ──────────────────────────────────────────────
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
            adv[t] = lg = d + GAMMA * LAM * nnt * lg
        rets = adv + values

    # ── Flatten ──────────────────────────────────────────────────────
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

    # ── Training loop — capture every minibatch ──────────────────────
    rng_before_train = torch.random.get_rng_state()
    mb_data = []

    for epoch in range(EPOCHS):
        b_inds = torch.randperm(BS)
        for s in range(0, BS, MBS):
            mb = b_inds[s : s + MBS]

            mb_m = stored_memories_stacked[b_mi[mb]]
            mb_mw = batched_index_select(mb_m, 1, b_mindices[mb])

            # Record minibatch inputs
            mb_rec = {
                "mb_inds": mb.clone(),
                "obs": b_obs[mb].clone(),
                "actions": b_act[mb].clone(),
                "advantages": b_adv[mb].clone(),
                "returns": b_ret[mb].clone(),
                "old_logprobs": b_lp[mb].clone(),
                "old_values": b_val[mb].clone(),
                "memory_window": mb_mw.clone(),
                "memory_mask": b_mmask[mb].clone(),
                "memory_indices": b_mindices[mb].clone(),
            }

            _, nlp, ent, nval, _ = agent.get_action_and_value(
                b_obs[mb], mb_mw, b_mmask[mb], b_mindices[mb], b_act[mb]
            )

            ma = b_adv[mb].unsqueeze(1).repeat(1, len(asp))
            lr_ = nlp - b_lp[mb]
            ratio = torch.exp(lr_)
            pg1 = -ma * ratio
            pg2 = -ma * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
            pg = torch.max(pg1, pg2).mean()
            vu = (nval - b_ret[mb]) ** 2
            vc = b_val[mb] + (nval - b_val[mb]).clamp(-CLIP, CLIP)
            vl = torch.max(vu, (vc - b_ret[mb]) ** 2).mean()
            el = ent.mean()
            loss = pg - ent_coef * el + vl * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD)
            optimizer.step()

            # Record outputs
            mb_rec["newlogprob"] = nlp.detach().clone()
            mb_rec["entropy"] = ent.detach().clone()
            mb_rec["newvalue"] = nval.detach().clone()
            mb_rec["pg_loss"] = pg.detach().clone()
            mb_rec["v_loss"] = vl.detach().clone()
            mb_rec["loss"] = loss.detach().clone()

            # Snapshot weights after this step
            mb_rec["weights_hash"] = sum(v.sum().item() for v in agent.state_dict().values())

            mb_data.append(mb_rec)

    weights_after = {k: v.clone() for k, v in agent.state_dict().items()}
    envs.close()

    return {
        "init_state": init_state,
        "rng_before_train": rng_before_train,
        "mb_data": mb_data,
        "weights_after": weights_after,
        "actual_max": actual_max,
        "ent_coef": ent_coef,
        "b_adv": b_adv.clone(),
        "b_ret": b_ret.clone(),
        "b_val": b_val.clone(),
        "b_lp": b_lp.clone(),
        "b_obs": b_obs.clone(),
        "b_act": b_act.clone(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Run NEW implementation – one full iteration,
# reusing the same initial weights, seed, and RNG state
# ═══════════════════════════════════════════════════════════════════════════════


def run_new_iteration(orig):
    """
    Run one PPO iteration of the new implementation.
    Uses the same initial weights and syncs the RNG before the training loop.
    """
    from src_new.memory.handler import MemoryHandler
    from src_new.model.agent import Agent
    from src_new.model.agent_module import Action
    from src_new.trainer.trainer import make_env
    from src_new.trainer.trajectory import Trajectory
    from src_new.utils import get_visible_cells

    init_state = orig["init_state"]

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

    # Load original weights into the new agent
    sd = agent.module.state_dict()
    for k, v in init_state.items():
        nk = (
            "transformer.pos_embedding_a.inv_freqs"
            if k == "transformer.pos_embedding.inv_freqs"
            else k
        )
        if nk in sd:
            sd[nk] = v.clone()
        else:
            print(f"  WARNING: key {k} -> {nk} not found in new state_dict")
    agent.module.load_state_dict(sd)

    optimizer = torch.optim.AdamW(agent.module.parameters(), lr=LR0)

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

    # ── Annealing ────────────────────────────────────────────────────
    gc = 0
    frac = 1 - gc / ANNEAL
    lr = (LR0 - LR1) * frac + LR1
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    ent_coef = (ENT0 - ENT1) * frac + ENT1

    # ── Rollout ──────────────────────────────────────────────────────
    envs_t = torch.zeros(N, dtype=torch.long, device=DEVICE)
    gs = 0
    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.as_tensor(next_obs, device=DEVICE)
    done = torch.zeros(N, dtype=torch.bool, device=DEVICE)

    trajy.new_rollout()

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

    # ── Bootstrap + GAE ──────────────────────────────────────────────
    with torch.no_grad():
        bmw = mh.get_memory_window_rollout(global_step=gs - 1)
        bv = agent.bootstrap_value(next_obs, bmw, envs_t)
        trajy.calculate_advantages_and_returns(bootstrap_value=bv, gamma=GAMMA, gae_lambda=LAM)

    # ── actual_max (must match original) ─────────────────────────────
    _all_indices = mh.memory_rollout_buffer.data.memory_indices_next
    _all_masks = mh.memory_rollout_buffer.data.memory_masks_next
    actual_max = int((_all_indices * _all_masks).max().item())

    # ── Sync RNG so minibatch permutations are identical ─────────────
    torch.random.set_rng_state(orig["rng_before_train"])

    # ── Training loop ────────────────────────────────────────────────
    minibatches = torch.cartesian_prod(torch.arange(T), torch.arange(N))
    mb_data = []

    for epoch in range(EPOCHS):
        b_inds = torch.randperm(BS)
        for s in range(0, BS, MBS):
            end = s + MBS
            mb_inds = b_inds[s:end]
            mb_steps, mb_envs = minibatches[mb_inds].T
            mb = trajy.get_minibatch(mb_envs=mb_envs, mb_steps=mb_steps)
            mb_mw = mh.get_memory_window_minibatch(
                env_ids=mb_envs, global_steps=mb.mem_retrieval_steps
            )

            _am = actual_max

            mb_rec = {
                "mb_inds": mb_inds.clone(),
                "obs": mb.obs.clone(),
                "actions": mb.actions.clone(),
                "advantages": mb.advantages.clone(),
                "returns": mb.returns.clone(),
                "old_logprobs": mb.log_probs.clone(),
                "old_values": mb.values.clone(),
                "memory_window": mb_mw.frames[:, :_am].clone(),
                "memory_mask": mb_mw.masks[:, :_am].clone(),
                "memory_indices": mb_mw.indices_env[:, :_am].clone(),
            }

            act_in = Action(
                external_action=mb.actions,
                external_log_probs=torch.empty(0, device=DEVICE),
                external_entropy=torch.empty(0, device=DEVICE),
            )
            result, nval, _, _ = agent.module.get_action_and_value(
                mb.obs,
                mb_mw.frames[:, :_am],
                mb_mw.masks[:, :_am],
                mb_mw.indices_env[:, :_am],
                act_in,
            )
            nlp = result.external_log_probs
            ent = result.external_entropy

            ma = mb.advantages.clone().unsqueeze(1).repeat(1, len(asp))
            lr_ = nlp - mb.log_probs
            ratio = torch.exp(lr_)
            pg1 = -ma * ratio
            pg2 = -ma * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
            pg = torch.max(pg1, pg2).mean()
            vu = (nval - mb.returns) ** 2
            vc = mb.values + (nval - mb.values).clamp(-CLIP, CLIP)
            vl = torch.max(vu, (vc - mb.returns) ** 2).mean()
            el = ent.mean()
            loss = pg - ent_coef * el + vl * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.module.parameters(), MAX_GRAD)
            optimizer.step()

            mb_rec["newlogprob"] = nlp.detach().clone()
            mb_rec["entropy"] = ent.detach().clone()
            mb_rec["newvalue"] = nval.detach().clone()
            mb_rec["pg_loss"] = pg.detach().clone()
            mb_rec["v_loss"] = vl.detach().clone()
            mb_rec["loss"] = loss.detach().clone()
            mb_rec["weights_hash"] = sum(v.sum().item() for v in agent.module.state_dict().values())

            mb_data.append(mb_rec)

    weights_after = {k: v.clone() for k, v in agent.module.state_dict().items()}
    envs.close()

    return {
        "mb_data": mb_data,
        "weights_after": weights_after,
        "actual_max": actual_max,
        "ent_coef": ent_coef,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════════════


def compare(orig, new):
    total_mb = len(orig["mb_data"])
    assert total_mb == len(new["mb_data"]), (
        f"Number of minibatch steps differs: orig={total_mb}, new={len(new['mb_data'])}"
    )

    print(f"\n  Total minibatch steps: {total_mb}")
    print(f"  actual_max: orig={orig['actual_max']}  new={new['actual_max']}")

    # ── (A) Minibatch input identity ─────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE A: Minibatch inputs identical?")
    print("=" * 80)

    a_ok = True
    for i, (om, nm) in enumerate(zip(orig["mb_data"], new["mb_data"])):
        step_ok = True
        step_ok &= report_diff(f"mb[{i}] obs", om["obs"], nm["obs"])
        step_ok &= report_diff(f"mb[{i}] actions", om["actions"].float(), nm["actions"].float())
        step_ok &= report_diff(f"mb[{i}] advantages", om["advantages"], nm["advantages"])
        step_ok &= report_diff(f"mb[{i}] returns", om["returns"], nm["returns"])
        step_ok &= report_diff(f"mb[{i}] old_logprobs", om["old_logprobs"], nm["old_logprobs"])
        step_ok &= report_diff(f"mb[{i}] old_values", om["old_values"], nm["old_values"])

        # Compare memory window shape first (actual_max may differ)
        om_mw = om["memory_window"]
        nm_mw = nm["memory_window"]
        if om_mw.shape != nm_mw.shape:
            print(f"    mb[{i}] memory_window SHAPE MISMATCH: orig={om_mw.shape} new={nm_mw.shape}")
            # Compare the overlapping part
            common = min(om_mw.shape[1], nm_mw.shape[1])
            step_ok &= report_diff(
                f"mb[{i}] memory_window[:, :{common}]",
                om_mw[:, :common],
                nm_mw[:, :common],
            )
            step_ok = False
        else:
            step_ok &= report_diff(f"mb[{i}] memory_window", om_mw, nm_mw)

        om_mm = om["memory_mask"]
        nm_mm = nm["memory_mask"]
        if om_mm.shape != nm_mm.shape:
            print(f"    mb[{i}] memory_mask SHAPE MISMATCH: orig={om_mm.shape} new={nm_mm.shape}")
            common = min(om_mm.shape[1], nm_mm.shape[1])
            step_ok &= report_diff(
                f"mb[{i}] memory_mask[:, :{common}]",
                om_mm[:, :common].float(),
                nm_mm[:, :common].float(),
            )
            step_ok = False
        else:
            step_ok &= report_diff(f"mb[{i}] memory_mask", om_mm.float(), nm_mm.float())

        om_mi = om["memory_indices"]
        nm_mi = nm["memory_indices"]
        if om_mi.shape != nm_mi.shape:
            print(
                f"    mb[{i}] memory_indices SHAPE MISMATCH: orig={om_mi.shape} new={nm_mi.shape}"
            )
            step_ok = False
        else:
            step_ok &= report_diff(f"mb[{i}] memory_indices", om_mi.float(), nm_mi.float())

        if not step_ok:
            a_ok = False

    if a_ok:
        print("  >> ALL minibatch inputs MATCH.")
    else:
        print("  >> SOME minibatch inputs DIFFER — see details above.")

    # ── (B) Per-minibatch outputs ────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE B: Per-minibatch logprobs, losses, values identical?")
    print("=" * 80)

    b_ok = True
    for i, (om, nm) in enumerate(zip(orig["mb_data"], new["mb_data"])):
        step_ok = True
        step_ok &= report_diff(
            f"mb[{i}] newlogprob", om["newlogprob"], nm["newlogprob"], atol=ATOL_LOSS
        )
        step_ok &= report_diff(f"mb[{i}] entropy", om["entropy"], nm["entropy"], atol=ATOL_LOSS)
        step_ok &= report_diff(f"mb[{i}] newvalue", om["newvalue"], nm["newvalue"], atol=ATOL_LOSS)
        step_ok &= report_diff(f"mb[{i}] pg_loss", om["pg_loss"], nm["pg_loss"], atol=ATOL_LOSS)
        step_ok &= report_diff(f"mb[{i}] v_loss", om["v_loss"], nm["v_loss"], atol=ATOL_LOSS)
        step_ok &= report_diff(f"mb[{i}] loss", om["loss"], nm["loss"], atol=ATOL_LOSS)

        wh_diff = abs(om["weights_hash"] - nm["weights_hash"])
        if wh_diff > ATOL_WEIGHT:
            print(f"    mb[{i}] weights_hash diff={wh_diff:.8f}  [MISMATCH]")
            step_ok = False

        if not step_ok:
            b_ok = False

    if b_ok:
        print("  >> ALL per-minibatch outputs MATCH.")
    else:
        print("  >> SOME per-minibatch outputs DIFFER — see details above.")

    # ── (C) Final weights ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE C: Final network weights identical?")
    print("=" * 80)

    orig_w = orig["weights_after"]
    new_w = new["weights_after"]

    c_ok = True
    for ok_key, ov in orig_w.items():
        nk = (
            "transformer.pos_embedding_a.inv_freqs"
            if ok_key == "transformer.pos_embedding.inv_freqs"
            else ok_key
        )
        if nk not in new_w:
            print(f"  Key {ok_key} -> {nk} missing in new weights!")
            c_ok = False
            continue
        d = (ov.float() - new_w[nk].float()).abs().max().item()
        if d > ATOL_WEIGHT:
            print(f"  {ok_key}: maxdiff={d:.10f}  [MISMATCH]")
            c_ok = False

    # Check for keys in new but not in original
    mapped_orig_keys = set()
    for k in orig_w:
        nk = (
            "transformer.pos_embedding_a.inv_freqs"
            if k == "transformer.pos_embedding.inv_freqs"
            else k
        )
        mapped_orig_keys.add(nk)
    for nk in new_w:
        if nk not in mapped_orig_keys:
            print(f"  Extra key in new weights: {nk}")

    if c_ok:
        print("  >> ALL final weights MATCH.")
    else:
        print("  >> SOME final weights DIFFER — see details above.")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  A (minibatch inputs):   {'PASS' if a_ok else 'FAIL'}")
    print(f"  B (losses & logprobs):  {'PASS' if b_ok else 'FAIL'}")
    print(f"  C (final weights):      {'PASS' if c_ok else 'FAIL'}")
    all_pass = a_ok and b_ok and c_ok
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAIL'}")
    print("=" * 80)

    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Running original implementation (1 PPO iteration) ...")
    print("=" * 80, flush=True)
    orig = run_original_iteration()
    print(
        f"  Done. actual_max={orig['actual_max']}, "
        f"num_mb_steps={len(orig['mb_data'])}, "
        f"ent_coef={orig['ent_coef']:.8f}",
        flush=True,
    )

    print("\n" + "=" * 80)
    print("Running new implementation (1 PPO iteration) ...")
    print("=" * 80, flush=True)
    new = run_new_iteration(orig)
    print(
        f"  Done. actual_max={new['actual_max']}, "
        f"num_mb_steps={len(new['mb_data'])}, "
        f"ent_coef={new['ent_coef']:.8f}",
        flush=True,
    )

    ok = compare(orig, new)
    sys.exit(0 if ok else 1)
