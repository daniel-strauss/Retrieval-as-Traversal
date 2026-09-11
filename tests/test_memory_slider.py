import pytest
import torch

from src_new.model.retrieval.memory_slider import MemorySlider
from src_new.model.retrieval.temporal_memory import TemporalMemory

N = 4  # num_envs
M = 8  # trxl_memory_length

SHARED_CONFIG = dict(
    num_envs=N,
    num_rollout_steps=128,
    max_episode_steps=50,
    trxl_memory_length=M,
    trxl_num_layers=2,
    trxl_dim=64,
    device=torch.device("cpu"),
)


def _make_slider(**overrides) -> MemorySlider:
    return MemorySlider(**{**SHARED_CONFIG, **overrides})


def _make_temporal(**overrides) -> TemporalMemory:
    return TemporalMemory(**{**SHARED_CONFIG, **overrides})


def _empty_spatial(n: int = N) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(n, 0, dtype=torch.long),
        torch.zeros(n, 0, dtype=torch.bool),
    )


def _step_slider(slider: MemorySlider, t: int, n: int = N, sp_idx=None, sp_mask=None):
    """Helper: advance slider one step with optional spatial, default empty."""
    if sp_idx is None:
        sp_idx, sp_mask = _empty_spatial(n)
    episode_step = torch.full((n,), t, dtype=torch.long)
    return slider.get_memory_indices_and_mask(episode_step, sp_idx, sp_mask)


# ---------------------------------------------------------------------------


def test_equals_temporal_memory_with_empty_spatial_retrieval():
    """Without spatial memories, MemorySlider (with M-1 internal slots) must
    produce indices and masks that match TemporalMemory's first M-1 columns.

    The agent pads slider output to (N, M) by appending the current step
    (masked out) as the M-th column, matching TM exactly.  Here we verify
    the raw slider output against TM[:, :M-1]."""
    slider = _make_slider(trxl_memory_length=M - 1)
    temporal = _make_temporal()  # standard TM with M slots
    sp_idx, sp_mask = _empty_spatial()

    for t in range(1, 21):
        episode_step = torch.full((N,), t, dtype=torch.long)
        s_idx, s_msk = slider.get_memory_indices_and_mask(episode_step, sp_idx, sp_mask)
        t_idx, t_msk = temporal.get_memory_indices_and_mask(episode_step)
        # Compare slider (N, M-1) against TM's first M-1 columns
        t_idx_trimmed = t_idx[:, : M - 1]
        t_msk_trimmed = t_msk[:, : M - 1]

        assert s_idx.shape == (N, M - 1), f"step {t}: indices shape {s_idx.shape}"
        assert s_msk.shape == (N, M - 1), f"step {t}: masks shape {s_msk.shape}"
        assert torch.equal(s_msk.bool(), t_msk_trimmed.bool()), (
            f"step {t}: masks differ\nslider:   {s_msk}\ntemporal: {t_msk_trimmed}"
        )
        # Only compare where mask is valid
        if s_msk.any():
            assert torch.equal(s_idx[s_msk.bool()], t_idx_trimmed[t_msk_trimmed.bool()]), (
                f"step {t}: valid indices differ\nslider: {s_idx}\ntm:     {t_idx_trimmed}"
            )


def test_random_adds_and_retrievals():
    """Spatial indices for an evicted step should re-appear in the returned window."""
    small_m = 4
    slider = _make_slider(trxl_memory_length=small_m)

    # Run steps 1..6 (temporal-only). With M=4, step 0 is evicted after step 5.
    for t in range(1, 7):
        _step_slider(slider, t)

    # At step 7, spatially retrieve step 0 (long-evicted from temporal window)
    episode_step = torch.full((N,), 7, dtype=torch.long)
    spatial_indices = torch.full((N, 1), 0, dtype=torch.long)
    spatial_masks = torch.ones(N, 1, dtype=torch.bool)

    indices, masks = slider.get_memory_indices_and_mask(
        episode_step, spatial_indices, spatial_masks
    )

    assert indices.shape == (N, small_m)
    assert masks.shape == (N, small_m)
    for env in range(N):
        valid = indices[env][masks[env].bool()]
        assert 0 in valid, (
            f"env {env}: spatially retrieved step 0 missing from indices {valid.tolist()}"
        )


def test_environment_resets():
    """After episode_step returns to 1, that env's memory window should reset."""
    slider = _make_slider()

    # Run 6 steps uniformly (steps 1..6)
    for t in range(1, 7):
        _step_slider(slider, t)

    # Reset envs 0 and 2 (back to step 1), keep envs 1 and 3 running at step 7
    episode_step = torch.tensor([1, 7, 1, 7], dtype=torch.long)
    sp_idx, sp_mask = _empty_spatial()
    indices, masks = slider.get_memory_indices_and_mask(episode_step, sp_idx, sp_mask)

    # Reset envs: at step 1 there is exactly 1 past memory (step 0)
    for env in [0, 2]:
        num_valid = masks[env].bool().sum().item()
        assert num_valid == 1, (
            f"env {env} was reset to t=1 but has {num_valid} valid memory slots (expected 1)"
        )

    # Non-reset envs still have accumulated memories
    for env in [1, 3]:
        num_valid = masks[env].bool().sum().item()
        assert num_valid > 0, f"env {env} was NOT reset but has 0 valid memory slots"


def test_out_of_order_retrieval():
    """Skipping a timestep should raise ValueError."""
    slider = _make_slider()

    _step_slider(slider, 1)

    # Skip step 2, jump to step 4
    with pytest.raises(ValueError, match="out of order|sequential|match"):
        _step_slider(slider, 4)


def test_with_no_spatial_mewmory_and_bufer_overwrite_and_many_resets_equals_temporal_memory():
    """Without spatial memories, MemorySlider must match TemporalMemory even when
    the circular buffer wraps many times and envs reset independently at random."""
    torch.manual_seed(123)
    n = 6
    m_len = 4
    max_ep = 30
    cfg = dict(
        num_envs=n,
        num_rollout_steps=128,
        max_episode_steps=max_ep,
        trxl_memory_length=m_len,
        trxl_num_layers=2,
        trxl_dim=64,
        device=torch.device("cpu"),
    )
    slider = MemorySlider(**{**cfg, "trxl_memory_length": m_len - 1})
    temporal = TemporalMemory(**cfg)  # standard TM with m_len slots

    episode_step = torch.ones(n, dtype=torch.long)
    sp_idx = torch.zeros(n, 0, dtype=torch.long)
    sp_mask = torch.zeros(n, 0, dtype=torch.bool)

    # Run 200 calls — way more than M*max_ep to force many buffer overwrites
    for call_idx in range(200):
        # Random resets: ~15% chance per env (not at first step)
        if call_idx > 0:
            reset_mask = (torch.rand(n) < 0.15) & (episode_step > 1)
            episode_step[reset_mask] = 1

        # Force reset envs that would exceed max_episode_steps.
        # Reset at >= max_ep (not >) to avoid TM's clamping bug at the boundary.
        episode_step[episode_step >= max_ep] = 1

        s_idx, s_msk = slider.get_memory_indices_and_mask(episode_step, sp_idx, sp_mask)
        t_idx_full, t_msk_full = temporal.get_memory_indices_and_mask(episode_step)
        t_idx = t_idx_full[:, : m_len - 1]
        t_msk = t_msk_full[:, : m_len - 1]

        assert s_idx.shape == (n, m_len - 1), f"call {call_idx}: shape {s_idx.shape}"
        assert s_msk.shape == (n, m_len - 1), f"call {call_idx}: mask shape {s_msk.shape}"
        assert torch.equal(s_msk.bool(), t_msk.bool()), (
            f"call {call_idx}: masks differ\nslider:   {s_msk}\ntemporal: {t_msk}\nepisode_step: {episode_step}"
        )
        if s_msk.any():
            assert torch.equal(s_idx[s_msk.bool()], t_idx[t_msk.bool()]), (
                f"call {call_idx}: valid indices differ\n"
                f"slider:   {s_idx}\ntemporal: {t_idx}\nepisode_step: {episode_step}"
            )

        episode_step += 1


def test_more_retrievals_than_memory_length():
    """When temporal + spatial indices exceed M, output shape is still (N, M)."""
    small_m = 4
    slider = _make_slider(trxl_memory_length=small_m)

    # Build up more steps than M (steps 1..small_m+3)
    for t in range(1, small_m + 4):
        _step_slider(slider, t)

    # At next step, also spatially request 2 old steps
    episode_step = torch.full((N,), small_m + 4, dtype=torch.long)
    spatial_indices = torch.tensor([[0, 1]] * N, dtype=torch.long)
    spatial_masks = torch.ones(N, 2, dtype=torch.bool)

    indices, masks = slider.get_memory_indices_and_mask(
        episode_step, spatial_indices, spatial_masks
    )

    # Shape must never exceed (N, M)
    assert indices.shape == (N, small_m)
    assert masks.shape == (N, small_m)
    # All M slots should be valid (slider writes only completed steps, no wasted slot)
    assert (masks.bool().sum(dim=1) == small_m).all(), (
        f"Expected {small_m} valid slots per env, got {masks.bool().sum(dim=1).tolist()}"
    )


def test_stress():
    """
    Ultra-hard stress test:
    - 100x more total writes than M across many episodes
    - Near-max (M-1) spatial retrievals per step
    - Random resets at random intervals
    - Per-env independent timelines
    - Validates shape, mask count, no current-step in valid window, spatial entries present,
      all valid indices in [0, max_episode_steps), and no NaN/negative indices.
    """
    torch.manual_seed(42)
    n = 8
    small_m = 6
    max_ep = 200
    cfg = dict(
        num_envs=n,
        num_rollout_steps=128,
        max_episode_steps=max_ep,
        trxl_memory_length=small_m,
        trxl_num_layers=2,
        trxl_dim=64,
        device=torch.device("cpu"),
    )
    slider = MemorySlider(**cfg)

    episode_step = torch.ones(n, dtype=torch.long)  # per-env timestep, +1 convention
    total_calls = small_m * 100  # 600 calls

    # Oracle: per-env ring buffer + write position, mirroring the slider exactly.
    # On reset write_pos=0; the slider writes current_step = retrieval_step - 1.
    oracle_rings: list[list[int]] = [list(range(small_m)) for _ in range(n)]
    oracle_wpos: list[int] = [0] * n

    # Build the same (M+1, M) mask template the slider uses
    mask_template = torch.tril(torch.ones((small_m + 1, small_m)), diagonal=-1).bool()

    for call_idx in range(total_calls):
        # --- Random resets: ~10% chance per env, but only some envs ---
        if call_idx > 0:
            reset_mask = torch.rand(n) < 0.10
            # Don't reset envs already at step 1
            reset_mask = reset_mask & (episode_step > 1)
            episode_step[reset_mask] = 1
            for env in torch.where(reset_mask)[0].tolist():
                oracle_rings[env] = list(range(small_m))
                oracle_wpos[env] = 0

        # --- Random spatial: each env gets 0 to M-1 spatial retrievals ---
        k = torch.randint(0, small_m, (1,)).item()  # same k for all envs this step
        if k > 0:
            # spatial indices are random past steps for each env (or 0 if env just started)
            sp_idx = torch.zeros(n, k, dtype=torch.long)
            sp_mask = torch.zeros(n, k, dtype=torch.bool)
            for env in range(n):
                # actual completed step = episode_step - 1
                t_env = episode_step[env].item() - 1
                # Can only spatially retrieve steps from [0, t_env)
                if t_env > 0:
                    num_spatial = torch.randint(0, min(k, t_env) + 1, (1,)).item()
                    if num_spatial > 0:
                        past_steps = torch.randint(0, t_env, (num_spatial,))
                        sp_idx[env, :num_spatial] = past_steps
                        sp_mask[env, :num_spatial] = True
        else:
            sp_idx = torch.zeros(n, 0, dtype=torch.long)
            sp_mask = torch.zeros(n, 0, dtype=torch.bool)

        # --- Update oracle ring buffer (spatial then temporal, same order as slider) ---
        for env in range(n):
            if k > 0:
                for j in range(k):
                    if sp_mask[env, j]:
                        oracle_rings[env][oracle_wpos[env] % small_m] = sp_idx[env, j].item()
                        oracle_wpos[env] += 1
            # Slider writes the completed step (retrieval_step - 1)
            oracle_rings[env][oracle_wpos[env] % small_m] = episode_step[env].item() - 1
            oracle_wpos[env] += 1

        # --- Call slider ---
        indices, masks = slider.get_memory_indices_and_mask(episode_step, sp_idx, sp_mask)

        # --- Compute expected indices and masks from oracle ---
        for env in range(n):
            wpos = oracle_wpos[env]
            ring = oracle_rings[env]

            # Expected mask: wpos selects template row (all writes are completed steps)
            mask_row = min(max(wpos, 0), small_m)
            expected_mask = mask_template[mask_row]

            assert torch.equal(masks[env], expected_mask), (
                f"call {call_idx}, env {env}: mask mismatch\n"
                f"  got:      {masks[env].tolist()}\n"
                f"  expected: {expected_mask.tolist()}\n"
                f"  wpos={wpos}, episode_step={episode_step[env].item()}"
            )

            # Expected indices: read M slots from logical_start, oldest first
            logical_start = max(wpos - small_m, 0)
            expected_window = [ring[(logical_start + i) % small_m] for i in range(small_m)]
            expected_indices = torch.tensor(expected_window, dtype=torch.long)

            assert torch.equal(indices[env], expected_indices), (
                f"call {call_idx}, env {env}: indices mismatch\n"
                f"  got:      {indices[env].tolist()}\n"
                f"  expected: {expected_indices.tolist()}\n"
                f"  ring={ring}, wpos={wpos}, episode_step={episode_step[env].item()}"
            )

        # Advance all envs for next call
        episode_step += 1
