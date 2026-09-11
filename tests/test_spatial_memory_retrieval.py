"""End-to-end integration tests for the spatial memory retrieval pipeline.

Tests the contract between:
  - Agent's perceived_fov construction  (write side)
  - SpatialIndex.add_fov / get_memory_indices_and_mask
  - InternalActionHead.action_to_position  (read side)

No mocks — uses real SpatialIndex + real head.action_to_position so that
coordinate offset bugs are caught rather than hidden.
"""

import pytest
import torch

from src_new.model.internal_action_head import (
    make_internal_head,
)
from src_new.model.retrieval.spatial_index import SpatialIndex

# ── Helpers ──────────────────────────────────────────────────────────────

GRID_W, GRID_H = 5, 5
DEVICE = torch.device("cpu")


def build_perceived_fov(
    num_envs: int,
    grid_w: int,
    grid_h: int,
    perceived_pos: torch.Tensor,
) -> torch.Tensor:
    """Reproduce Agent.sample_action's 3×3 perceived_fov construction."""
    W = grid_w * 2 - 1
    H = grid_h * 2 - 1
    offset = torch.tensor([grid_w - 1, grid_h - 1])
    perceived_fov = torch.zeros(num_envs, W, H, dtype=torch.bool)
    env_idx = torch.arange(num_envs)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            idx = perceived_pos + offset + torch.tensor([di, dj])
            valid = (idx[:, 0] >= 0) & (idx[:, 0] < W) & (idx[:, 1] >= 0) & (idx[:, 1] < H)
            perceived_fov[env_idx[valid], idx[valid, 0], idx[valid, 1]] = True
    return perceived_fov


def spawn_rel_to_categorical_action(pos: torch.Tensor, grid_w: int, grid_h: int) -> torch.Tensor:
    """Inverse of CategoricalXYHead.action_to_position: spawn-rel → raw action."""
    center = torch.tensor([grid_w - 1, grid_h - 1])
    return (pos + center).long()


def spawn_rel_to_gaussian_action(pos: torch.Tensor) -> torch.Tensor:
    """Inverse of GaussianSpatialHead.action_to_position: spawn-rel → raw float action.
    round(action).long() == pos, so just use float(pos)."""
    return pos.float()


# ── Test 1: Direct write → head.action_to_position → retrieve ────────


@pytest.mark.parametrize("head_type", ["gaussian", "categorical_xy"])
def test_direct_write_then_retrieve_via_head(head_type: str):
    """
    Write a single spawn-relative cell into SpatialIndex at step 0.
    Construct a raw action that action_to_position maps to that cell.
    Assert retrieval returns step 0.
    Also assert that a different position returns empty.
    """
    num_envs = 1
    si = SpatialIndex(num_envs=num_envs, grid_w=GRID_W, grid_h=GRID_H, device=DEVICE)
    head = make_internal_head(head_type, trxl_dim=16, grid_w=GRID_W, grid_h=GRID_H)

    # ── Step 0: write spawn-rel (1, -2) ──
    target_pos = torch.tensor([[1, -2]])  # spawn-relative
    perceived_fov = build_perceived_fov(num_envs, GRID_W, GRID_H, target_pos)
    si.add_fov(perceived_fov, env_steps=torch.tensor([0]))

    # ── Build raw action that maps back to (1, -2) ──
    if head_type == "categorical_xy":
        raw_action = spawn_rel_to_categorical_action(target_pos, GRID_W, GRID_H)
    else:
        raw_action = spawn_rel_to_gaussian_action(target_pos)

    # ── Retrieve via head ──
    position = head.action_to_position(raw_action)
    assert position.dtype == torch.long
    assert torch.equal(position, target_pos.long()), (
        f"action_to_position should return {target_pos}, got {position}"
    )

    indices, mask = si.get_memory_indices_and_mask(position)
    valid_steps = indices[0][mask[0]].tolist()
    assert 0 in valid_steps, f"Step 0 should be retrievable at (1,-2), got {valid_steps}"

    # ── Miss: position (3, 3) was never written ──
    miss_pos = torch.tensor([[3, 3]])
    mi, mm = si.get_memory_indices_and_mask(miss_pos)
    assert mm.sum() == 0, f"Position (3,3) should be empty, got indices={mi}"


# ── Test 2: Full round-trip with 3×3 neighborhood fov ────────────────


@pytest.mark.parametrize("head_type", ["gaussian", "categorical_xy"])
def test_neighborhood_fov_roundtrip(head_type: str):
    """
    Agent at spawn-rel (2, 3) writes a 3×3 neighbourhood.
    Every cell in the neighbourhood should be retrievable.
    A cell outside (2+2, 3) should not.
    """
    num_envs = 1
    si = SpatialIndex(num_envs=num_envs, grid_w=GRID_W, grid_h=GRID_H, device=DEVICE)
    head = make_internal_head(head_type, trxl_dim=16, grid_w=GRID_W, grid_h=GRID_H)

    perceived_pos = torch.tensor([[2, 3]])
    perceived_fov = build_perceived_fov(num_envs, GRID_W, GRID_H, perceived_pos)
    si.add_fov(perceived_fov, env_steps=torch.tensor([0]))

    # ── Every 3×3 neighbour should be a hit ──
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            query = torch.tensor([[2 + di, 3 + dj]])

            if head_type == "categorical_xy":
                raw_action = spawn_rel_to_categorical_action(query, GRID_W, GRID_H)
            else:
                raw_action = spawn_rel_to_gaussian_action(query)

            position = head.action_to_position(raw_action)
            assert torch.equal(position, query.long()), (
                f"head round-trip failed for ({2 + di},{3 + dj}): got {position}"
            )

            indices, mask = si.get_memory_indices_and_mask(position)
            assert mask[0].any(), f"Neighbour ({2 + di},{3 + dj}) should be retrievable, got empty"
            assert indices[0, 0].item() == 0

    # ── Outside the 3×3 → miss ──
    far_query = torch.tensor([[4, 3]])
    _, far_mask = si.get_memory_indices_and_mask(far_query)
    assert far_mask.sum() == 0, "Position (4,3) is outside 3×3 neighbourhood and should miss"


# ── Test 3: Multi-step accumulation + assertions after each step ──────


@pytest.mark.parametrize("head_type", ["gaussian", "categorical_xy"])
def test_multi_step_accumulation(head_type: str):
    """
    Simulate 4 steps where the agent moves to different positions.
    After each step, assert that all positions seen SO FAR are retrievable,
    and verify the exact set of step-indices returned.
    """
    num_envs = 1
    si = SpatialIndex(num_envs=num_envs, grid_w=GRID_W, grid_h=GRID_H, device=DEVICE)
    head = make_internal_head(head_type, trxl_dim=16, grid_w=GRID_W, grid_h=GRID_H)

    # Agent visits these spawn-relative positions at steps 0..3.
    # Step 0 and 2 share position (0,0), so retrieving (0,0) after step 2
    # should return {0, 2}.
    positions_per_step = [
        torch.tensor([[0, 0]]),  # step 0
        torch.tensor([[1, -1]]),  # step 1
        torch.tensor([[0, 0]]),  # step 2
        torch.tensor([[-3, 2]]),  # step 3
    ]

    # Expected cumulative hits at each position after each step
    expected_at: dict[tuple[int, int], list[int]] = {}

    def _to_raw(pos_t: torch.Tensor) -> torch.Tensor:
        if head_type == "categorical_xy":
            return spawn_rel_to_categorical_action(pos_t, GRID_W, GRID_H)
        return spawn_rel_to_gaussian_action(pos_t)

    for step, pos in enumerate(positions_per_step):
        # Write
        fov = build_perceived_fov(num_envs, GRID_W, GRID_H, pos)
        si.add_fov(fov, env_steps=torch.tensor([step]))

        # Update expected hits (the 3×3 neighbourhood around pos)
        cx, cy = int(pos[0, 0]), int(pos[0, 1])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                key = (cx + di, cy + dj)
                expected_at.setdefault(key, []).append(step)

        # ── Assert every known position matches expected steps ──
        for (ex, ey), exp_steps in expected_at.items():
            query = torch.tensor([[ex, ey]])
            position = head.action_to_position(_to_raw(query))
            indices, mask = si.get_memory_indices_and_mask(position)
            valid = sorted(indices[0][mask[0]].tolist())
            assert valid == sorted(exp_steps), (
                f"After step {step}: at ({ex},{ey}) expected steps {sorted(exp_steps)}, got {valid}"
            )


# ── Test 4: Environment reset clears spatial index ────────────────────


@pytest.mark.parametrize("head_type", ["gaussian", "categorical_xy"])
def test_reset_clears_and_restarts(head_type: str):
    """
    2 envs. Both write at step 0. Reset env 0 only.
    - Env 0: old data gone, can write at step 0 again.
    - Env 1: data intact.
    Then both write one more step and verify.
    """
    num_envs = 2
    si = SpatialIndex(num_envs=num_envs, grid_w=GRID_W, grid_h=GRID_H, device=DEVICE)
    head = make_internal_head(head_type, trxl_dim=16, grid_w=GRID_W, grid_h=GRID_H)

    def _to_raw(pos_t: torch.Tensor) -> torch.Tensor:
        if head_type == "categorical_xy":
            return spawn_rel_to_categorical_action(pos_t, GRID_W, GRID_H)
        return spawn_rel_to_gaussian_action(pos_t)

    # ── Step 0: env0 at (1,1), env1 at (-1,-1) ──
    pos_step0 = torch.tensor([[1, 1], [-1, -1]])
    fov = build_perceived_fov(num_envs, GRID_W, GRID_H, pos_step0)
    si.add_fov(fov, env_steps=torch.tensor([0, 0]))

    # Verify both written
    q0 = head.action_to_position(_to_raw(torch.tensor([[1, 1], [-1, -1]])))
    _, m0 = si.get_memory_indices_and_mask(q0)
    assert m0[0].any(), "Env 0 should have data at (1,1)"
    assert m0[1].any(), "Env 1 should have data at (-1,-1)"

    # ── Reset env 0 only ──
    si.reset_envs(dones=torch.tensor([True, False]))

    # Env 0: old data gone
    q_post = head.action_to_position(_to_raw(torch.tensor([[1, 1], [-1, -1]])))
    idx_post, m_post = si.get_memory_indices_and_mask(q_post)
    assert not m_post[0].any(), "Env 0 data should be cleared after reset"
    assert m_post[1].any(), "Env 1 data should survive the reset"
    assert idx_post[1][m_post[1]].tolist() == [0], "Env 1 should still have step 0"

    # ── Env 0 writes at step 0 again (new episode), env 1 at step 1 ──
    pos_after_reset = torch.tensor([[2, 2], [0, 0]])
    fov2 = build_perceived_fov(num_envs, GRID_W, GRID_H, pos_after_reset)
    si.add_fov(fov2, env_steps=torch.tensor([0, 1]))

    # Env 0: only new data at (2,2)
    q_env0 = head.action_to_position(_to_raw(torch.tensor([[2, 2], [0, 0]])))
    idx2, m2 = si.get_memory_indices_and_mask(q_env0)
    assert idx2[0][m2[0]].tolist() == [0], (
        f"Env 0 after reset+write should have step 0 at (2,2), got {idx2[0][m2[0]].tolist()}"
    )

    # Env 0: old position (0,0) is outside the 3×3 of (2,2) → still empty
    q_old = head.action_to_position(_to_raw(torch.tensor([[0, 0], [0, 0]])))
    _, m_old = si.get_memory_indices_and_mask(q_old)
    assert not m_old[0].any(), "Env 0 position (0,0) should be empty (outside 3×3 of (2,2))"

    # Env 1: (0,0) was written at step 1 via the 3×3 of step 1's perceived_pos=(0,0)
    # and also step 0's perceived_pos=(-1,-1) had (0,0) in its 3×3 neighbourhood
    env1_steps = sorted(idx2[1][m2[1]].tolist())
    assert env1_steps == [0, 1], f"Env 1 at (0,0) should have steps [0, 1], got {env1_steps}"


# ── Test 5: Multi-env multi-step stress test with resets ──────────────


def test_multi_env_multi_step_with_resets():
    """
    3 envs, 10 steps, deterministic resets at known steps.
    Maintain a reference dict and compare after every step.
    """
    num_envs = 3
    grid_w, grid_h = 4, 4
    si = SpatialIndex(num_envs=num_envs, grid_w=grid_w, grid_h=grid_h, device=DEVICE)

    torch.manual_seed(99)

    # Reference: env -> (x,y) -> set of steps
    ref: list[dict[tuple[int, int], set[int]]] = [dict() for _ in range(num_envs)]
    next_step = [0] * num_envs

    # Pre-generate positions and reset schedule
    all_positions = torch.randint(-(grid_w - 1), grid_w, (10, num_envs, 2))
    reset_at = {3: [0], 5: [1, 2], 8: [0]}  # step -> list of envs to reset

    for step in range(10):
        # Build per-env perceived_pos
        pos = all_positions[step]  # (num_envs, 2)
        fov = build_perceived_fov(num_envs, grid_w, grid_h, pos)
        env_steps = torch.tensor([next_step[e] for e in range(num_envs)])
        si.add_fov(fov, env_steps)

        # Update reference (only cells within the valid perceived_fov bounds)
        W = grid_w * 2 - 1
        H = grid_h * 2 - 1
        for env in range(num_envs):
            cx, cy = int(pos[env, 0]), int(pos[env, 1])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nx, ny = cx + di, cy + dj
                    # Same bounds check as build_perceived_fov
                    ti, tj = nx + (grid_w - 1), ny + (grid_h - 1)
                    if 0 <= ti < W and 0 <= tj < H:
                        ref[env].setdefault((nx, ny), set()).add(next_step[env])
            next_step[env] += 1

        # Handle resets
        if step in reset_at:
            dones = torch.zeros(num_envs, dtype=torch.bool)
            for env in reset_at[step]:
                dones[env] = True
                ref[env] = dict()
                next_step[env] = 0
            si.reset_envs(dones)

        # ── Verify every position in reference ──
        for env in range(num_envs):
            for (rx, ry), expected_steps in ref[env].items():
                query = torch.zeros(num_envs, 2, dtype=torch.long)
                query[env] = torch.tensor([rx, ry])
                indices, mask = si.get_memory_indices_and_mask(query)
                valid = sorted(indices[env][mask[env]].tolist())
                assert valid == sorted(expected_steps), (
                    f"Step {step}, env {env}, pos ({rx},{ry}): "
                    f"expected {sorted(expected_steps)}, got {valid}"
                )
