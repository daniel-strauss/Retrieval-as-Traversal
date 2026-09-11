import pytest
import torch

from src_new.model.retrieval.spatial_index import SpatialIndex

MAX_EPISODE_STEPS_FOR_TESTS = 300


@pytest.fixture(params=[False, True], ids=["dict", "tensor"], autouse=True)
def spatial_backend(request, monkeypatch):
    monkeypatch.setattr(SpatialIndex, "store_as_tensor", request.param)


# Patch SpatialIndex constructor so tests that don't pass max_episode_steps still work
_orig_init = SpatialIndex.__init__


def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("max_episode_steps", MAX_EPISODE_STEPS_FOR_TESTS)
    _orig_init(self, *args, **kwargs)


SpatialIndex.__init__ = _patched_init  # type: ignore[method-assign]


# Helper: build a spawn-relative perceived_fov tensor (N, 2W-1, 2H-1)
# from a list of (env_idx, spawn_rel_x, spawn_rel_y) triples.
def _make_fov(
    num_envs: int, grid_w: int, grid_h: int, cells: list[tuple[int, int, int]]
) -> torch.Tensor:
    W, H = 2 * grid_w - 1, 2 * grid_h - 1
    ox, oy = grid_w - 1, grid_h - 1
    fov = torch.zeros((num_envs, W, H), dtype=torch.bool)
    for env, sx, sy in cells:
        fov[env, sx + ox, sy + oy] = True
    return fov


def test_retrieve_correct_memory_frames():
    # spawn-relative positions: env0 sees (1,1) and (2,2) at step 0, (1,1) at step 1, (3,3) at step 2
    # env1 sees (0,0) at step 0 and (4,4) at step 2
    grid_w, grid_h = 5, 5
    device = torch.device("cpu")
    spatial_index = SpatialIndex(num_envs=2, grid_w=grid_w, grid_h=grid_h, device=device)

    for step in range(3):
        cells: list[tuple[int, int, int]] = []
        if step == 0:
            cells += [(0, 1, 1), (0, 2, 2), (1, 0, 0)]
        elif step == 1:
            cells += [(0, 1, 1)]
        elif step == 2:
            cells += [(0, 3, 3), (1, 4, 4)]
        fov = _make_fov(2, grid_w, grid_h, cells)
        spatial_index.add_fov(perceived_fov=fov, env_steps=torch.tensor([step] * 2))

    # Retrieve: env0 at spawn-rel (1,1), env1 at spawn-rel (4,4)
    positions = torch.tensor([[1, 1], [4, 4]])
    memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(positions=positions)

    assert memory_indices.shape == (2, 2)
    assert torch.equal(memory_indices[0], torch.tensor([0, 1]))
    assert torch.equal(memory_indices[1], torch.tensor([2, -1]))
    assert torch.equal(memory_mask[0], torch.tensor([True, True]))
    assert torch.equal(memory_mask[1], torch.tensor([True, False]))


def test_retrieve_unvisited():
    grid_w, grid_h = 5, 5
    device = torch.device("cpu")
    spatial_index = SpatialIndex(num_envs=1, grid_w=grid_w, grid_h=grid_h, device=device)

    fov = _make_fov(1, grid_w, grid_h, [(0, 1, 1)])
    spatial_index.add_fov(perceived_fov=fov, env_steps=torch.tensor([0]))

    # (2,2) was never seen
    positions = torch.tensor([[2, 2]])
    memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(positions=positions)
    assert memory_indices.shape == (1, 0)
    assert memory_mask.shape == (1, 0)


def test_retrieve_multiple_memories():
    grid_w, grid_h = 5, 5
    device = torch.device("cpu")
    spatial_index = SpatialIndex(num_envs=1, grid_w=grid_w, grid_h=grid_h, device=device)

    for step in range(3):
        fov = _make_fov(1, grid_w, grid_h, [(0, 1, 1)])
        spatial_index.add_fov(perceived_fov=fov, env_steps=torch.tensor([step]))

    # (2,1) never visited
    positions = torch.tensor([[2, 1]])
    memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(positions=positions)
    assert memory_indices.shape == (1, 0)

    # Multi-env: both see (1,1) for 3 steps
    spatial_index = SpatialIndex(num_envs=2, grid_w=grid_w, grid_h=grid_h, device=device)
    for step in range(3):
        fov = _make_fov(2, grid_w, grid_h, [(0, 1, 1), (1, 1, 1)])
        spatial_index.add_fov(perceived_fov=fov, env_steps=torch.tensor([step] * 2))

    # env0 retrieves (1,1) → 3 hits, env1 retrieves (0,0) → 0 hits
    positions = torch.tensor([[1, 1], [0, 0]])
    memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(positions=positions)
    assert memory_indices.shape == (2, 3)
    assert torch.equal(memory_indices, torch.tensor([[0, 1, 2], [-1, -1, -1]]))
    assert torch.equal(memory_mask, torch.tensor([[True, True, True], [False, False, False]]))


def test_reset_deletes_correctly():
    grid_w, grid_h = 5, 5
    device = torch.device("cpu")
    spatial_index = SpatialIndex(num_envs=2, grid_w=grid_w, grid_h=grid_h, device=device)

    fov = _make_fov(2, grid_w, grid_h, [(0, 1, 1), (1, 1, 1)])
    spatial_index.add_fov(perceived_fov=fov, env_steps=torch.tensor([0] * 2))

    # Reset env 0 only
    spatial_index.reset_envs(dones=torch.tensor([True, False]))

    positions = torch.tensor([[1, 1], [1, 1]])
    memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(positions=positions)

    assert memory_indices.shape == (2, 1)
    assert torch.equal(memory_indices[0], torch.tensor([-1]))
    assert torch.equal(memory_indices[1], torch.tensor([0]))
    assert torch.equal(memory_mask[0], torch.tensor([False]))
    assert torch.equal(memory_mask[1], torch.tensor([True]))


def test_negative_spawn_relative_coords():
    """Agent spawns in the middle of the grid, moves to cells with negative spawn-relative coords."""
    grid_w, grid_h = 6, 6
    device = torch.device("cpu")
    spatial_index = SpatialIndex(num_envs=1, grid_w=grid_w, grid_h=grid_h, device=device)

    # Agent at spawn-relative (-2, -3)
    fov = _make_fov(1, grid_w, grid_h, [(0, -2, -3)])
    spatial_index.add_fov(perceived_fov=fov, env_steps=torch.tensor([0]))

    positions = torch.tensor([[-2, -3]])
    memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(positions=positions)
    assert memory_indices.shape == (1, 1)
    assert torch.equal(memory_indices[0], torch.tensor([0]))
    assert torch.equal(memory_mask[0], torch.tensor([True]))

    # Out-of-dict-range retrieval returns empty (via .get fallback)
    positions_oob = torch.tensor([[-5, -5]])
    mi, mm = spatial_index.get_memory_indices_and_mask(positions=positions_oob)
    assert mi.shape == (1, 0)


def test_many_asynchronous_resets():
    # Stress-test with random fovs and async resets, all in spawn-relative coords.
    num_envs = 3
    grid_w = 5
    grid_h = grid_w
    device = torch.device("cpu")
    spatial_index = SpatialIndex(num_envs=num_envs, grid_w=grid_w, grid_h=grid_h, device=device)
    W, H = 2 * grid_w - 1, 2 * grid_h - 1
    ox, oy = grid_w - 1, grid_h - 1
    t = 200
    torch.manual_seed(42)
    random_dones = torch.rand(t, num_envs) > 0.2
    all_fovs = torch.rand((t, num_envs, W, H)) > 0.5  # spawn-relative fov

    # Reference model: spawn-relative keys
    ref: list[dict[tuple[int, int], set[int]]] = [
        {(x, y): set() for x in range(-(grid_w - 1), grid_w) for y in range(-(grid_h - 1), grid_h)}
        for _ in range(num_envs)
    ]
    next_step = [0] * num_envs

    for step in range(t):
        fov = all_fovs[step]
        env_steps = torch.tensor([next_step[e] for e in range(num_envs)])
        spatial_index.add_fov(perceived_fov=fov, env_steps=env_steps)

        # Update reference
        for env in range(num_envs):
            for ti, tj in zip(*torch.where(fov[env])):
                sx, sy = int(ti.item()) - ox, int(tj.item()) - oy
                ref[env][(sx, sy)].add(next_step[env])
            next_step[env] += 1

        dones = random_dones[step]
        spatial_index.reset_envs(dones=dones)

        for env in torch.where(dones)[0]:
            env = int(env)
            ref[env] = {
                (x, y): set()
                for x in range(-(grid_w - 1), grid_w)
                for y in range(-(grid_h - 1), grid_h)
            }
            next_step[env] = 0

        # Check retrieval for random spawn-relative positions
        for _ in range(50):
            positions = torch.randint(-(grid_w - 1), grid_w, (num_envs, 2))
            memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(
                positions=positions
            )

            for env in range(num_envs):
                x, y = int(positions[env, 0]), int(positions[env, 1])
                expected_steps = sorted(ref[env][(x, y)])
                valid = memory_indices[env][memory_mask[env]].tolist()
                assert valid == expected_steps, (
                    f"Step {step}, env {env}, pos ({x},{y}): expected {expected_steps}, got {valid}"
                )
                invalid = memory_indices[env][~memory_mask[env]].tolist()
                assert all(v == -1 for v in invalid), (
                    f"Step {step}, env {env}: padded entries should be -1, got {invalid}"
                )


def test_agent_one_visited_retrieved_step_more_often_than_agent_2():
    # Agent 0 sees spawn-rel (1,1) at steps 0,1,2; agent 1 only at step 0.
    grid_w, grid_h = 5, 5
    device = torch.device("cpu")
    spatial_index = SpatialIndex(num_envs=2, grid_w=grid_w, grid_h=grid_h, device=device)

    for step in range(3):
        cells: list[tuple[int, int, int]] = [(0, 1, 1)]
        if step == 0:
            cells.append((1, 1, 1))
        fov = _make_fov(2, grid_w, grid_h, cells)
        spatial_index.add_fov(perceived_fov=fov, env_steps=torch.tensor([step] * 2))

    positions = torch.tensor([[1, 1], [1, 1]])
    memory_indices, memory_mask = spatial_index.get_memory_indices_and_mask(positions=positions)

    assert memory_indices.shape == torch.Size([2, 3])
    assert memory_mask.shape == torch.Size([2, 3])
    assert (memory_indices[0] == torch.tensor([0, 1, 2])).all()
    assert (memory_indices[1] == torch.tensor([0, -1, -1])).all()
    assert (memory_mask[0] == torch.tensor([True, True, True])).all()
    assert (memory_mask[1] == torch.tensor([True, False, False])).all()
