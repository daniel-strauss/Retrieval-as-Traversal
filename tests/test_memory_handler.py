import pytest
import torch

from src_new.memory.handler import MemoryHandler
from src_new.memory.transformations import apply_memory_masks_
from src_new.memory.types import MemoryWindow, MemoryWriteRecord
from src_new.validation import validate_memory_window


def _build_handler(
    *,
    num_envs: int = 4,
    num_rollout_steps: int = 6,
    max_episode_steps: int = 10,
    trxl_memory_length: int = 4,
    trxl_num_layers: int = 2,
    trxl_dim: int = 3,
    grid_w: int = 2,
    grid_h: int = 2,
) -> MemoryHandler:
    return MemoryHandler(
        num_envs=num_envs,
        num_rollout_steps=num_rollout_steps,
        max_episode_steps=max_episode_steps,
        trxl_memory_length=trxl_memory_length,
        trxl_num_layers=trxl_num_layers,
        trxl_dim=trxl_dim,
        grid_w=grid_w,
        grid_h=grid_h,
        device=torch.device("cpu"),
    )


def _default_env_steps(handler: MemoryHandler, step: int) -> torch.Tensor:
    m = handler.memory_rollout_buffer.m
    return torch.full((handler.memory_rollout_buffer.N,), min(step, m - 1), dtype=torch.long)


def _make_write_record(
    handler: MemoryHandler,
    step: int,
    *,
    env_steps: torch.Tensor | None = None,
    memory_indices_next: torch.Tensor | None = None,
    memory_masks_next: torch.Tensor | None = None,
) -> MemoryWriteRecord:
    rb = handler.memory_rollout_buffer
    env_ids = torch.arange(rb.N, dtype=torch.long)
    if env_steps is None:
        env_steps = _default_env_steps(handler, step)

    # Encode env and step in stored frames to make retrieval source checks explicit.
    value = env_ids[:, None, None].float() * 1000.0 + float(step + 1)
    pure_memory_frame = value.expand(rb.N, rb.L, rb.D).clone()
    rf_scaled = value[:, :, :, None].expand(rb.N, rb.L + 1, rb.grid_w, rb.grid_h).clone()

    if memory_indices_next is None:
        memory_indices_next = torch.arange(rb.M, dtype=torch.long)[None, :].repeat(rb.N, 1)
    if memory_masks_next is None:
        memory_masks_next = torch.ones((rb.N, rb.M), dtype=torch.bool)

    return MemoryWriteRecord(
        env_ids=env_ids,
        env_steps=env_steps,
        frame=pure_memory_frame,
        indices_next=memory_indices_next,
        masks_next=memory_masks_next,
        receptive_fields_scaled=rf_scaled,
        receptive_fields_binary=torch.ones(
            (rb.N, rb.L + 1, rb.grid_w, rb.grid_h), dtype=torch.bool
        ),
        perceived_positions=torch.zeros((rb.N, 2), dtype=torch.float32),
        retrieval_pos=torch.zeros((rb.N, 2), dtype=torch.float32),
    )


def _add_step(
    handler: MemoryHandler,
    step: int,
    *,
    env_steps: torch.Tensor | None = None,
    memory_indices_next: torch.Tensor | None = None,
    memory_masks_next: torch.Tensor | None = None,
    dones: torch.Tensor | None = None,
) -> None:
    rb = handler.memory_rollout_buffer
    if dones is None:
        dones = torch.zeros((rb.N,), dtype=torch.bool)
    write = _make_write_record(
        handler,
        step,
        env_steps=env_steps,
        memory_indices_next=memory_indices_next,
        memory_masks_next=memory_masks_next,
    )
    handler.add_memory_write_record(step, write, dones, write.env_steps)


def _assert_row_fully_zero(window: MemoryWindow, row: int) -> None:
    assert torch.all(window.frames[row] == 0)
    assert torch.all(window.indices_env[row] == 0)
    assert torch.all(window.masks[row] == 0)
    assert torch.all(window.receptive_fields_scaled[row] == 0)
    assert torch.all(window.receptive_fields_binary[row] == 0)


class TestMemoryHandler:
    def test_bootstrap_rollout_window_is_zero_and_types_are_correct(self):
        handler = _build_handler()
        window = handler.get_memory_window_rollout(-1)

        assert torch.all(window.frames == 0)
        # Bootstrap (done/step=-1) sets indices to default arange(M) per the rollout buffer contract
        M = handler.memory_rollout_buffer.M
        expected_idx = torch.arange(M, dtype=torch.long)[None, :].expand(
            handler.memory_rollout_buffer.N, -1
        )
        assert torch.equal(window.indices_env, expected_idx)
        assert torch.all(window.masks == False)
        assert torch.all(window.receptive_fields_scaled == 0)
        assert torch.all(window.receptive_fields_binary == 0)

        assert window.frames.dtype == torch.float32
        assert window.indices_env.dtype == torch.long
        assert window.masks.dtype == torch.bool
        assert window.receptive_fields_scaled.dtype == torch.float32
        assert window.receptive_fields_binary.dtype == torch.bool

    def test_rollout_masking_zeros_only_the_masked_indices(self):
        handler = _build_handler(num_envs=3, max_episode_steps=8, trxl_memory_length=4)
        rb = handler.memory_rollout_buffer

        for step in range(6):
            _add_step(handler, step)

        # Retrieval plan used at step 6: unique indices so token->time mapping is unambiguous.
        indices = torch.tensor(
            [
                [0, 2, 4, 5],
                [1, 3, 4, 5],
                [0, 1, 3, 5],
            ],
            dtype=torch.long,
        )
        masks = torch.tensor(
            [
                [True, False, True, False],
                [False, True, True, False],
                [True, True, False, False],
            ],
            dtype=torch.bool,
        )
        _add_step(handler, 6, memory_indices_next=indices, memory_masks_next=masks)

        window = handler.get_memory_window_rollout(6)
        apply_memory_masks_(window)
        for env in range(rb.N):
            for tok in range(rb.M):
                t = int(indices[env, tok].item())
                got = float(window.frames[env, tok, 0, 0].item())
                if masks[env, tok]:
                    assert got == env * 1000.0 + float(t + 1)
                else:
                    assert got == 0.0

    def test_minibatch_getter_preserves_pair_order_and_done_rows_stay_zero(self):
        handler = _build_handler(num_envs=5, max_episode_steps=8)

        _add_step(handler, 0)
        _add_step(handler, 1)
        _add_step(
            handler, 2, dones=torch.tensor([False, True, False, True, False], dtype=torch.bool)
        )
        _add_step(
            handler,
            3,
            env_steps=torch.tensor([3, 0, 3, 0, 3], dtype=torch.long),
        )

        envs = torch.tensor([3, 4, 1, 0, 2], dtype=torch.long)
        steps = torch.tensor([2, 3, 2, 3, 3], dtype=torch.long)
        window = handler.get_memory_window_minibatch(envs, steps)
        apply_memory_masks_(window)

        assert torch.equal(window.env_ids, envs)
        assert torch.equal(window.global_steps, steps)

        # rows 0 and 2 are done rows at step=2
        _assert_row_fully_zero(window, 0)
        _assert_row_fully_zero(window, 2)

        # non-done rows should contain the right env-specific last written value at queried step.
        assert window.frames[1, 3, 0, 0].item() == 4004.0
        assert window.frames[3, 3, 0, 0].item() == 4.0
        assert window.frames[4, 3, 0, 0].item() == 2004.0

    def test_apply_memory_masks_can_be_disabled(self):
        handler = _build_handler(num_envs=1, trxl_memory_length=3, max_episode_steps=8)

        for step in range(4):
            _add_step(handler, step)

        idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
        msk = torch.tensor([[True, False, True]], dtype=torch.bool)
        _add_step(handler, 4, memory_indices_next=idx, memory_masks_next=msk)

        # Masks are no longer applied inside the handler; content is always preserved.
        window = handler.get_memory_window_rollout(4)

        # Without apply mask, indexed slot stays original content.
        assert window.frames[0, 1, 0, 0].item() == 3.0

    def test_validate_memory_window_rejects_wrong_dtype_bounds(self):
        handler = _build_handler(num_envs=2, trxl_memory_length=3, max_episode_steps=6)
        history_span = handler.memory_rollout_buffer.m

        good = MemoryWindow(
            frames=torch.ones((2, 3, 2, 3), dtype=torch.float32),
            indices_env=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            masks=torch.ones((2, 3), dtype=torch.bool),
            receptive_fields_scaled=torch.zeros((2, 3, 3, 2, 2), dtype=torch.float32),
            receptive_fields_binary=torch.zeros((2, 3, 3, 2, 2), dtype=torch.bool),
            env_ids=torch.tensor([0, 1], dtype=torch.long),
            envs_t=torch.tensor([0, 0], dtype=torch.long),
            global_steps=torch.tensor([0, 0], dtype=torch.long),
            perceived_positions=torch.zeros((2, 2), dtype=torch.float32),
            retrieval_pos=torch.zeros((2, 2), dtype=torch.float32),
        )
        validate_memory_window(good, history_span=history_span)

        bad_dtype = MemoryWindow(**{**good.__dict__, "masks": good.masks.long()})
        with pytest.raises(ValueError, match="memory_masks must be bool"):
            validate_memory_window(bad_dtype, history_span=history_span)

        bad_bounds = MemoryWindow(
            **{
                **good.__dict__,
                "indices_env": torch.tensor([[0, 1, 6], [0, 1, 2]], dtype=torch.long),
            }
        )
        with pytest.raises(ValueError, match="out of range"):
            validate_memory_window(bad_bounds, history_span=history_span)

    def test_randomized_masking_stress_on_minibatch_pairs(self):
        handler = _build_handler(
            num_envs=9,
            num_rollout_steps=12,
            max_episode_steps=20,
            trxl_memory_length=4,
            trxl_num_layers=2,
            trxl_dim=3,
        )
        rb = handler.memory_rollout_buffer
        gen = torch.Generator().manual_seed(20260325)

        # Keep step < m to avoid env_step saturation and keep expected values simple.
        for step in range(12):
            env_steps = torch.full((rb.N,), step, dtype=torch.long)
            idx_list = []
            mask_list = []
            for _env in range(rb.N):
                perm = torch.randperm(step + 1, generator=gen)
                picked = perm[: rb.M]
                if picked.shape[0] < rb.M:
                    picked = torch.cat([picked, picked.new_zeros((rb.M - picked.shape[0],))], dim=0)
                idx_list.append(picked)

                m = torch.rand((rb.M,), generator=gen) > 0.35
                mask_list.append(m)

            idx = torch.stack(idx_list, dim=0).long()
            msk = torch.stack(mask_list, dim=0).bool()
            _add_step(
                handler, step, env_steps=env_steps, memory_indices_next=idx, memory_masks_next=msk
            )

        query_count = 80
        envs = torch.randint(0, rb.N, (query_count,), generator=gen, dtype=torch.long)
        steps = torch.randint(3, 12, (query_count,), generator=gen, dtype=torch.long)
        order = torch.randperm(query_count, generator=gen)
        envs = envs[order]
        steps = steps[order]

        window = handler.get_memory_window_minibatch(envs, steps)
        apply_memory_masks_(window)
        assert torch.equal(window.env_ids, envs)
        assert torch.equal(window.global_steps, steps)

        for row in range(query_count):
            env = int(envs[row].item())
            step = int(steps[row].item())
            ring = step % rb._buffersize

            row_idx = rb.data.memory_indices_next[env, ring]
            row_msk = rb.data.memory_masks_next[env, ring]

            for tok in range(rb.M):
                t = int(row_idx[tok].item())
                got = float(window.frames[row, tok, 0, 0].item())
                if row_msk[tok]:
                    # With our write pattern and env_step=step (<m), time t maps to value env*1000+(t+1).
                    assert got == env * 1000.0 + float(t + 1)
                else:
                    assert got == 0.0
