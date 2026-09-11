import pytest
import torch

from src_new.memory.rollout_buffer import MemoryRolloutBuffer
from src_new.memory.types import MemoryWriteRecord


def _build_buffer(
    *,
    num_rollout_steps: int = 4,
    num_envs: int = 3,
    trxl_num_layers: int = 2,
    trxl_dim: int = 3,
    trxl_memory_length: int = 4,
    max_episode_steps: int = 6,
    grid_w: int = 2,
    grid_h: int = 2,
) -> MemoryRolloutBuffer:
    return MemoryRolloutBuffer(
        num_rollout_steps=num_rollout_steps,
        num_envs=num_envs,
        trxl_num_layers=trxl_num_layers,
        trxl_dim=trxl_dim,
        trxl_memory_length=trxl_memory_length,
        max_episode_steps=max_episode_steps,
        grid_w=grid_w,
        grid_h=grid_h,
        device=torch.device("cpu"),
    )


def _make_write_record(
    rb: MemoryRolloutBuffer,
    step: int,
    *,
    env_steps: torch.Tensor | None = None,
    mask_value: bool = True,
) -> MemoryWriteRecord:
    env_ids = torch.arange(rb.N, device=rb.device, dtype=torch.long)
    if env_steps is None:
        env_steps = torch.full((rb.N,), min(step, rb.m - 1), device=rb.device, dtype=torch.long)

    # Encode env and step into every memory token so cross-env mixups are easy to detect.
    base = env_ids[:, None, None].float() * 1000.0 + float(step)
    pure_memory_frame = base.expand(rb.N, rb.L, rb.D).clone()

    memory_indices_next = torch.arange(rb.M, device=rb.device, dtype=torch.long)[None, :].repeat(
        rb.N, 1
    )
    memory_indices_next = torch.clamp(memory_indices_next, max=rb.m - 1)

    memory_masks_next = torch.full((rb.N, rb.M), mask_value, device=rb.device, dtype=torch.bool)

    rf_base = env_ids[:, None, None, None].float() * 1000.0 + float(step)
    receptive_fields_scaled = rf_base.expand(rb.N, rb.L + 1, rb.grid_w, rb.grid_h).clone()
    receptive_fields_binary = receptive_fields_scaled > 0

    return MemoryWriteRecord(
        env_ids=env_ids,
        env_steps=env_steps,
        frame=pure_memory_frame,
        indices_next=memory_indices_next,
        masks_next=memory_masks_next,
        receptive_fields_scaled=receptive_fields_scaled,
        receptive_fields_binary=receptive_fields_binary,
        perceived_positions=torch.zeros((rb.N, 2), device=rb.device, dtype=torch.float32),
        retrieval_pos=torch.zeros((rb.N, 2), device=rb.device, dtype=torch.float32),
    )


def _write_step(
    rb: MemoryRolloutBuffer,
    step: int,
    *,
    env_steps: torch.Tensor | None = None,
    dones: torch.Tensor | None = None,
    mask_value: bool = True,
    envs_t: torch.Tensor | None = None,
) -> None:
    if dones is None:
        dones = torch.zeros((rb.N,), device=rb.device, dtype=torch.bool)
    record = _make_write_record(rb, step, env_steps=env_steps, mask_value=mask_value)
    if envs_t is None:
        envs_t = record.env_steps.clone()
    rb.add_step(
        global_step=step,
        memory_write_record=record,
        dones=dones,
        envs_t=envs_t,
    )


def _assert_case_a(rb: MemoryRolloutBuffer, window, row: int) -> None:
    """Assert case-A (done / bootstrap) retrieval properties for a single row."""
    # Masks must be all-False (nothing visible at rollout time)
    assert torch.all(~window.masks[row])
    # Indices are the default placeholder arange(M)
    assert torch.equal(
        window.indices_env[row],
        torch.arange(rb.M, device=window.indices_env.device),
    )
    # envs_t must be -1 (the "-1'th" step of the next episode)
    assert window.envs_t[row].item() == -1

    # Verify that case-A pure_memory_frames and rf equal those from
    # a normal (case-B) retrieval of the next global step.
    env_id = int(window.env_ids[row].item())
    done_step = int(window.global_steps[row].item())
    next_step = done_step + 1

    # Skip for global_step == -1 (bootstrap; get_step(-1) has special entry checks)
    if done_step == -1:
        return

    # Monkey-patch: ensure next_step is in-buffer and case-B for env_id,
    # then compare the two retrieval paths.  Write directly to rb.data to
    # bypass add_step validation.  Save & restore afterwards.
    orig_next_entry = rb.next_entry
    ring = next_step % rb._buffersize
    _fields = [
        "env_steps",
        "envs_t",
        "pure_memory_frames",
        "memory_indices_next",
        "memory_masks_next",
        "receptive_fields_scaled",
        "receptive_fields_binary",
        "dones",
    ]
    saved = {f: getattr(rb.data, f)[:, ring].clone() for f in _fields}

    try:
        # Force env_step=0, not-done for this env at the ring slot
        rb.data.env_steps[env_id, ring] = 0
        rb.data.dones[env_id, ring] = False
        # Advance next_entry so that next_step is considered "in buffer"
        if next_step >= rb.next_entry:
            rb.next_entry = next_step + 1

        case_a_win = rb.get_step(done_step)
        case_b_win = rb.get_step(next_step)
        assert torch.equal(
            case_a_win.frames[env_id],
            case_b_win.frames[env_id],
        ), "case-A pure_memory_frames != case-B next-step pure_memory_frames"
        assert torch.equal(
            case_a_win.receptive_fields_scaled[env_id],
            case_b_win.receptive_fields_scaled[env_id],
        ), "case-A rf_scaled != case-B next-step rf_scaled"
        assert torch.equal(
            case_a_win.receptive_fields_binary[env_id],
            case_b_win.receptive_fields_binary[env_id],
        ), "case-A rf_binary != case-B next-step rf_binary"
    finally:
        rb.next_entry = orig_next_entry
        for f, t in saved.items():
            getattr(rb.data, f)[:, ring] = t


class TestMemoryRolloutBuffer:
    def test_initialization(self):
        rb = _build_buffer()
        assert rb.next_entry == 0
        assert rb.data.env_steps.shape == (rb.N, rb._buffersize)
        assert rb.data.pure_memory_frames.shape == (rb.N, rb._buffersize, rb.L, rb.D)
        assert rb.data.memory_indices_next.shape == (rb.N, rb._buffersize, rb.M)
        assert rb.data.memory_masks_next.shape == (rb.N, rb._buffersize, rb.M)
        assert rb.data.receptive_fields_scaled.shape == (
            rb.N,
            rb._buffersize,
            rb.L + 1,
            rb.grid_w,
            rb.grid_h,
        )
        assert rb.data.receptive_fields_binary.shape == (
            rb.N,
            rb._buffersize,
            rb.L + 1,
            rb.grid_w,
            rb.grid_h,
        )
        assert rb.data.dones.shape == (rb.N, rb._buffersize)

    def test_add_step(self):
        rb = _build_buffer()
        record = _make_write_record(rb, step=0)
        dones = torch.tensor([False, True, False], device=rb.device)

        rb.add_step(
            global_step=0, memory_write_record=record, dones=dones, envs_t=record.env_steps.clone()
        )

        assert rb.next_entry == 1
        ring = 0
        assert torch.equal(rb.data.env_steps[:, ring], record.env_steps)
        assert torch.equal(rb.data.pure_memory_frames[:, ring], record.frame)
        assert torch.equal(rb.data.memory_indices_next[:, ring], record.indices_next)
        assert torch.equal(rb.data.memory_masks_next[:, ring], record.masks_next)
        assert torch.equal(rb.data.receptive_fields_scaled[:, ring], record.receptive_fields_scaled)
        assert torch.equal(rb.data.receptive_fields_binary[:, ring], record.receptive_fields_binary)
        assert torch.equal(rb.data.dones[:, ring], dones)

    def test_get_step(self):
        rb = _build_buffer(max_episode_steps=5)
        for step in range(3):
            _write_step(rb, step)

        window = rb.get_step(2)
        assert window.frames.shape == (rb.N, rb.m, rb.L, rb.D)
        assert window.indices_env.shape == (rb.N, rb.M)
        assert window.masks.shape == (rb.N, rb.M)
        assert window.receptive_fields_scaled.shape == (rb.N, rb.m, rb.L + 1, rb.grid_w, rb.grid_h)
        assert window.receptive_fields_binary.shape == (rb.N, rb.m, rb.L + 1, rb.grid_w, rb.grid_h)

        # For env_steps=t, valid history is [0..t], then future slots are zero padded.
        assert torch.all(window.frames[:, 3:] == 0)
        assert torch.all(window.receptive_fields_scaled[:, 3:] == 0)
        assert torch.all(window.receptive_fields_binary[:, 3:] == 0)

        expected_indices = torch.arange(rb.M, dtype=torch.long)[None, :].repeat(rb.N, 1)
        expected_indices = torch.clamp(expected_indices, max=rb.m - 1)
        assert torch.equal(window.indices_env, expected_indices)
        assert torch.all(window.masks)

    def test_get_batch(self):
        rb = _build_buffer(max_episode_steps=6)
        for step in range(4):
            _write_step(rb, step)

        envs = torch.tensor([2, 0, 1], dtype=torch.long)
        steps = torch.tensor([3, 2, 1], dtype=torch.long)
        window = rb.get_window_by_pairs(envs, steps)

        assert window.frames.shape[0] == 3
        assert torch.equal(window.env_ids, envs)
        assert torch.equal(window.global_steps, steps)

        # Row order must follow input pair order exactly.
        row0_last = window.frames[0, steps[0], 0, 0].item()
        row1_last = window.frames[1, steps[1], 0, 0].item()
        row2_last = window.frames[2, steps[2], 0, 0].item()
        assert row0_last == 2003.0
        assert row1_last == 2.0
        assert row2_last == 1001.0

    def test_adding_more_than_buffer_size_steps_overwrites_old_steps(self):
        rb = _build_buffer(num_rollout_steps=3)
        total_steps = 4 * rb._buffersize
        for step in range(total_steps):
            _write_step(rb, step)

        too_old = total_steps - rb._buffersize - 1
        with pytest.raises(ValueError, match="too old"):
            rb.get_step(too_old)

    def test_case_a_and_b_dont_mix_env_ordering(self):
        # Build a done/not-done run-length pattern with Fibonacci segment lengths:
        # done(1), not_done(1), done(2), not_done(3), done(5), ...
        num_envs = 12
        rb = _build_buffer(num_envs=num_envs, max_episode_steps=5)
        _write_step(rb, 0, dones=torch.zeros((num_envs,), dtype=torch.bool))

        fib = [1, 1]
        while sum(fib) < num_envs:
            fib.append(fib[-1] + fib[-2])

        done_pattern_list: list[bool] = []
        for run_idx, run_len in enumerate(fib):
            run_is_done = run_idx % 2 == 0
            done_pattern_list.extend([run_is_done] * run_len)
            if len(done_pattern_list) >= num_envs:
                break

        done_pattern = torch.tensor(done_pattern_list[:num_envs], dtype=torch.bool)
        _write_step(rb, 1, dones=done_pattern)

        envs = torch.arange(num_envs, dtype=torch.long)
        steps = torch.full((num_envs,), 1, dtype=torch.long)
        window = rb.get_window_by_pairs(envs, steps)

        assert torch.equal(window.env_ids, envs)
        assert torch.equal(window.global_steps, steps)

        # Order of rows must remain identical to input order despite A/B split.
        observed_done = torch.tensor(
            [torch.all(window.frames[row] == 0).item() for row in range(num_envs)],
            dtype=torch.bool,
        )
        assert torch.equal(observed_done, done_pattern)

        for row in range(num_envs):
            if done_pattern[row]:
                _assert_case_a(rb, window, row)
            else:
                assert window.frames[row, 1, 0, 0].item() == row * 1000.0 + 1.0
                assert window.masks[row].all()

    def test_get_minus_one_will_return_correct_memory(self):
        rb = _build_buffer(num_rollout_steps=10)
        window = rb.get_step(-1)

        _assert_case_a(rb, window, row=0)

        # test correctness after steps are added

        _write_step(rb, 0)
        _write_step(rb, 1)

        _assert_case_a(rb, rb.get_step(-1), row=0)

    def test_all_postdecessing_steps_are_0(self):
        rb = _build_buffer(max_episode_steps=6)
        # Construct a valid sequence that yields env_steps [0, 2, 4] at step 4.
        _write_step(
            rb,
            0,
            env_steps=torch.tensor([0, 0, 0], dtype=torch.long),
            dones=torch.tensor([False, True, False]),
        )
        _write_step(
            rb,
            1,
            env_steps=torch.tensor([1, 0, 1], dtype=torch.long),
            dones=torch.tensor([False, True, False]),
        )
        _write_step(
            rb,
            2,
            env_steps=torch.tensor([2, 0, 2], dtype=torch.long),
            dones=torch.tensor([False, False, False]),
        )
        _write_step(
            rb,
            3,
            env_steps=torch.tensor([3, 1, 3], dtype=torch.long),
            dones=torch.tensor([True, False, False]),
        )
        env_steps = torch.tensor([0, 2, 4], dtype=torch.long)
        _write_step(rb, 4, env_steps=env_steps)

        window = rb.get_step(4)
        for env in range(rb.N):
            last_valid = int(env_steps[env].item())
            if last_valid + 1 < rb.m:
                assert torch.all(window.frames[env, last_valid + 1 :] == 0)
                assert torch.all(window.receptive_fields_scaled[env, last_valid + 1 :] == 0)
                assert torch.all(window.receptive_fields_binary[env, last_valid + 1 :] == 0)

    def test_all_memories_come_from_same_env(self):
        # Hard stress test: many rollouts, random done flags, random pair retrievals in random order.
        rb = _build_buffer(
            num_rollout_steps=12,
            num_envs=17,
            trxl_num_layers=2,
            trxl_dim=4,
            trxl_memory_length=6,
            max_episode_steps=12,
        )

        gen = torch.Generator().manual_seed(12345)
        num_rollouts = 50
        total_steps = num_rollouts * rb.num_rollout_steps

        env_ids = torch.arange(rb.N, device=rb.device, dtype=torch.long)
        memory_indices_next = torch.arange(rb.M, device=rb.device, dtype=torch.long)[
            None, :
        ].repeat(rb.N, 1)
        memory_indices_next = torch.clamp(memory_indices_next, max=rb.m - 1)

        # Keep env_steps as the per-env step counter since last done; cap at m-1.
        env_steps = torch.zeros((rb.N,), device=rb.device, dtype=torch.long)

        for global_step in range(total_steps):
            dones = torch.rand((rb.N,), generator=gen, device=rb.device) < 0.20

            # Every memory frame for env e is filled with value e.
            pure_memory_frame = env_ids[:, None, None].float().expand(rb.N, rb.L, rb.D).clone()
            receptive_fields_scaled = (
                env_ids[:, None, None, None]
                .float()
                .expand(rb.N, rb.L + 1, rb.grid_w, rb.grid_h)
                .clone()
            )

            record = MemoryWriteRecord(
                env_ids=env_ids,
                env_steps=env_steps.clone(),
                frame=pure_memory_frame,
                indices_next=memory_indices_next,
                masks_next=torch.ones((rb.N, rb.M), device=rb.device, dtype=torch.bool),
                receptive_fields_scaled=receptive_fields_scaled,
                receptive_fields_binary=torch.ones(
                    (rb.N, rb.L + 1, rb.grid_w, rb.grid_h), device=rb.device, dtype=torch.bool
                ),
                perceived_positions=torch.zeros((rb.N, 2), device=rb.device, dtype=torch.float32),
                retrieval_pos=torch.zeros((rb.N, 2), device=rb.device, dtype=torch.float32),
            )
            rb.add_step(
                global_step=global_step,
                memory_write_record=record,
                dones=dones,
                envs_t=env_steps.clone(),
            )

            env_steps = torch.where(dones, torch.zeros_like(env_steps), env_steps + 1)
            env_steps = torch.clamp(env_steps, max=rb.m - 1)

        current_global_step = rb.next_entry - 1
        current_window = rb.get_step(current_global_step)
        ring_current = current_global_step % rb._buffersize

        # Validate current-step retrieval for all envs.
        for env in range(rb.N):
            is_done = bool(rb.data.dones[env, ring_current].item())
            if is_done:
                _assert_case_a(rb, current_window, env)
                continue

            last_valid = int(rb.data.env_steps[env, ring_current].item())
            assert torch.all(current_window.frames[env, : last_valid + 1] == float(env))

        # Random retrievals from the last rollout only, with many env IDs in random orders.
        start_last_rollout = rb.next_entry - rb.num_rollout_steps
        num_random_queries = 40
        batch_size = 64

        for _ in range(num_random_queries):
            envs = torch.randint(
                0, rb.N, (batch_size,), generator=gen, device=rb.device, dtype=torch.long
            )
            global_steps = torch.randint(
                start_last_rollout,
                rb.next_entry,
                (batch_size,),
                generator=gen,
                device=rb.device,
                dtype=torch.long,
            )
            order = torch.randperm(batch_size, generator=gen, device=rb.device)
            envs = envs[order]
            global_steps = global_steps[order]

            window = rb.get_window_by_pairs(envs, global_steps)
            assert torch.equal(window.env_ids, envs)
            assert torch.equal(window.global_steps, global_steps)

            for row in range(batch_size):
                env = int(envs[row].item())
                step = int(global_steps[row].item())
                ring = step % rb._buffersize
                is_done = bool(rb.data.dones[env, ring].item())

                if is_done:
                    _assert_case_a(rb, window, row)
                    continue

                last_valid = int(rb.data.env_steps[env, ring].item())
                assert torch.all(window.frames[row, : last_valid + 1] == float(env))

    def test_get_window_by_pairs_preserves_order_and_indices_retrieve_correct_env_memory(self):
        rb = _build_buffer(num_envs=6, max_episode_steps=10, trxl_memory_length=4)
        idx_pattern = torch.tensor([2, 1, 0, 3], dtype=torch.long)
        mask_pattern = torch.tensor([True, True, True, False], dtype=torch.bool)

        for step in range(7):
            env_ids = torch.arange(rb.N, device=rb.device, dtype=torch.long)
            env_steps = torch.full((rb.N,), step, dtype=torch.long, device=rb.device)
            base = env_ids[:, None, None].float() * 1000.0 + float(step)
            record = MemoryWriteRecord(
                env_ids=env_ids,
                env_steps=env_steps,
                frame=base.expand(rb.N, rb.L, rb.D).clone(),
                indices_next=idx_pattern[None, :].repeat(rb.N, 1),
                masks_next=mask_pattern[None, :].repeat(rb.N, 1),
                receptive_fields_scaled=base[:, :, :, None]
                .expand(rb.N, rb.L + 1, rb.grid_w, rb.grid_h)
                .clone(),
                receptive_fields_binary=torch.ones(
                    (rb.N, rb.L + 1, rb.grid_w, rb.grid_h), dtype=torch.bool, device=rb.device
                ),
                perceived_positions=torch.zeros((rb.N, 2), device=rb.device, dtype=torch.float32),
                retrieval_pos=torch.zeros((rb.N, 2), device=rb.device, dtype=torch.float32),
            )
            rb.add_step(
                global_step=step,
                memory_write_record=record,
                dones=torch.zeros((rb.N,), dtype=torch.bool),
                envs_t=env_steps,
            )

        envs = torch.tensor([5, 2, 4, 1, 3, 0], dtype=torch.long)
        steps = torch.tensor([6, 5, 4, 3, 6, 5], dtype=torch.long)
        window = rb.get_window_by_pairs(envs, steps)

        assert torch.equal(window.env_ids, envs)
        assert torch.equal(window.global_steps, steps)

        for row in range(envs.shape[0]):
            env = int(envs[row].item())
            step = int(steps[row].item())

            assert torch.equal(window.indices_env[row], idx_pattern)
            assert torch.equal(window.masks[row], mask_pattern)

            valid_positions = torch.where(mask_pattern)[0]
            for pos in valid_positions.tolist():
                env_idx = int(window.indices_env[row, pos].item())
                value = float(window.frames[row, env_idx, 0, 0].item())
                expected = env * 1000.0 + float(env_idx)
                assert value == expected, (
                    f"Wrong memory retrieved for row={row}, env={env}, step={step}, "
                    f"token_pos={pos}, env_idx={env_idx}."
                )

    def test_getter_output_dtypes_including_bootstrap(self):
        rb = _build_buffer(max_episode_steps=6)

        def _assert_window_types(window):
            assert window.frames.dtype == torch.float32
            assert window.indices_env.dtype == torch.long
            assert window.masks.dtype == torch.bool
            assert window.receptive_fields_scaled.dtype == torch.float32
            assert window.receptive_fields_binary.dtype == torch.bool
            assert window.env_ids.dtype == torch.long
            assert window.global_steps.dtype == torch.long

        # bootstrap getter
        bootstrap = rb.get_step(-1)
        _assert_window_types(bootstrap)

        # normal single-step getter
        _write_step(rb, 0)
        _write_step(rb, 1)
        one_step = rb.get_step(1)
        _assert_window_types(one_step)

        # pair getter
        envs = torch.tensor([2, 0, 1], dtype=torch.long)
        steps = torch.tensor([1, 1, 0], dtype=torch.long)
        by_pairs = rb.get_window_by_pairs(envs, steps)
        _assert_window_types(by_pairs)

    def test_add_step_rejects_invalid_inputs(self):
        rb = _build_buffer(num_envs=3, max_episode_steps=8)

        valid_record = _make_write_record(
            rb, step=0, env_steps=torch.zeros((rb.N,), dtype=torch.long)
        )
        valid_dones = torch.zeros((rb.N,), dtype=torch.bool)

        valid_envs_t = torch.zeros((rb.N,), dtype=torch.long)
        with pytest.raises(ValueError, match="Out of order"):
            rb.add_step(
                global_step=1,
                memory_write_record=valid_record,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )

        # env_ids check needs next_entry > 0 (the early return at next_entry==0 skips it)
        rb.add_step(
            global_step=0, memory_write_record=valid_record, dones=valid_dones, envs_t=valid_envs_t
        )
        bad_env_ids = _make_write_record(
            rb, step=1, env_steps=torch.ones((rb.N,), dtype=torch.long)
        )
        bad_env_ids.env_ids = torch.tensor([0, 2, 1], dtype=torch.long)
        with pytest.raises(ValueError, match="Environment IDs do not match"):
            rb.add_step(
                global_step=1,
                memory_write_record=bad_env_ids,
                dones=valid_dones,
                envs_t=torch.ones((rb.N,), dtype=torch.long),
            )

        # Fresh buffer for remaining shape/dtype checks (need global_step == next_entry == 0)
        rb = _build_buffer(num_envs=3, max_episode_steps=8)
        valid_record = _make_write_record(
            rb, step=0, env_steps=torch.zeros((rb.N,), dtype=torch.long)
        )
        valid_dones = torch.zeros((rb.N,), dtype=torch.bool)
        valid_envs_t = torch.zeros((rb.N,), dtype=torch.long)

        bad_dones_shape = torch.zeros((rb.N, 1), dtype=torch.bool)
        with pytest.raises(ValueError, match="dones must have shape"):
            rb.add_step(
                global_step=0,
                memory_write_record=valid_record,
                dones=bad_dones_shape,
                envs_t=valid_envs_t,
            )

        bad_dones_dtype = torch.zeros((rb.N,), dtype=torch.long)
        with pytest.raises(ValueError, match="dones must have dtype"):
            rb.add_step(
                global_step=0,
                memory_write_record=valid_record,
                dones=bad_dones_dtype,
                envs_t=valid_envs_t,
            )

        bad_env_steps0 = _make_write_record(
            rb, step=0, env_steps=torch.ones((rb.N,), dtype=torch.long)
        )
        with pytest.raises(ValueError, match="global step 0"):
            rb.add_step(
                global_step=0,
                memory_write_record=bad_env_steps0,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )

        bad_shape_record = _make_write_record(
            rb, step=0, env_steps=torch.zeros((rb.N,), dtype=torch.long)
        )
        bad_shape_record.frame = torch.zeros((rb.N, rb.L, rb.D, 1), dtype=torch.float32)
        with pytest.raises(ValueError, match="pure_memory_frame must have shape"):
            rb.add_step(
                global_step=0,
                memory_write_record=bad_shape_record,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )

        bad_dtype_record = _make_write_record(
            rb, step=0, env_steps=torch.zeros((rb.N,), dtype=torch.long)
        )
        bad_dtype_record.indices_next = bad_dtype_record.indices_next.float()
        with pytest.raises(ValueError, match="memory_indices_next must have dtype"):
            rb.add_step(
                global_step=0,
                memory_write_record=bad_dtype_record,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )

        bad_dtype_record2 = _make_write_record(
            rb, step=0, env_steps=torch.zeros((rb.N,), dtype=torch.long)
        )
        bad_dtype_record2.masks_next = bad_dtype_record2.masks_next.long()
        with pytest.raises(ValueError, match="memory_masks_next must have dtype"):
            rb.add_step(
                global_step=0,
                memory_write_record=bad_dtype_record2,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )

        bad_dtype_record3 = _make_write_record(
            rb, step=0, env_steps=torch.zeros((rb.N,), dtype=torch.long)
        )
        bad_dtype_record3.receptive_fields_binary = (
            bad_dtype_record3.receptive_fields_binary.float()
        )
        with pytest.raises(ValueError, match="receptive_fields_binary must have dtype"):
            rb.add_step(
                global_step=0,
                memory_write_record=bad_dtype_record3,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )

        bad_dtype_record4 = _make_write_record(
            rb, step=0, env_steps=torch.zeros((rb.N,), dtype=torch.long)
        )
        bad_dtype_record4.env_steps = bad_dtype_record4.env_steps.float()
        with pytest.raises(ValueError, match="env_steps must have dtype"):
            rb.add_step(
                global_step=0,
                memory_write_record=bad_dtype_record4,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )

        # Now verify env_steps ordering consistency across steps and dones transitions.
        rb.add_step(
            global_step=0,
            memory_write_record=valid_record,
            dones=torch.tensor([True, False, False]),
            envs_t=valid_envs_t,
        )

        inconsistent = _make_write_record(
            rb, step=1, env_steps=torch.tensor([1, 0, 1], dtype=torch.long)
        )
        with pytest.raises(ValueError, match="env_steps are inconsistent"):
            rb.add_step(
                global_step=1,
                memory_write_record=inconsistent,
                dones=valid_dones,
                envs_t=valid_envs_t,
            )
