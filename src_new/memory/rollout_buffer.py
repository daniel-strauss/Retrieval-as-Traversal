import warnings
from dataclasses import dataclass

import torch

from src_new.memory.types import MemoryWindow, MemoryWriteRecord


@dataclass
class Memory:
    # The memory if the MemoryRolloutBuffer. Put into an extra dataclass for better readability.
    #
    # N = num_envs, M = trxl_memory_length, k buffer size in rollout domain, L = num_layers, D = hidden_dim
    # m = max_episode_steps
    # below k*M is just in the case of when MemoryRolloutBoffer is storing the memory.
    # In the case of retrieval, it has a different size.
    #
    # ConsistencyRule Row t stores produced_at_t frame and retrieval_plan_for_t_plus_1.

    env_steps: torch.Tensor  # N,k*M {in [m]} : needed to retrieve the env step from the global step
    # (we pass only the env step to agent module)
    pure_memory_frames: torch.Tensor  # N,k*M,L,D {in R}
    memory_indices_next: torch.Tensor  # N,k*M,M {in [m]}
    memory_masks_next: torch.Tensor  # N,k*M,M {in {0,1}}
    receptive_fields_scaled: torch.Tensor  # N, k*M, L+1, grid_w, grid_h {in R}
    receptive_fields_binary: torch.Tensor  # N, k*M, L+1, grid_w, grid_h {in {0,1}}
    perceived_positions: (
        torch.Tensor
    )  # N, k*M, 2 {in R}, for grid cell encoding, can be None if not used
    retrieval_pos: torch.Tensor  # N, k*M, 2 — memory retrieval position at time of creation

    envs_t: torch.Tensor  # N, k*M {in [m]} : the env step at which each item was added
    dones: torch.Tensor  # N, k*M {in {0,1}} needed to mark the end of an episode


class MemoryRolloutBuffer:
    """
    Authorative source of all memories.
    Memories are indexed by env_id, global_step.
        - memorie indixes should be converted to blobal before storage
        - memory maks not


    After completion of the n'th training episode, it delets the n-num_stored_rollouts+1'th rollout to
    always have the last rollout in the buffer.


    Storage rule: MemoryRolloutBuffer holds data that is collected during rollout,
    needed during training (PPO update), AND still needed after training
    (persists across rollout boundaries). Contrast with Trajectory, which
    stores data that is needed during training but NOT after.

    """

    # number of rollout windows stored. 2 sores the current and the last rollout
    num_stored_rollouts = 2

    def __init__(
        self,
        num_rollout_steps: int,
        num_envs: int,
        trxl_num_layers: int,
        trxl_dim: int,
        trxl_memory_length: int,
        max_episode_steps: int,
        grid_w: int,
        grid_h: int,
        device: torch.device,
    ):
        """
        Args:
            num_rollout_steps: number of steps in one rollout (i.e. the episode length in the rollout phase).
            num_envs: number of parallel environments.
            num_layers: number of transformer layers (L).
            hidden_dim: dimension of the hidden states (D).
            trxl_memory_length: number of tokens in the memory window (M).
            device: torch device to store the memory on.
        """

        if self.num_stored_rollouts < 2:
            raise ValueError(
                "num_stored_rollouts must be at least 2 to store the current and the last rollout."
            )
        if trxl_memory_length > max_episode_steps:
            raise ValueError(
                "trxl_memory_length cannot be larger than max_episode_steps, because the memory "
                "window larger than the episode history is not supported."
            )

        self.num_rollout_steps = num_rollout_steps
        self.N = num_envs
        self.L = trxl_num_layers
        self.D = trxl_dim
        self.M = trxl_memory_length
        self.m = max_episode_steps
        self.device = device
        self.grid_w = grid_w
        self.grid_h = grid_h

        self._buffersize: int = self.num_stored_rollouts * num_rollout_steps
        self.data: Memory = Memory(
            env_steps=torch.zeros((self.N, self._buffersize), device=device, dtype=torch.long),
            pure_memory_frames=torch.zeros(
                (self.N, self._buffersize, self.L, self.D), device=device, dtype=torch.float32
            ),
            memory_indices_next=torch.zeros(
                (self.N, self._buffersize, self.M), device=device, dtype=torch.long
            ),
            memory_masks_next=torch.zeros(
                (self.N, self._buffersize, self.M), device=device, dtype=torch.bool
            ),
            receptive_fields_scaled=torch.zeros(
                (self.N, self._buffersize, self.L + 1, self.grid_w, self.grid_h),
                device=device,
                dtype=torch.float32,
            ),
            receptive_fields_binary=torch.zeros(
                (self.N, self._buffersize, self.L + 1, self.grid_w, self.grid_h),
                device=device,
                dtype=torch.bool,
            ),
            envs_t=torch.zeros((self.N, self._buffersize), device=device, dtype=torch.long),
            dones=torch.zeros((self.N, self._buffersize), device=device, dtype=torch.bool),
            perceived_positions=torch.zeros(
                (self.N, self._buffersize, 2), device=device, dtype=torch.float32
            ),
            retrieval_pos=torch.zeros(
                (self.N, self._buffersize, 2), device=device, dtype=torch.long
            ),
        )

        self.next_entry = 0  # pointer to the next entry in data, in global timestep domain.

    #######
    # Main API
    #######

    def add_step(
        self,
        global_step: int,
        memory_write_record: MemoryWriteRecord,
        dones: torch.Tensor,
        envs_t: torch.Tensor,
    ):
        """
        Adds a memory write record for one step to the memory. This is called at every step of the rollout
        phase, and adds the memory write record for that step to the memory.
        """

        if global_step != self.next_entry:
            raise ValueError("Out of order write requested.")

        self._validate_write_inputs(memory_write_record, dones)

        self.data.env_steps[:, self.next_entry % self._buffersize] = memory_write_record.env_steps
        self.data.envs_t[:, self.next_entry % self._buffersize] = envs_t
        self.data.pure_memory_frames[:, self.next_entry % self._buffersize] = (
            memory_write_record.frame
        )
        self.data.memory_indices_next[:, self.next_entry % self._buffersize] = (
            memory_write_record.indices_next
        )
        self.data.memory_masks_next[:, self.next_entry % self._buffersize] = (
            memory_write_record.masks_next
        )
        self.data.receptive_fields_scaled[:, self.next_entry % self._buffersize] = (
            memory_write_record.receptive_fields_scaled
        )
        self.data.receptive_fields_binary[:, self.next_entry % self._buffersize] = (
            memory_write_record.receptive_fields_binary
        )
        self.data.perceived_positions[:, self.next_entry % self._buffersize] = (
            memory_write_record.perceived_positions
        )
        self.data.retrieval_pos[:, self.next_entry % self._buffersize] = (
            memory_write_record.retrieval_pos
        )
        self.data.dones[:, self.next_entry % self._buffersize] = dones
        self.next_entry += 1

    def get_step(self, global_step: int, exclude_out_of_episode: bool) -> MemoryWindow:
        """
        For a given global step, returns the memory window for that step containing all envs.
        For each env, at that global step, the memory window is constructed by retrieving
            - the pure memory frames at the global steps starting from the last start of episode
                (i.e. the global step - the env step) until the current global step.
            - the same for receptive fields
            - the memory masks and memory indices are retrieved for the current global step.
        """

        envs = torch.arange(self.N, device=self.device)
        global_steps = torch.full((self.N,), global_step, device=self.device)
        return self.get_window_by_pairs(envs, global_steps, exclude_out_of_episode)

    def get_window_by_pairs(
        self, envs: torch.Tensor, _global_steps: torch.Tensor, exclude_out_of_episode: bool
    ) -> MemoryWindow:
        """For a suite of (env, global_step) pairs, returns the memory window.

        args: 
            envs, global_steps: indies pairs to retrieve 
            exclude_out_of_episode_memories: True -> all memories that are not in the retrieved 
                                                    episode, are 0ed  
        
        Contract - what is zeroed and what is kept:

        PURE MEMORY FRAMES & RECEPTIVE FIELDS  (history domain, shape (B, m, ...))
        ──────────────────────────────────────────────────────────────────────────
        Position k in the history maps to  global_step = episode_start + k.

        For case B (requested step is NOT a done step):
            episode_start = requested_global_step - env_step_at_requested_step.

        For case A (requested step IS a done step or requested step is global step -1):
            episode_start = requested_global_step + 1
                        (the requested global step is basically the -1'th step of the NEXT episode).

        Kept:   all frames within the target episode (including future intra-episode steps)
        Zeroed: frames belonging to a different episode (after a done boundary) (iff exclude_out_of_episode_memories)
                frames not yet written to the buffer

        Case B - requested step is NOT a done step (normal case):
            requested:                  |
                 history pos k:   0     1     2     3     4     5     6     7
            data.env step:        0     1     2     3    DONE   0'    1'    —
            data.global step:     5     6     7     8     9    10    11    12
                                 ╰─── current episode ──╯     ╰─next ep─╯
            kept:                 ✓    ✓     ✓    ✓     ✓    ✗     ✗     ✗
            
        Case A - requested step IS a done step:
            The caller is at envs_t=-1 of the NEXT episode. Frames from the next
            episode are returned so that hindsight transforms can access them.
            Masks and indices are all-zero (nothing visible at rollout time).

            requested global_step = 9 (done step of old episode)
            episode_start = 10 (first step of next episode)

            requested:                              |
                 history pos k:   0     1     2     3     4     5     6     7
            data.env step:        0'    1'    2'   DONE'  0''   1''   2''   3''
            data.global step:    10    11    12    13     14    15    16    17
                                 ╰── last episode ──╯    ╰─ current episode ─╯
            kept:                 ✗    ✗     ✗    ✗     ✓    ✓     ✓    ✓
            


        MEMORY MASKS & INDICES  (retrieval domain, shape (B, M))
        ──────────────────────────────────────────────────────────
        Retrieved from stored values at the requested global_step.
        If requested step is a done step → all-False masks, default indices [0..M-1].
        """

        # check input shapes, can act as a suite of pairs
        if envs.ndim != 1 or _global_steps.ndim != 1:
            raise ValueError(
                f"Expected 1D envs/global_steps, got {envs.shape=} and {_global_steps.shape=}."
            )
        if envs.shape[0] != _global_steps.shape[0]:
            raise ValueError(
                f"envs and _global_steps must have same length, got {envs.shape[0]} and "
                f"{_global_steps.shape[0]}."
            )

        # check environments are in valid range
        if ((envs < 0) | (envs >= self.N)).any():
            bad = envs[(envs < 0) | (envs >= self.N)][0].item()
            raise ValueError(f"Invalid env id in envs: {bad}, valid range is [0, {self.N}).")

        # check global steps are in valid range
        self._check_global_step_in_buffer(_global_steps)

        in_frame_steps = _global_steps.clone()
        # identify done steps and update ring_global to point to the first step of the next episode for
        # those steps, so that we can handle them together with the normal steps in one retrieval.
        a_idx = self.data.dones[envs, in_frame_steps % self._buffersize].bool() | (
            in_frame_steps == -1
        )  # (B_a,)

        # for done steps, we read the "-1"'th step of the next episode
        in_frame_steps[a_idx] += 1
        # buffer indixes
        ring_in_frame = in_frame_steps % self._buffersize

        ########################################################
        # Retrieve pure memory frames and receptive fields
        ########################################################

        # retrieve the pure memory frames and receptive fields for the full history span (m)
        # for each env, r fetches first episodal step and the following m-1 steps

        # how many steps have passed since the start of the episode at the requested global step?
        episode_starts = torch.empty_like(ring_in_frame)
        episode_starts[~a_idx] = (
            in_frame_steps[~a_idx] - self.data.env_steps[envs[~a_idx], ring_in_frame[~a_idx]]
        )
        # we must overwrite the episode starts, since for case a steps might be out of buffer
        episode_starts[a_idx] = in_frame_steps[a_idx]

        # for each env the steps from env starts to +m, this will include out of buffer steps,
        # but those will be masked out later
        r = (
            envs[:, None],
            (episode_starts[:, None] + torch.arange(self.m, device=self.device)[None, :]),
        )
        r_ring = r[0], r[1] % self._buffersize

        # Note: temporary out of buffer garbage reads will be masked right afterwards
        pure_memory_frames = self.data.pure_memory_frames[r_ring]  # (B, m, L, D)
        rf_scaled = self.data.receptive_fields_scaled[r_ring]  # (B, m, L, W, H)
        rf_binary = self.data.receptive_fields_binary[r_ring]  # (B, m, L, W, H)
        perceived_positions = self.data.perceived_positions[r_ring]  # (B, m, 2)
        dones = self.data.dones[r_ring]  # (B, m)

        # apply out of buffer mask
        in_buffer = (self.next_entry - self._buffersize <= r[1]) & (r[1] < self.next_entry)
        pure_memory_frames *= in_buffer[:, :, None, None]
        rf_scaled *= in_buffer[:, :, None, None, None]
        rf_binary &= in_buffer[:, :, None, None, None]
        perceived_positions *= in_buffer[:, :, None]
        dones &= in_buffer

        # apply out of episode mask to pure memory frames and receptive fields
        # cumsum counts done events up to but not including j
        if exclude_out_of_episode:
            prior_dones = dones.int().cumsum(dim=1) - dones.int()  # (B, m)
            in_current_episode = prior_dones == 0  # True = same episode, False = next episode

            pure_memory_frames *= in_current_episode[:, :, None, None]
            rf_scaled *= in_current_episode[:, :, None, None, None]
            rf_binary &= in_current_episode[:, :, None, None, None]
            perceived_positions *= in_current_episode[:, :, None]
            dones &= in_current_episode

        ########################################################
        # Retrieve memory masks, indices envs_t
        ########################################################

        memory_masks = self.data.memory_masks_next[envs, ring_in_frame]  # (B, M)
        memory_indices = self.data.memory_indices_next[envs, ring_in_frame]  # (B, M)
        # for case A (done steps), we zero out the masks and set the indices to default [0..M-1]
        memory_masks[a_idx] = False
        num_as = a_idx.sum().item()
        if not isinstance(num_as, int):
            raise ValueError(f"num_as must be a scalar, got {num_as}.")
        memory_indices[a_idx] = (
            torch.arange(self.M, device=self.device).unsqueeze(0).expand(num_as, -1)
        )

        envs_t = self.data.envs_t[envs, ring_in_frame]  # (B,)
        # for case A (done steps), envs_t should be -1, because envs_t revers to the environment step, at
        # which the memory window was CREATED.
        envs_t[a_idx] = -1

        retrieval_pos = self.data.retrieval_pos[envs, ring_in_frame]  # (B, 2)
        # for case A (done steps), zero out — no previous retrieval position exists after reset
        retrieval_pos[a_idx] = 0

        ###############
        # Pack and return
        ###############

        return MemoryWindow(
            frames=pure_memory_frames,
            indices_env=memory_indices,
            masks=memory_masks,
            receptive_fields_scaled=rf_scaled,
            receptive_fields_binary=rf_binary,
            env_ids=envs,
            envs_t=envs_t,
            perceived_positions=perceived_positions,
            global_steps=_global_steps,
            retrieval_pos=retrieval_pos,
        )

    ########
    # Helper functions
    ########

    def _validate_write_inputs(
        self, memory_write_record: MemoryWriteRecord, dones: torch.Tensor
    ) -> None:

        #####
        # Dtype and shape sanity checks
        #####

        if dones.shape != (self.N,):
            raise ValueError(f"dones must have shape {(self.N,)}, got {tuple(dones.shape)}.")
        if dones.dtype != torch.bool:
            raise ValueError(f"dones must have dtype torch.bool, got {dones.dtype}.")

        if memory_write_record.env_ids.shape != (self.N,):
            raise ValueError(
                f"env_ids must have shape {(self.N,)}, got {tuple(memory_write_record.env_ids.shape)}."
            )
        if memory_write_record.env_steps.shape != (self.N,):
            raise ValueError(
                f"env_steps must have shape {(self.N,)}, got {tuple(memory_write_record.env_steps.shape)}."
            )
        if memory_write_record.frame.shape != (self.N, self.L, self.D):
            raise ValueError(
                f"pure_memory_frame must have shape {(self.N, self.L, self.D)}, "
                f"got {tuple(memory_write_record.frame.shape)}."
            )
        if memory_write_record.indices_next.shape != (self.N, self.M):
            raise ValueError(
                f"memory_indices_next must have shape {(self.N, self.M)}, "
                f"got {tuple(memory_write_record.indices_next.shape)}."
            )
        if memory_write_record.masks_next.shape != (self.N, self.M):
            raise ValueError(
                f"memory_masks_next must have shape {(self.N, self.M)}, "
                f"got {tuple(memory_write_record.masks_next.shape)}."
            )
        if memory_write_record.receptive_fields_scaled.shape != (
            self.N,
            self.L + 1,
            self.grid_w,
            self.grid_h,
        ):
            raise ValueError(
                f"receptive_fields_scaled must have shape {(self.N, self.L + 1, self.grid_w, self.grid_h)}, "
                f"got {tuple(memory_write_record.receptive_fields_scaled.shape)}."
            )
        if memory_write_record.receptive_fields_binary.shape != (
            self.N,
            self.L + 1,
            self.grid_w,
            self.grid_h,
        ):
            raise ValueError(
                f"receptive_fields_binary must have shape {(self.N, self.L + 1, self.grid_w, self.grid_h)}, "
                f"got {tuple(memory_write_record.receptive_fields_binary.shape)}."
            )

        if memory_write_record.env_ids.dtype != torch.long:
            raise ValueError(
                f"env_ids must have dtype torch.long, got {memory_write_record.env_ids.dtype}."
            )
        if memory_write_record.env_steps.dtype != torch.long:
            raise ValueError(
                f"env_steps must have dtype torch.long, got {memory_write_record.env_steps.dtype}."
            )
        if not torch.is_floating_point(memory_write_record.frame):
            raise ValueError("pure_memory_frame must be a floating tensor.")
        if memory_write_record.indices_next.dtype != torch.long:
            raise ValueError(
                f"memory_indices_next must have dtype torch.long, got {memory_write_record.indices_next.dtype}."
            )
        if memory_write_record.masks_next.dtype != torch.bool:
            raise ValueError(
                f"memory_masks_next must have dtype torch.bool, got {memory_write_record.masks_next.dtype}."
            )
        if not torch.is_floating_point(memory_write_record.receptive_fields_scaled):
            raise ValueError("receptive_fields_scaled must be a floating tensor.")
        if memory_write_record.receptive_fields_binary.dtype != torch.bool:
            raise ValueError(
                f"receptive_fields_binary must have dtype torch.bool, got {memory_write_record.receptive_fields_binary.dtype}."
            )

        ####
        # Value sanity checks
        #####

        if (memory_write_record.env_steps < 0).any():
            raise ValueError("env_steps must be non-negative.")

        if self.next_entry == 0:
            if (memory_write_record.env_steps != 0).any():
                raise ValueError(
                    f"At global step 0, all env_steps must be 0, but got"
                    f" {memory_write_record.env_steps}."
                )
            return

        prev_ring = (self.next_entry - 1) % self._buffersize
        prev_env_steps = self.data.env_steps[:, prev_ring]
        prev_dones = self.data.dones[:, prev_ring]

        expected_if_not_done = torch.minimum(
            prev_env_steps + 1,
            torch.full_like(prev_env_steps, self.m - 1),
        )
        expected = torch.where(prev_dones, torch.zeros_like(prev_env_steps), expected_if_not_done)

        if not torch.equal(memory_write_record.env_steps, expected):
            raise ValueError(
                "env_steps are inconsistent with previous row/dones. "
                "Expected 0 after previous done, otherwise +1 (capped at m-1)."
            )

        if not torch.all(memory_write_record.env_ids == torch.arange(self.N, device=self.device)):
            raise ValueError(
                f"Environment IDs do not match. Env_ids must be [0, 1, ..., N-1], got"
                f"{memory_write_record.env_ids}."
            )

    def _check_global_step_in_buffer(self, global_step: torch.Tensor | int):
        """
        Checks that the requested global step(s) are within the range of global steps currently stored
        in the buffer.
        """
        if isinstance(global_step, int):
            global_step = torch.tensor(global_step, device=self.device)

        if (global_step == -1).any() and self.next_entry not in range(self.num_rollout_steps + 1):
            raise ValueError(
                f"-1 retrieval if reserved for the 0'th retrieval in rollout buffer, or the "
                f"first training loop. Next entry is {self.next_entry}, num_rollout_steps is"
                f" {self.num_rollout_steps}."
            )

        if (global_step < -1).any():
            raise ValueError(
                f"Requested global step {global_step} is negative, which is invalid.\
                             (Exept for -1, which is reserved for the 0'th retrieval in rollout buffer)"
            )

        if (global_step < self.next_entry - self._buffersize).any():
            raise ValueError(
                f"Requested global step {global_step} is too old and has been overwritten. "
            )
        if (global_step >= self.next_entry).any():
            raise ValueError(f"Requested global step {global_step} has not been stored yet. ")

    def _empty_memory_window(self) -> MemoryWindow:
        """
        Returns an empty memory window, i.e. a memory window where all memory frames are 0,
        all memory masks are 0, and all memory indices are 0.
        """
        return MemoryWindow(
            frames=torch.zeros((self.N, self.m, self.L, self.D), device=self.device),
            indices_env=torch.arange(self.M, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand((self.N, self.M))
            .clone(),  # torch.zeros((self.N, self.M), device=self.device, dtype=torch.long),
            masks=torch.zeros((self.N, self.M), device=self.device, dtype=torch.bool),
            receptive_fields_scaled=torch.zeros(
                (self.N, self.m, self.L + 1, self.grid_w, self.grid_h), device=self.device
            ),
            receptive_fields_binary=torch.zeros(
                (self.N, self.m, self.L + 1, self.grid_w, self.grid_h),
                device=self.device,
                dtype=torch.bool,
            ),
            perceived_positions=torch.zeros(
                (self.N, self.m, 2), device=self.device, dtype=torch.float32
            ),
            env_ids=torch.arange(self.N, device=self.device),
            envs_t=torch.zeros((self.N,), device=self.device, dtype=torch.long),
            global_steps=torch.zeros((self.N,), device=self.device, dtype=torch.long),
            retrieval_pos=torch.zeros((self.N, 2), device=self.device, dtype=torch.float32),
        )
