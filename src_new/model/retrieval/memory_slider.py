import warnings

import torch


class MemorySlider:
    """Handles episodic memory management for Transformer-XL based agents.

    This class manages:
    - Sliding window memory indices for attention
    - Memory masks for causal attention
    - Storage of episode memories across rollout steps
    - Batching of memories for training

    Uses a circular buffer of size M (trxl_memory_length) per environment.
    Spatial retrievals and the completed temporal step are written into the ring;
    the read-out is the last M entries, all of which are valid past steps at
    steady state (no wasted slot for "current step" unlike TemporalMemory).
    """

    def __init__(
        self,
        num_envs: int,
        num_rollout_steps: int,
        max_episode_steps: int,
        trxl_memory_length: int,
        trxl_num_layers: int,
        trxl_dim: int,
        device: torch.device,
    ):
        self.N = num_envs
        self.num_rollout_steps = num_rollout_steps
        self.m = max_episode_steps
        self.M = trxl_memory_length
        self.trxl_num_layers = trxl_num_layers
        self.trxl_dim = trxl_dim
        self.device = device

        # Current episodic memory for each environment
        # Shape: (num_envs, max_episode_steps, num_layers, dim)
        self.next_memory = torch.zeros(
            (num_envs, max_episode_steps, trxl_num_layers, trxl_dim),
            dtype=torch.float32,
        )

        # Circular buffer of env-step indices.  Initialised to arange(M) so that
        # unwritten slots match TemporalMemory's padding at early timesteps.
        self.buffer = torch.arange(self.M).repeat(self.N, 1)  # (N, M)

        # Total number of writes per env.  Shape (N, 1) for broadcasting.
        self.write_pos = torch.zeros((self.N, 1), dtype=torch.long)

        # Mask template: (M+1, M) lower-tri with diagonal=-1.
        # Row i has i valid entries (True), so row 0 = all-False, row M = all-True.
        # Unlike TemporalMemory's (M, M) template (which wastes 1 slot for the
        # "current step"), the slider only writes completed steps, so all M slots
        # can be valid at steady state.
        self.memory_mask_template = torch.tril(torch.ones((self.M + 1, self.M)), diagonal=-1).bool()

        self._last_step = torch.full((self.N,), -1, dtype=torch.long)
        self.env_idxs = torch.arange(self.N)

    def get_memory_indices_and_mask(
        self,
        retrieval_step: torch.Tensor,
        spatial_indexes: torch.Tensor,
        spatial_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Writes current step into the ring buffer and returns the memory window.

        Args:
            retrieval_step: (num_envs,) per-env episode step to prepare the window for.
                Typically ``current_step + 1`` from the caller (the +1 pre-computes
                the window that will be consumed at the *next* forward pass).
                A value of 1 signals the start of a new episode (reset).
            spatial_indexes: (num_envs, k) tensor of spatial indexes, where k is the highest number
                of spatial memories an environment retrieves at this step
            spatial_masks: (num_envs, k) tensor of spatial masks
        Returns:
            memory_indices: (num_envs, trxl_memory_length) tensor of memory indices.
            memory_masks: (num_envs, trxl_memory_length) bool tensor of memory masks.
        """
        k = spatial_indexes.shape[1]
        if k >= self.M:
            warnings.warn("More spatial memories than fit in the memory window.")

        # Reset envs starting a new episode
        reset_envs = retrieval_step == 1
        self.buffer[reset_envs] = (
            torch.arange(self.M).unsqueeze(0).expand(reset_envs.sum().item(), -1)
        )
        self.write_pos[reset_envs] = 0
        self._last_step[reset_envs] = 0

        # Sequential order check
        if not torch.all(retrieval_step == self._last_step + 1):
            raise ValueError(
                f"out of order retrieval encountered. Got retrieval_step={retrieval_step}, "
                f"last_step={self._last_step}"
            )

        # Hard check: no env should exceed max_episode_steps
        if torch.any(retrieval_step > self.m):
            raise ValueError(
                f"retrieval_step exceeds max_episode_steps ({self.m}). "
                f"Got max={retrieval_step.max().item()}"
            )

        current_step = retrieval_step - 1  # the completed step to write

        # --- Write spatial indices into the circular buffer ---
        if k > 0:
            offsets = torch.arange(k).unsqueeze(0)  # (1, k)
            positions = (self.write_pos + offsets) % self.M  # (N, k)
            env_exp = self.env_idxs.unsqueeze(1).expand_as(positions)
            self.buffer[env_exp[spatial_masks], positions[spatial_masks]] = spatial_indexes[
                spatial_masks
            ]
            self.write_pos += spatial_masks.sum(dim=1, keepdim=True)

        # --- Write completed step into the ring ---
        self.buffer[self.env_idxs, (self.write_pos.squeeze(1) % self.M)] = current_step
        self.write_pos += 1

        # --- Read window: M slots, oldest first ---
        logical_start = torch.clamp(self.write_pos - self.M, min=0)  # (N, 1)
        ring_pos = (logical_start + torch.arange(self.M).unsqueeze(0)) % self.M  # (N, M)
        memory_indices = torch.gather(self.buffer, dim=1, index=ring_pos)

        # --- Mask: write_pos selects template row; row M = all-True (full window) ---
        mask_row = torch.clamp(self.write_pos.squeeze(1), min=0, max=self.M)  # (N,)
        memory_masks = self.memory_mask_template[mask_row]  # (N, M)

        self._last_step += 1

        return memory_indices, memory_masks
