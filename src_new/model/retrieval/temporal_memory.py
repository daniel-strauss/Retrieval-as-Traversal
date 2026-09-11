
import torch

class TemporalMemory:
    """Handles episodic memory management for Transformer-XL based agents.

    This class manages:
    - Sliding window memory indices for attention
    - Memory masks for causal attention
    - Storage of episode memories across rollout steps
    - Batching of memories for training


    TODO there is confusion with the two types of timestep.

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
        self.num_envs = num_envs
        self.num_rollout_steps = num_rollout_steps
        self.max_episode_steps = max_episode_steps
        self.trxl_memory_length = trxl_memory_length
        self.trxl_num_layers = trxl_num_layers
        self.trxl_dim = trxl_dim
        self.device = device

        # Current episodic memory for each environment
        # Shape: (num_envs, max_episode_steps, num_layers, dim)
        self.next_memory = torch.zeros(
            (num_envs, max_episode_steps, trxl_num_layers, trxl_dim),
            dtype=torch.float32,
        )

        # Build template tensors (computed once)
        self._build_templates()

    def _build_templates(self):
        # Generate episodic memory mask used in attention
        self.memory_mask_template = torch.tril(
            torch.ones((self.trxl_memory_length, self.trxl_memory_length)), diagonal=-1
        ).bool()
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions = torch.repeat_interleave(
            torch.arange(0, self.trxl_memory_length).unsqueeze(0),
            self.trxl_memory_length - 1,
            dim=0,
        ).long()
        memory_indices = torch.stack(
            [
                torch.arange(i, i + self.trxl_memory_length)
                for i in range(self.max_episode_steps - self.trxl_memory_length + 1)
            ]
        ).long()
        self.memory_indices_template = torch.cat((repetitions, memory_indices))
        """ e.g. the memory window indices tensor looks like this if 
                memory_length = 4 and max_episode_length = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """

    def _build_templates_NEW_MIGHT_HAVE_BUG(self):
        """Build the memory mask and sliding window index templates."""
        """
        OLD
        # Lower triangular memory mask template
        # e.g. for memory_length=4:
        # 0, 0, 0, 0
        # 1, 0, 0, 0
        # 1, 1, 0, 0
        # 1, 1, 1, 0
        self.memory_mask_template = torch.tril(
            torch.ones((self.trxl_memory_length, self.trxl_memory_length)),
            diagonal=-1,
        )
        """

        """
        OLD
        # Sliding window indices template
        # e.g. for memory_length=4 and max_episode_steps=7:
        # TODO isnt there a 0,1,2,3 - line missing here as the first mask just has 0s?
        # 0, 1, 2, 3  (step 0)
        # 0, 1, 2, 3  (step 1)
        # 0, 1, 2, 3  (step 2)
        # 0, 1, 2, 3  (step 3)
        # 1, 2, 3, 4  (step 4)
        # 2, 3, 4, 5  (step 5)
        # 3, 4, 5, 6  (step 6)
        repetitions = torch.repeat_interleave(
            torch.arange(0, self.trxl_memory_length).unsqueeze(0),
            self.trxl_memory_length - 1,
            dim=0,
        ).long()
        sliding_indices = torch.stack(
            [
                torch.arange(i, i + self.trxl_memory_length)
                for i in range(self.max_episode_steps - self.trxl_memory_length + 1)
            ]
        ).long()
        self.memory_indices_template = torch.cat((repetitions, sliding_indices))
        """

        # flipped Lower triangular memory mask template
        # e.g. for memory_length=4:
        # 0, 0, 0, 0
        # 0, 0, 0, 1
        # 0, 0, 1, 1
        # 0, 1, 1, 1
        # 1, 1, 1, 1
        self.memory_mask_template = torch.tril(
            torch.ones((self.trxl_memory_length + 1, self.trxl_memory_length)),
            diagonal=-1,
        ).flip(dims=[1])

        # Sliding window indices template
        # e.g. for memory_length=4 and max_episode_steps=7:
        # -1,-1,-1,-1  (step 0)
        # -1,-1,-1, 0  (step 1)
        # -1,-1, 0, 1  (step 2)
        # -1, 0, 1, 2  (step 3)
        # 0, 1, 2, 3  (step 4)
        # 1, 2, 3, 4  (step 5)
        # 2, 3, 4, 5  (step 6)
        # 3, 4, 5, 6  (step 7)

        relative_window = torch.repeat_interleave(
            torch.arange(-self.trxl_memory_length, 0).unsqueeze(0),
            self.max_episode_steps,
            dim=0,
        ).long()

        self.memory_indices_template = torch.clip(
            relative_window + torch.arange(self.max_episode_steps)[:, None], min=-1
        )

    def get_memory_indices_and_mask(self, episode_steps_envs) -> tuple[torch.Tensor, torch.Tensor]:
        """Get memory indices for all environments based on their current episode steps.

        Args:
            episode_steps_envs: (N) tensor of current episode steps for each environment in the batch.
            memory_mask: (N, trxl_memory_length) tensor of memory masks for each environment.
            memory_indices: (N, trxl_memory_length) tensor of memory indices for each environment.
        Returns:
            memory_indices with appropriate clipping for steps < memory length.
        """

        if torch.any(episode_steps_envs >= self.max_episode_steps + 1):
            ## TODO handle properly
            raise ValueError(
                f"Episode step exceeds max episode steps. "
                f"Got {episode_steps_envs.max().item()}, but max episode steps is "
                f"{self.max_episode_steps}"
            )

        episode_steps_envs_clamped = torch.clamp(episode_steps_envs, max=self.max_episode_steps - 1)
        # clip only needed for mask not indices
        clipped_steps = torch.clip(
            episode_steps_envs_clamped,
            0,
            self.trxl_memory_length - 1,  # -1 because we are using the old memory mask template
        )

        memory_indices = self.memory_indices_template[episode_steps_envs_clamped]

        # sanity check to ensure that internal memory mask and memory indices match
        memory_mask = self.memory_mask_template[clipped_steps]

        # commented assertion because reverted to old memory mask
        # assert torch.all((memory_indices != -1) == memory_mask.bool()),\
        #    "Error: memory mask and memory indices do not match."

        return memory_indices, memory_mask