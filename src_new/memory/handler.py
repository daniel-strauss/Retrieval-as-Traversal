import torch

from src_new.memory.rollout_buffer import MemoryRolloutBuffer
from src_new.memory.types import MemoryWindow, MemoryWriteRecord
from src_new.validation import validate_memory_window


class Injector:
    """
    It can boost training performance to add memories from the future during training time.
    """


class MemoryHandler:
    """
    Stores and retrieves memories, owns temporal and spatial memory.

    Vocab:
    - A "pure memory frame" the set the tokens of all layers for one environment for one timestep.
        i.e. a memory frame stores one forward pass
    - A "pure memory episode" is the set current set of all "memory frames" for one environent.

    - A "full memory frame" for an environment and a timestep is the retrieved memory frames at that
        timestep in  that environment + the memory frame created at that timestep by that environment.
    - A "full memory episode"


    This class provides the interface for Agent to retrieve and store memory from MemoryRolloutBuffer.
    The main functionalities are:
    - For rollout:
        - provide every environment with the pure memory, retrieved by the environment step
        - store every pure memory frame, and the receptive fields
        - raise an error if being used falsely (e.g. forgetting to store memories, overwriting memories)



    The Design:
    - TemporalMemory and SpatialMemory decide which memories to store
    - MemoryRolloutBuffer stores the data (indexed by global timestep)
        - CurrentMemoryAccess provides access to it by episode timestep
    - MemoryHandler orchestrates the above mentioned components.
        [TemporalMemory || SpatialMemory] <---MemoryHandler-----> [MemoryRolloutBuffer <-> CurrentMemoryAccess]
                                |                   |
                                |                   |
                              Agent           Agent, Trainer, etc.
    - It acts as the interface for the agent to interact with memory, and ensures correct usage
        (e.g. no forgetting or overwriting of memories, correct retrieval of memories with respect to
        episode timesteps, etc.)

    """

    def __init__(
        self,
        num_envs: int,
        num_rollout_steps: int,
        max_episode_steps: int,
        trxl_memory_length: int,
        trxl_num_layers: int,
        trxl_dim: int,
        grid_w: int,
        grid_h: int,
        device: torch.device,
    ):

        self.memory_rollout_buffer = MemoryRolloutBuffer(
            num_rollout_steps=num_rollout_steps,
            num_envs=num_envs,
            trxl_num_layers=trxl_num_layers,
            trxl_dim=trxl_dim,
            trxl_memory_length=trxl_memory_length,
            max_episode_steps=max_episode_steps,
            grid_w=grid_w,
            grid_h=grid_h,
            device=device,
        )

    ######
    # Public API
    ######

    def get_memory_window_rollout(
        self, global_step: int, exclude_out_of_episode: bool
    ) -> MemoryWindow:
        """
        Args:
            global_step: the global step for which to retrieve the current memories
            exclude_out_of_episode: if out of episode values should be set to 0
        Retrieves the memory window for every environment in the rollout at the given global step.
        The memory window includes
            - all pure memory frames + receptive fields created until now in the rollout
            - memory indices, memory masks, created in the last step to use in the current step for retrieval
        """

        memory_window = self.memory_rollout_buffer.get_step(
            global_step, exclude_out_of_episode=exclude_out_of_episode
        )
        memory_window = self._index_memory_window(memory_window)

        validate_memory_window(memory_window, history_span=self.memory_rollout_buffer.m)

        return memory_window

    def add_memory_write_record(
        self,
        global_step: int,
        write_record: MemoryWriteRecord,
        done_envs: torch.Tensor,
        envs_t: torch.Tensor,
    ):
        """
        Adds the memory write record for every environment in the rollout created at the given global step.
        """
        self.memory_rollout_buffer.add_step(global_step, write_record, done_envs, envs_t)

    def get_memory_window_minibatch(
        self, env_ids: torch.Tensor, global_steps: torch.Tensor, exclude_out_of_episode: bool
    ) -> MemoryWindow:
        """
        Retrieves the memory window for the given environment ids and global steps.

        - memorie indices and memory masks are used to index into the pure memory frames
            and global steps at the specific time steps
        - pure memory frames contains all memory frames created in that environment, including ones created
            after the given global steps.
        """

        memory_window = self.memory_rollout_buffer.get_window_by_pairs(
            env_ids, global_steps, exclude_out_of_episode
        )
        memory_window = self._index_memory_window(memory_window)

        validate_memory_window(memory_window, history_span=self.memory_rollout_buffer.m)

        return memory_window

    #######
    # Private methods
    #######

    def _index_memory_window(self, memory_window: MemoryWindow) -> MemoryWindow:
        """Convert history-domain tensors (B, m, ...) into retrieval-domain windows (B, M, ...)."""

        pure_memory_frames = memory_window.frames
        rfs_scaled = memory_window.receptive_fields_scaled
        rfs_binary = memory_window.receptive_fields_binary
        memory_indices = memory_window.indices_env

        B = pure_memory_frames.shape[0]
        batch_idx = torch.arange(B, device=pure_memory_frames.device)[:, None]

        pure_memory_frames_indexed = pure_memory_frames[batch_idx, memory_indices]
        rfs_scaled_indexed = rfs_scaled[batch_idx, memory_indices]
        rfs_binary_indexed = rfs_binary[batch_idx, memory_indices]
        positions_indexed = memory_window.perceived_positions[batch_idx, memory_indices]

        return MemoryWindow(
            frames=pure_memory_frames_indexed,
            indices_env=memory_window.indices_env,
            masks=memory_window.masks,
            receptive_fields_scaled=rfs_scaled_indexed,
            receptive_fields_binary=rfs_binary_indexed,
            perceived_positions=positions_indexed,
            env_ids=memory_window.env_ids,
            envs_t=memory_window.envs_t,
            global_steps=memory_window.global_steps,
            retrieval_pos=memory_window.retrieval_pos,
        )
