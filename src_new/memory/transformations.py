"""
This file contains transformations of memory windows.
During training the trainer might transform the memory windows to add some exploration noise
or to apply the memory masks.
"""

from src_new.memory.types import MemoryWindow
from src_new.trainer.trajectory import Minibatch


def apply_memory_masks_(mw: MemoryWindow):
    pmfs = mw.frames
    rfs_scaled = mw.receptive_fields_scaled
    rfs_binary = mw.receptive_fields_binary
    mi = mw.indices_env
    memory_masks = mw.masks  # (N, M)

    # eplore masks
    # pmfs[~memory_masks] = 0.0
    # rfs_scaled[~memory_masks] = 0.0
    # rfs_binary[~memory_masks] = False

    mw.frames = pmfs * memory_masks.unsqueeze(-1).unsqueeze(-1)
    mw.indices_env = mi * memory_masks
    mw.receptive_fields_scaled = rfs_scaled * memory_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    mw.receptive_fields_binary = rfs_binary & memory_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def add_hindsight_at_0_(mw: MemoryWindow, mb: Minibatch):
    """
    Set all memory masks to True, where the env, global step is the first env step

    """

    # find indices where env step is 0
    start_indixes = mb.envs_t == 0

    mw.masks[start_indixes] = True

    if start_indixes.any():
        pass
    # M = mw.memory_masks.shape[1]
    # mw.memory_indices_env[start_indixes] = torch.arange(M, device=mw.memory_indices_env.device).unsqueeze(0).expand_as(mw.memory_indices_env[start_indixes])
