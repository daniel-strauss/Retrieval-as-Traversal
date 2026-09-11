from dataclasses import dataclass

import torch


@dataclass
class MemoryWindow:
    """
    Per-step context bundle that accompanies the observation through the forward pass.
    """

    frames: torch.Tensor  # (B, M, L, D)
    indices_env: torch.Tensor  # (B, M)
    masks: torch.Tensor  # (B, M) # TOD continue gere m or M
    receptive_fields_scaled: torch.Tensor  # (B, M, L+1, W, H)
    receptive_fields_binary: torch.Tensor  # (B, M, L+1, W, H)
    env_ids: torch.Tensor  # (B,)
    envs_t: torch.Tensor  # (B,)
    global_steps: torch.Tensor  # (B,)
    # (N, M, 2) perceived positions at time of memory creation, for grid cell encoding
    perceived_positions: torch.Tensor
    # (B, 2) memory retrieval position chosen at time of memory creation (same time as envs_t).
    # Zeros when spatial retrieval is disabled.
    retrieval_pos: torch.Tensor


@dataclass
class MemoryWriteRecord:
    env_ids: torch.Tensor  # (N,)
    env_steps: torch.Tensor  # (N,) (debug/sanity)
    frame: torch.Tensor  # (N, L, D) (produced at t)
    indices_next: torch.Tensor  # (N, M) (retrieval plan generated at t)
    masks_next: torch.Tensor  # (N, M) (retrieval plan generated at t)
    receptive_fields_scaled: torch.Tensor  # (N, L+1, W, H)
    receptive_fields_binary: torch.Tensor  # (N, L+1, W, H)
    # (N, M, 2) perceived positions at time of memory creation, for grid cell encoding
    perceived_positions: torch.Tensor
    # (N, 2) memory retrieval position chosen at time of memory creation (same time as envs_t).
    # Zeros when spatial retrieval is disabled.
    retrieval_pos: torch.Tensor
