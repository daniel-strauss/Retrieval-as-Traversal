from dataclasses import dataclass

import torch


@dataclass
class ForwardDiagnostics:
    """Optional side-channel filled during forward passes for visualization.

    Created by the caller only when recording is active, passed into
    forward methods, and filled in-place. Costs nothing when ``None``.
    """

    # everything here refers to env_0

    # (L+1, D): rows 0..L-1 are inputs to each transformer layer,
    # row L is the final transformer output.
    layer_activations: torch.Tensor | None = None

    # position empedding of the currently generated memory frame
    pe_current_token: torch.Tensor | None = None
