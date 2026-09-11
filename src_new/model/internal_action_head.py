"""Pluggable internal-action heads for spatial memory retrieval.

Each head is an nn.Module whose forward() returns (action, log_prob, entropy)
with uniform shapes so AgentModule and Trainer never need type-switches.

Shapes returned by forward():
    action:   (N, 2)   — the raw action (meaning depends on head type)
    log_prob: (N,)     — scalar log-prob per sample (summed over action dims)
    entropy:  (N,)     — scalar entropy per sample (summed over action dims)
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


class InternalActionHead(nn.Module, ABC):
    """Base class for all spatial retrieval heads."""

    @abstractmethod
    def forward(
        self, x: torch.Tensor, action: torch.Tensor | None = None,
        retrieval_pos_encoded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, trxl_dim) post-transformer features.
            action: (N, 2) actions to evaluate, or None to sample new ones.
            retrieval_pos_encoded: (N, trxl_dim) encoded previous retrieval position,
                or None when grid_cell_encoding is disabled.
        Returns:
            (action, log_prob, entropy)
        """
        ...

    @abstractmethod
    def action_to_position(self, action: torch.Tensor, prev_position: torch.Tensor) -> torch.Tensor:
        """
        Convert raw action tensor to integer grid position (N, 2).
        Must make sure that all positions are inside bounds of the gird.
        """
        ...


class GaussianSpatialHead(InternalActionHead):
    """Continuous Normal(mu, sigma) over (x, y) in spawn-relative coords."""

    def __init__(self, trxl_dim: int, grid_w: int, grid_h: int, grid_cell_encoding: bool = False, hidden_size: int = 0):
        super().__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        input_dim = trxl_dim * 2 if grid_cell_encoding else trxl_dim
        if hidden_size > 0:
            self.hidden = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
            )
            head_input_dim = hidden_size
        else:
            self.hidden = None
            head_input_dim = input_dim
        self.mu_head = nn.Linear(head_input_dim, 2)
        self.log_sigma_head = nn.Linear(head_input_dim, 2)
        nn.init.orthogonal_(self.mu_head.weight, np.sqrt(0.01))
        nn.init.orthogonal_(self.log_sigma_head.weight, np.sqrt(0.01))
        nn.init.constant_(self.log_sigma_head.bias, 1.0)  # σ ≈ 2.7 at init

    def forward(self, x, action=None, retrieval_pos_encoded=None):
        if retrieval_pos_encoded is not None:
            x = torch.cat([x, retrieval_pos_encoded], dim=-1)
        if self.hidden is not None:
            x = self.hidden(x)
        mu = self.mu_head(x)  # (N, 2)
        log_sigma = self.log_sigma_head(x).clamp(-2.3, math.log(max(self.grid_w, self.grid_h)))
        sigma = log_sigma.exp()
        dist = Normal(mu, sigma)
        if action is None:
            action = dist.rsample()  # (N, 2)
        log_prob = dist.log_prob(action).sum(dim=-1)  # (N,)
        entropy = dist.entropy().sum(dim=-1)  # (N,)
        return action, log_prob, entropy

    def action_to_position(self, action, prev_position):
        unclamped = action.round().long()
        clamped = torch.clamp(
            unclamped,
            min=torch.tensor([-(self.grid_w - 1), -(self.grid_h - 1)], device=action.device),
            max=torch.tensor([self.grid_w - 1, self.grid_h - 1], device=action.device),
        )
        return clamped


class CategoricalXYHead(InternalActionHead):
    """Two independent Categorical distributions over x ∈ [0, grid_w) and y ∈ [0, grid_h)."""

    def __init__(self, trxl_dim: int, grid_w: int, grid_h: int, grid_cell_encoding: bool = False, hidden_size: int = 0):
        super().__init__()
        # + w - 1 since starting position is unkown by the agent and agent specifies position
        # relative to the starting position, which can be anywhere in the grid.
        input_dim = trxl_dim * 2 if grid_cell_encoding else trxl_dim
        if hidden_size > 0:
            self.hidden = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
            )
            head_input_dim = hidden_size
        else:
            self.hidden = None
            head_input_dim = input_dim
        self.x_head = nn.Linear(head_input_dim, grid_w + grid_w - 1)
        self.y_head = nn.Linear(head_input_dim, grid_h + grid_h - 1)
        nn.init.orthogonal_(self.x_head.weight, np.sqrt(0.01))
        nn.init.orthogonal_(self.y_head.weight, np.sqrt(0.01))
        self.grid_w = grid_w
        self.grid_h = grid_h

    def forward(self, x, action=None, retrieval_pos_encoded=None):
        if retrieval_pos_encoded is not None:
            x = torch.cat([x, retrieval_pos_encoded], dim=-1)
        if self.hidden is not None:
            x = self.hidden(x)
        x_dist = Categorical(logits=self.x_head(x))  # over grid_w
        y_dist = Categorical(logits=self.y_head(x))  # over grid_h
        if action is None:
            ax = x_dist.sample()  # (N,)
            ay = y_dist.sample()  # (N,)
            action = torch.stack([ax, ay], dim=-1)  # (N, 2)
        log_prob = x_dist.log_prob(action[:, 0]) + y_dist.log_prob(action[:, 1])  # (N,)
        entropy = x_dist.entropy() + y_dist.entropy()  # (N,)
        return action, log_prob, entropy

    def action_to_position(self, action: torch.Tensor, prev_position: torch.Tensor) -> torch.Tensor:
        center = torch.tensor([self.grid_w - 1, self.grid_h - 1], device=action.device)
        return action.long() - center  # already integer grid coords


class PositionMovingHead(InternalActionHead):
    """3x3 directional move: action ∈ {0..8} mapped to (dx, dy) ∈ {-1,0,1}².

    The agent chooses a single categorical action from 9 options.
    action_to_position applies the delta to the *previous* retrieval position
    (maintained externally by Agent).
    """

    # Lookup: action_id → (dx, dy)
    _DELTAS = torch.tensor(
        [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ]
    )  # (9, 2)

    def __init__(self, trxl_dim: int, grid_w: int, grid_h: int, grid_cell_encoding: bool = False, hidden_size: int = 0):
        super().__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        input_dim = trxl_dim * 2 if grid_cell_encoding else trxl_dim
        if hidden_size > 0:
            self.hidden = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
            )
            head_input_dim = hidden_size
        else:
            self.hidden = None
            head_input_dim = input_dim
        self.head = nn.Linear(head_input_dim, len(self._DELTAS))  # logits for 9 actions
        nn.init.orthogonal_(self.head.weight, np.sqrt(0.01))

    def forward(self, x, action=None, retrieval_pos_encoded=None):
        if retrieval_pos_encoded is not None:
            x = torch.cat([x, retrieval_pos_encoded], dim=-1)
        if self.hidden is not None:
            x = self.hidden(x)
        dist = Categorical(logits=self.head(x))  # 9 classes
        if action is None:
            action_id = dist.sample()  # (N,)
        else:
            action_id = action.squeeze(-1) if action.dim() > 1 else action  # (N,)
        log_prob = dist.log_prob(action_id)  # (N,)
        entropy = dist.entropy()  # (N,)
        # Store as (N, 1) so the Action dataclass shape is consistent
        return action_id.unsqueeze(-1), log_prob, entropy

    def action_to_position(self, action, prev_position) -> torch.Tensor:
        """Apply delta to previous position.

        Args:
            action: (N, 1) action ids
            prev_position: (N, 2) previous retrieval position (spawn-relative)
        """
        
        deltas = self._DELTAS.to(action.device)
        delta = deltas[action.squeeze(-1).long()]  # (N, 2)

        ret_pos =  prev_position + delta
        ret_pos = torch.clamp(
            ret_pos,
            min=torch.tensor([-(self.grid_w - 1), -(self.grid_h - 1)], device=action.device),
            max=torch.tensor([self.grid_w - 1, self.grid_h - 1], device=action.device),
        )
        return ret_pos



# ── Factory ──────────────────────────────────────────────────────────────

INTERNAL_HEAD_REGISTRY: dict[str, type[InternalActionHead]] = {
    "gaussian": GaussianSpatialHead,
    "categorical_xy": CategoricalXYHead,
    "position_moving": PositionMovingHead,
}


def make_internal_head(name: str, trxl_dim: int, grid_w: int, grid_h: int, grid_cell_encoding: bool = False, hidden_size: int = 0) -> InternalActionHead:
    if name not in INTERNAL_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown internal head '{name}'. Available: {list(INTERNAL_HEAD_REGISTRY.keys())}"
        )
    return INTERNAL_HEAD_REGISTRY[name](trxl_dim, grid_w, grid_h, grid_cell_encoding=grid_cell_encoding, hidden_size=hidden_size)
