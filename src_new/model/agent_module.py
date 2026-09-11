from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src_new.config import TrainConfig
from src_new.memory.types import MemoryWindow
from src_new.model.internal_action_head import InternalActionHead, make_internal_head
from src_new.model.trxl import Transformer
from src_new.trainer.forward_diagnostics import ForwardDiagnostics


@dataclass
class Action:
    """Return type of ``get_action_and_value``.

    Bundles the external (env) action and the optional internal (memory
    retrieval) action together with their log-probabilities and entropy so
    that callers never need to unpack a parallel tuple of tensors.
    """

    external_action: torch.Tensor
    """Discrete env action, shape (N, num_branches)."""
    external_log_probs: torch.Tensor
    """Log-prob of external_action under the current policy, same shape."""
    external_entropy: torch.Tensor
    """Per-sample entropy of the external action distribution, shape (N,)."""
    internal_action: torch.Tensor | None = None
    """Memory retrieval (row, col) action, shape (N, 2), or None."""
    internal_log_probs: torch.Tensor | None = None
    """Log-prob of internal_action, shape (N, 2), or None."""
    internal_entropy: torch.Tensor | None = None
    """Per-sample entropy of the internal action distribution, shape (N,), or None."""
    retr_positions: torch.Tensor | None = None
    """(x, y) positions corresponding to the internal action, shape (N, 2), or None."""
    retrieval_hit: torch.Tensor | None = None
    """1.0 if the retrieval position had stored memories, 0.0 otherwise, shape (N,), or None."""


class AgentModule(nn.Module):
    """All neural network parameters for the PPO+TrXL agent.

    Owns the encoder, transformer, actor/critic heads, and optional
    spatial-memory heads and reconstruction decoder.  No rollout state
    or memory-handler references live here.
    """

    def __init__(
        self,
        obs_shape: tuple,
        action_space_shape: tuple,
        max_episode_steps: int,
        train_conf: TrainConfig,
        grid_w: int,
        grid_h: int,
    ):
        super().__init__()
        tc = train_conf  # for convenience

        self.obs_shape = obs_shape
        self.use_spatial_memory = tc.use_spatial_memory
        self.internal_head_warmup_steps = tc.internal_head_warmup_steps

        if len(obs_shape) > 1:
            self.encoder = nn.Sequential(
                self.layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                self.layer_init(nn.Linear(64 * 7 * 7, tc.trxl_dim)),
                nn.ReLU(),
            )
        elif len(obs_shape) == 1:
            self.encoder = self.layer_init(nn.Linear(obs_shape[0], tc.trxl_dim))
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")

        self.transformer = Transformer(tc, max_episode_steps)

        self.hidden_post_trxl = nn.Sequential(
            self.layer_init(nn.Linear(tc.trxl_dim, tc.trxl_dim)),
            nn.ReLU(),
        )

        self.actor_branches = nn.ModuleList(
            [
                self.layer_init(nn.Linear(tc.trxl_dim, out_features=n), np.sqrt(0.01))
                for n in action_space_shape
            ]
        )
        self.critic = self.layer_init(nn.Linear(tc.trxl_dim, 1), 1)

        self.internal_head: InternalActionHead | None = None
        if tc.use_spatial_memory:
            self.internal_head = make_internal_head(
                tc.spatial_head_type, trxl_dim=tc.trxl_dim, grid_w=grid_w, grid_h=grid_h,
                grid_cell_encoding=tc.grid_cell_encoding,
                hidden_size=tc.internal_head_hidden_size,
            )

        self.transposed_cnn = None
        if tc.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                self.layer_init(nn.Linear(tc.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                self.layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                self.layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                self.layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )

        self.submodules = {
            "encoder": self.encoder,
            "transformer": self.transformer,
            "hidden_post_trxl": self.hidden_post_trxl,
            "actor_branches": self.actor_branches,
            "internal_head": self.internal_head,
            "critic": self.critic,
            "transposed_cnn": self.transposed_cnn,
        }

    def load_submodules(self, saved_states: dict[str, dict], names: list[str]) -> None:
        """
        Load state dicts for the specified submodules from the given checkpoint states.
        Args:
            saved_states: A dict mapping submodule names to their state dicts, as returned by
                    submodule_states().
            names: A list of submodule names to load from the checkpoint.
                    Must be keys in saved_states.
        """
        for name in names:
            if name not in self.submodules.keys():
                raise ValueError(f"'{name}' is not a checkpointable submodule")
            if name not in saved_states:
                raise KeyError(f"Submodule '{name}' not found in checkpoint")
            self.submodules[name].load_state_dict(saved_states[name], strict=False)

    def submodule_states(self) -> dict[str, dict]:
        """Return a dict mapping submodule names to their state dicts, for checkpointing."""
        return {
            name: module.state_dict() if module is not None else {"empty": "empty"}
            for name, module in self.submodules.items()
        }

    def get_value(
        self,
        x:torch.Tensor,
        memory_window: MemoryWindow,
        envs_t: torch.Tensor,
        perceived_pos: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, _, _ = self.transformer(
            x=x,
            memory_window=memory_window,
            envs_t=envs_t,
            perceived_pos=perceived_pos,
        )
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(
        self,
        x: torch.Tensor,
        memory_window: MemoryWindow,
        envs_t: torch.Tensor,
        perceived_pos: torch.Tensor,
        action: Action | None = None,
        global_step: int | None = None,
        forward_diagnostics: ForwardDiagnostics | None = None,
    ) -> tuple[Action, torch.Tensor, torch.Tensor, list]:
        """
        Args:
            x: (N, *obs_shape) tensor of the current observations for the minibatch items.
            memory_window: MemoryWindow object containing the retrieval window for the minibatch items.
            envs_t: (N,) tensor of the current env steps for positional encoding
            action: Optional Action object containing the actions to evaluate. If None, new actions will be
                    sampled from the policy.
            global_step: Current training step count; used to decide whether to detach
                the internal head from the backbone during warmup. Only needed to pass during
                training phase, not rollout collection or evaluation
        """

        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)

        x, new_memory_frame, attention_weights = self.transformer(
            x=x,
            memory_window=memory_window,
            envs_t=envs_t,
            perceived_pos=perceived_pos,
            forward_diagnostics=forward_diagnostics,
        )

        x = self.hidden_post_trxl(x)

        ####
        # External action distribution and sampling^
        ####

        external_probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]

        # sample external and internal actions from the policy
        if action is None:
            external_action = torch.stack([dist.sample() for dist in external_probs], dim=1)
        # if action is provided, use the given actions instead and provide the log probs
        else:
            external_action = action.external_action

        external_log_probs = torch.stack(
            [dist.log_prob(external_action[:, i]) for i, dist in enumerate(external_probs)], dim=1
        )
        external_entropy = (
            torch.stack([dist.entropy() for dist in external_probs], dim=1).sum(1).reshape(-1)
        )

        ####
        # Internal action distribution and sampling
        ####

        if self.use_spatial_memory:
            if self.internal_head is None:
                raise ValueError("Internal head is not initialized but use_spatial_memory is True.")
            detach = (
                self.internal_head_warmup_steps >= 0
                and global_step is not None
                and global_step < self.internal_head_warmup_steps
            )
            x_internal = x.detach() if detach else x
            retrieval_pos_encoded = (
                self.transformer.encode_position(memory_window.retrieval_pos)
                if self.transformer.grid_cell_encoding
                else None
            )
            internal_action, internal_log_probs, internal_entropy = self.internal_head(
                x_internal, action.internal_action if action is not None else None,
                retrieval_pos_encoded=retrieval_pos_encoded,
            )
            retr_positions = self.internal_head.action_to_position(internal_action, prev_position=memory_window.retrieval_pos)
        else:
            internal_action = None
            internal_log_probs = None
            internal_entropy = None
            retr_positions = None
        return (
            Action(
                external_action=external_action,
                external_log_probs=external_log_probs,
                external_entropy=external_entropy,
                internal_action=internal_action,
                internal_log_probs=internal_log_probs,
                internal_entropy=internal_entropy,
                retr_positions=retr_positions,
            ),
            self.critic(x).flatten(),
            new_memory_frame,
            attention_weights,
        )

    def reconstruct_observation(self) -> torch.Tensor:
        if self.transposed_cnn is None:
            raise ValueError("reconstruct_observation() called but reconstruction_coef is 0.")
        raise NotImplementedError(
            "TODO: reconstruct observation follows a bad pattern"
            "(_last_policy_features is blind cached)"
        )

        if self._last_policy_features is None:
            raise RuntimeError("No cached policy features. Call get_action_and_value() first.")
        x = self.transposed_cnn(self._last_policy_features)
        return x.permute((0, 2, 3, 1))

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        # torch.nn.init.constant_(layer.bias, bias_const)
        return layer
