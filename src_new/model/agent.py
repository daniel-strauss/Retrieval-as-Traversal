import torch

from src_new.config import TrainConfig
from src_new.memory.types import MemoryWindow, MemoryWriteRecord
from src_new.model import receptive_field_utils as rf_utils
from src_new.model.agent_module import Action, AgentModule
from src_new.model.retrieval.memory_slider import MemorySlider
from src_new.model.retrieval.spatial_index import SpatialIndex
from src_new.model.retrieval.temporal_memory import TemporalMemory
from src_new.trainer.checkpoint import CheckpointData
from src_new.trainer.forward_diagnostics import ForwardDiagnostics
from src_new.validation import validate_agent_memory_window as _validate_memory_window
from src_new.validation import validate_attention_weights as _validate_attention_weights


class Agent:
    # whether the class does sanity checks on all that it eats
    gourmet_mode = True

    def __init__(
        self,
        train_conf: TrainConfig,
        observation_space,
        action_space_shape,
        max_episode_steps: int,
        device: torch.device,
        grid_w: int,
        grid_h: int,
    ):
        self.device = device
        self.args = train_conf
        self.max_episode_steps: int = max_episode_steps

        self.module = AgentModule(
            obs_shape=observation_space.shape,
            action_space_shape=action_space_shape,
            max_episode_steps=max_episode_steps,
            train_conf=train_conf,
            grid_w=grid_w,
            grid_h=grid_h,
        ).to(device)

        # just used for checking
        self.temporal_memory: TemporalMemory = TemporalMemory(
            num_envs=train_conf.num_envs,
            num_rollout_steps=train_conf.num_rollout_steps,
            max_episode_steps=max_episode_steps,
            trxl_memory_length=train_conf.trxl_memory_length,
            trxl_num_layers=train_conf.trxl_num_layers,
            trxl_dim=train_conf.trxl_dim,
            device=device,
        )

        self.memory_slider: MemorySlider = MemorySlider(
            num_envs=train_conf.num_envs,
            num_rollout_steps=train_conf.num_rollout_steps,
            max_episode_steps=max_episode_steps,
            trxl_memory_length=train_conf.trxl_memory_length
            - 1,  # TODO M-1: hacke the buck from TM backl for validation last slot is added manually
            trxl_num_layers=train_conf.trxl_num_layers,
            trxl_dim=train_conf.trxl_dim,
            device=device,
        )

        self.num_envs = train_conf.num_envs
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.trxl_memory_length = train_conf.trxl_memory_length

        self.use_spatial_memory = train_conf.use_spatial_memory
        if self.use_spatial_memory:
            self.spatial_index = SpatialIndex(
                grid_h=grid_h,
                grid_w=grid_w,
                num_envs=train_conf.num_envs,
                device=device,
                max_episode_steps=max_episode_steps,
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_data: CheckpointData,
        load_submodules: list[str],
        **init_kwargs,
    ) -> "Agent":
        agent = cls(**init_kwargs)
        agent.module.load_submodules(checkpoint_data.submodule_states, load_submodules)
        return agent

    def reconstruct_observation(self):
        return self.module.reconstruct_observation()

    # ========== High-level API  ===============================
    def sample_action(
        self,
        obs: torch.Tensor,
        episode_step: torch.Tensor,
        fov: torch.Tensor,  # TODO contains forbidden information, currently only used for rfs
        memory_window: MemoryWindow,
        perceived_pos: torch.Tensor,
        forward_diagnostics: ForwardDiagnostics | None = None,
    ) -> tuple[Action, torch.Tensor, MemoryWriteRecord]:
        """
        Run a forward pass to sample an action, and prepare the memory write record for the next step.
        Args:
            obs: (N, *obs_shape) tensor of the current observations for the minibatch items.
            episode_step: (N,) per-env episode step. Used for sanity checks in gourmet mode and for
                memory management.
            fov: (N, grid_w, grid_h) boolean tensor of which grid cells are currently visible.
                Used for receptive field computation and optionally for spatial memory.
            memory_window: MemoryWindow object containing all info about the current memory state to
                condition on.
            perceived_pos: (N, 2) tensor of the currently perceived position in the global map,
                used for spatial memory indexing.
        Returns:
            action: Action object containing the sampled action to take in the environment, and optionally internal_action
            value: (N,) tensor of value estimates for the current state
            memory_write_record: MemoryWriteRecord object containing all info to write into memory for the next step.
            trxl_output: (N, D) tensor of the output of the last transformer layer, used for
                analysis and visualization. Not needed for training or memory.
            forward_diagnostics: ForwardDiagnostics object for collecting intermediate activations and other diagnostics.
        """

        if self.use_spatial_memory:
            W = self.grid_w * 2 - 1
            H = self.grid_h * 2 - 1
            offset = torch.tensor([self.grid_w - 1, self.grid_h - 1], device=obs.device)
            perceived_fov = torch.zeros(self.num_envs, W, H, dtype=torch.bool, device=obs.device)
            deltas = torch.tensor(
                [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
                device=obs.device,
            )
            # (N, 2) + (1, 2) + (9, 2) -> (N, 9, 2)
            idx = perceived_pos.unsqueeze(1) + offset + deltas.unsqueeze(0)
            valid = (idx[..., 0] >= 0) & (idx[..., 0] < W) & (idx[..., 1] >= 0) & (idx[..., 1] < H)
            env_exp = torch.arange(self.num_envs, device=obs.device).unsqueeze(1).expand_as(valid)
            perceived_fov[env_exp[valid], idx[..., 0][valid].long(), idx[..., 1][valid].long()] = (
                True
            )
            self.spatial_index.add_fov(perceived_fov, episode_step)

        action, value, pure_memory_frame, attention_weights = self._policy_value_forward(
            obs=obs,
            memory_window=memory_window,
            episode_step=episode_step,
            perceived_pos=perceived_pos,
            evaluate_action=None,
            forward_diagnostics=forward_diagnostics,
        )

        if self.gourmet_mode:
            _validate_attention_weights(
                attention_weights,
                memory_window,
                num_layers=len(self.module.transformer.transformer_layers),
                memory_length=self.trxl_memory_length,
            )

        # trainer will update this tensor
        episode_step = episode_step.clone()

        receptive_fields_scaled, receptive_fields_binary = rf_utils.compute_receptive_fields(
            memory_window=memory_window,
            fov=fov,
            attention_weights=attention_weights,
            envs_t=episode_step,
        )

        ###
        # Retrieve memories for the NEXT step
        ###

        if self.use_spatial_memory:
            if (
                self.module.internal_head is None
                or action.internal_action is None
                or action.retr_positions is None
            ):
                raise ValueError(
                    "Spatial memory is enabled but no internal head is defined in the module."
                )
            spatial_indixes, spatial_masks = self.spatial_index.get_memory_indices_and_mask(
                positions=action.retr_positions
            )
            action.retrieval_hit = spatial_masks.any(dim=1).float()  # (N,)
        else:
            spatial_indixes = torch.zeros((self.num_envs, 0), dtype=torch.long, device=obs.device)
            spatial_masks = torch.zeros((self.num_envs, 0), dtype=torch.bool, device=obs.device)

        # Pre-compute memory window for the next forward pass
        next_step = episode_step + 1

        memory_indices_next_VAL, memory_masks_next_VAL = (
            self.temporal_memory.get_memory_indices_and_mask(next_step)
        )

        memory_indices_next, memory_masks_next = self.memory_slider.get_memory_indices_and_mask(
            retrieval_step=next_step,
            spatial_indexes=spatial_indixes,
            spatial_masks=spatial_masks,
        )

        # TODO M-1: hack the bug from TM back
        # Slider uses M-1 internal slots (all for completed past steps).
        # Append the current step as the M-th column (always masked out) to
        # match TemporalMemory's (N, M) output format.
        memory_indices_next = torch.cat(
            [memory_indices_next, memory_indices_next[:, -1][:, None] + 1], dim=1
        )
        memory_masks_next = torch.cat(
            [
                memory_masks_next,
                torch.zeros(self.num_envs, 1, dtype=torch.bool, device=episode_step.device),
            ],
            dim=1,
        )

        # Only compare envs where TM doesn't clamp (next_step < max_episode_steps).
        # At the boundary TM clamps next_step to max-1 producing a stale window
        # that will be discarded on reset anyway.
        if not self.use_spatial_memory:
            check = next_step < self.max_episode_steps
            if check.any():
                assert torch.equal(memory_masks_next[check], memory_masks_next_VAL[check]), (
                    "MemorySlider and TemporalMemory should produce the same memory masks when no spatial memory is used. "
                    f"Got memory_masks_next={memory_masks_next[check]}, expected {memory_masks_next_VAL[check]}. "
                    f"episode_step: {episode_step[check]}"
                )
                masks_check = memory_masks_next[check].bool()
                if masks_check.any():
                    assert torch.equal(
                        memory_indices_next[check][masks_check],
                        memory_indices_next_VAL[check][masks_check],
                    ), (
                        "MemorySlider and TemporalMemory should produce the same memory indices. "
                        f"Got memory_indices_next={memory_indices_next[check]}, expected {memory_indices_next_VAL[check]}. "
                        f"Episode steps: {episode_step[check]}"
                    )

        memory_write_record = MemoryWriteRecord(
            env_ids=torch.arange(episode_step.shape[0], device=episode_step.device),
            env_steps=episode_step,
            frame=pure_memory_frame,
            indices_next=memory_indices_next,
            masks_next=memory_masks_next,
            receptive_fields_scaled=receptive_fields_scaled,
            receptive_fields_binary=receptive_fields_binary,
            perceived_positions=perceived_pos,
            retrieval_pos=action.retr_positions
                if action.retr_positions is not None
                else torch.zeros_like(perceived_pos),
        )
        return action, value, memory_write_record

    def reset_envs(self, dones: torch.Tensor):
        if self.use_spatial_memory:
            self.spatial_index.reset_envs(dones)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        memory_window: MemoryWindow,
        action: Action,
        episode_step: torch.Tensor,
        perceived_pos: torch.Tensor,
    ) -> tuple[Action, torch.Tensor]:
        result, value, _, _ = self._policy_value_forward(
            obs=obs,
            memory_window=memory_window,
            episode_step=episode_step,
            perceived_pos=perceived_pos,
            evaluate_action=action,
        )
        return result, value

    def bootstrap_value(
        self,
        obs: torch.Tensor,
        memory_window: MemoryWindow,
        episode_step: torch.Tensor,
        perceived_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get value estimate for bootstrapping (no memory update).
        In the next rollout the exact same forward pass will be made, so if we updated memory here,
        the memory rollout would have to enter a faulty state.

        Args:
            obs: Current observations tensor
            memory_window: Memory window for the current batch items.
            episode_step: (N,) per-env current episode step.
        Returns:
            Value estimates tensor
        """
        if self.gourmet_mode:
            _validate_memory_window(
                memory_window,
                episode_step,
                memory_length=self.trxl_memory_length,
                num_layers=len(self.module.transformer.transformer_layers),
                grid_w=self.grid_w,
                grid_h=self.grid_h,
                max_episode_steps=self.max_episode_steps,
            )

        return self.module.get_value(
            obs,
            memory_window,
            envs_t=episode_step,
            perceived_pos=perceived_pos,
        )

    # ========= Core functions ===============================
    def _policy_value_forward(
        self,
        obs: torch.Tensor,
        memory_window: MemoryWindow,
        episode_step: torch.Tensor,
        perceived_pos: torch.Tensor,
        evaluate_action: Action | None,
        forward_diagnostics: ForwardDiagnostics | None = None,
    ) -> tuple[Action, torch.Tensor, torch.Tensor, list]:
        """Run a pure policy/value forward pass.

        Args:
            obs: Current observations tensor (N, *obs_shape)
            memory_window: Memory window for the current batch items.
            episode_step: (N,) per-env episode step. Used for sanity checks in gourmet mode.
            perceived_pos: (N, 2) tensor of the perceived global position, used for transformer positional encoding.
            evaluate_action: If provided, evaluate this action under current policy;
                if None, sample a new action.

        Returns:
            Tuple of action, value, full_memory_frame, attention_weights, trxl_output
        """

        if self.gourmet_mode:
            _validate_memory_window(
                memory_window,
                episode_step,
                memory_length=self.trxl_memory_length,
                num_layers=len(self.module.transformer.transformer_layers),
                grid_w=self.grid_w,
                grid_h=self.grid_h,
                max_episode_steps=self.max_episode_steps,
            )

        ### Forward pass ######################################################
        action, value, pure_memory_frame, attention_weights = self.module.get_action_and_value(
            x=obs,
            memory_window=memory_window,
            envs_t=episode_step,
            perceived_pos=perceived_pos,
            action=evaluate_action,
            forward_diagnostics=forward_diagnostics,
        )

        return action, value, pure_memory_frame, attention_weights
