import random
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary

import wandb
from src_new.config import Config
from src_new.env.facade import TrainEnv
from src_new.env.factory import make_env
from src_new.memory.handler import MemoryHandler
from src_new.memory.transformations import add_hindsight_at_0_, apply_memory_masks_
from src_new.memory.types import MemoryWindow
from src_new.model.agent import Agent
from src_new.model.agent_module import Action
from src_new.trainer.checkpoint import CheckpointManager
from src_new.trainer.forward_diagnostics import ForwardDiagnostics
from src_new.trainer.metrics import Metrics
from src_new.trainer.trajectory import Trajectory

"""
Here the main todos in the project:
TODO
- add explizit device handling
"""


@dataclass
class RunState:
    """Mutable per-run state passed between training phases."""

    episode_step: torch.Tensor  # (num_envs,) long — current env step per env
    next_obs: torch.Tensor  # latest observation tensor
    done: torch.Tensor  # (num_envs,) bool
    spawn_pos: torch.Tensor  # (num_envs, 2) long — spawn position per env
    global_step: int  # number of rollout steps so far
    global_count: int  # global_step * num_envs
    perceived_pos: torch.Tensor  # (num_envs, 2) long — agent position relative to spawn


class Trainer:
    """Encapsulates the full PPO + TrXL training run for a single experiment."""

    def __init__(self, config: Config):
        self.config = config
        log_conf = config.log
        train_conf = config.train
        self.run_name = (
            f"{train_conf.env_id}__{train_conf.exp_name}__{train_conf.seed}__"
            f"{time.strftime('%m-%d_%H:%M', time.localtime())}"
        )

        # ── Logging ──────────────────────────────────────────────────────────
        if log_conf.track:
            # wandb.init() already called in main.py; just update the run name
            if wandb.run is not None:
                wandb.run.name = self.run_name
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [
                        f"|{key}|{value}|"
                        for key, value in {**vars(log_conf), **vars(train_conf)}.items()
                    ]
                )
            ),
        )

        # ── Seeding ───────────────────────────────────────────────────────────
        random.seed(train_conf.seed)
        np.random.seed(train_conf.seed)
        torch.manual_seed(train_conf.seed)
        # torch.backends.cudnn.deterministic = train_conf.torch_deterministic
        torch.use_deterministic_algorithms(True)

        # ── Device ───────────────────────────────────────────────────────────
        if train_conf.cuda:
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available but train.cuda is True")
            self.device = torch.device("cuda")
            torch.set_default_device(self.device)
        else:
            self.device = torch.device("cpu")

        # ── Environments ─────────────────────────────────────────────────────
        self.envs: gym.vector.SyncVectorEnv = gym.vector.SyncVectorEnv(
            [
                make_env(
                    train_conf.env_id,
                    i,
                    log_conf.capture_video,
                    self.run_name,
                    num_vids_per_trigger=log_conf.num_vids_per_trigger,
                    render_mode="rgb_array",
                    trxl_layers=train_conf.trxl_num_layers,
                    trxl_dim=train_conf.trxl_dim,
                )
                for i in range(train_conf.num_envs)
            ]
        )

        self.train_envs: list[TrainEnv] = cast(list[TrainEnv], self.envs.envs)
        self.env0: TrainEnv = self.train_envs[0]

        self.observation_space = self.envs.single_observation_space
        if self.observation_space.shape is None:
            raise ValueError("Observation space shape must not be None.")
        self.observation_shape: tuple[int] = cast(tuple[int], self.observation_space.shape)
        self.action_space_shape: tuple[int] = (
            (self.envs.single_action_space.n,)
            if isinstance(self.envs.single_action_space, gym.spaces.Discrete)
            else tuple(self.envs.single_action_space.nvec)  # type: ignore
        )

        # for env-general way of getting max steps, scroll down to the cemetry of this file
        self.max_episode_steps: int = self.env0.max_episode_steps

        # Clip memory length to episode length
        train_conf.trxl_memory_length = min(train_conf.trxl_memory_length, self.max_episode_steps)

        # ── Agent & optimiser ─────────────────────────────────────────────────

        self.agent = Agent(
            train_conf=train_conf,
            observation_space=self.observation_space,
            action_space_shape=self.action_space_shape,
            max_episode_steps=self.max_episode_steps,
            grid_w=self.env0.grid_width,
            grid_h=self.env0.grid_height,
            device=self.device,
        )

        if train_conf.checkpoint_path is not None:
            submodules_states = CheckpointManager.load(
                train_conf.checkpoint_path, device=self.device
            )
            self.agent.module.load_submodules(submodules_states, train_conf.checkpoint_submodules)

        model_summary = summary(self.agent.module, verbose=0)
        print(f"{'#' * 20} Agent architecture {'#' * 20}")
        print(model_summary)
        print(f"{'#' * 60}")
        if log_conf.track and wandb.run is not None:
            wandb.run.summary["model_summary"] = str(model_summary)

        self.optimizer = optim.AdamW(self.agent.module.parameters(), lr=train_conf.init_lr)
        self.bce_loss = nn.BCELoss()

        # ── Memory ──────────────────────────────────────────────────────────────

        self.memory_handler = MemoryHandler(
            num_envs=train_conf.num_envs,
            num_rollout_steps=train_conf.num_rollout_steps,
            max_episode_steps=self.max_episode_steps,
            trxl_memory_length=train_conf.trxl_memory_length,
            trxl_num_layers=train_conf.trxl_num_layers,
            trxl_dim=train_conf.trxl_dim,
            grid_w=self.env0.grid_width,
            grid_h=self.env0.grid_height,
            device=self.device,
        )

        # ── Rollout storage ───────────────────────────────────────────────────
        self.trajectory = Trajectory(
            num_envs=train_conf.num_envs,
            rollout_steps=train_conf.num_rollout_steps,
            observation_space_shape=self.observation_shape,
            device=self.device,
        )

        # ── Checkpointing ─────────────────────────────────────────────────────
        if log_conf.save_model:
            self.checkpoint = CheckpointManager(run_name=self.run_name, train_config=train_conf)

    def run(self):
        train_conf = self.config.train
        device = self.device

        # ── Derived constants ─────────────────────────────────────────────────
        batch_size: int = train_conf.num_envs * train_conf.num_rollout_steps
        minibatch_size: int = batch_size // train_conf.num_minibatches
        num_iterations: int = train_conf.total_timesteps // batch_size

        # ── Initial env state ─────────────────────────────────────────────────
        next_obs_np, _ = self.envs.reset(seed=train_conf.seed)
        spawn_pos = torch.zeros((train_conf.num_envs, 2), dtype=torch.long, device=device)
        for i, env in enumerate(self.train_envs):
            spawn_pos[i] = torch.tensor(env.agent_pos, dtype=torch.long, device=device)

        rs = RunState(
            episode_step=torch.zeros((train_conf.num_envs,), dtype=torch.long, device=device),
            next_obs=torch.as_tensor(next_obs_np, device=device),
            done=torch.zeros(train_conf.num_envs, dtype=torch.bool, device=device),
            spawn_pos=spawn_pos,
            global_step=0,
            global_count=0,
            perceived_pos=torch.zeros((train_conf.num_envs, 2), dtype=torch.long, device=device),
        )

        start_time = time.time()
        episode_infos: deque[dict] = deque(maxlen=100)
        metrics = Metrics(alpha=0.1)

        for iteration in range(1, num_iterations + 1):
            # ── Anneal lr / entropy ───────────────────────────────────────────
            do_anneal = train_conf.anneal_steps > 0 and rs.global_count < train_conf.anneal_steps
            frac = 1 - rs.global_count / train_conf.anneal_steps if do_anneal else 0
            lr = (train_conf.init_lr - train_conf.final_lr) * frac + train_conf.final_lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            ent_coef = (
                train_conf.init_ent_coef - train_conf.final_ent_coef
            ) * frac + train_conf.final_ent_coef

            # ── Phases ────────────────────────────────────────────────────────
            sampled_episode_infos = self._rollout_phase(rs)

            bootstrap_value = self._compute_bootstrap(rs)
            self.trajectory.calculate_advantages_and_returns(
                bootstrap_value=bootstrap_value,
                gamma=train_conf.gamma,
                gae_lambda=train_conf.gae_lambda,
            )

            self._ppo_update(
                metrics=metrics,
                ent_coef=ent_coef,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                global_step=rs.global_count,
            )

            # ── Logging ───────────────────────────────────────────────────────
            episode_infos.extend(sampled_episode_infos)
            self._log_iteration(
                iteration=iteration,
                metrics=metrics,
                episode_infos=episode_infos,
                lr=lr,
                ent_coef=ent_coef,
                global_count=rs.global_count,
                start_time=start_time,
            )

            if self.config.log.save_model:
                self.checkpoint.checkpoint(
                    module=self.agent.module,
                    metrics=metrics,
                    iteration=rs.global_step,
                )
            self._log_new_videos_to_wandb()
        self._log_new_videos_to_wandb()
        self.writer.close()
        self.envs.close()
        print(
            "Training finished. Total time: {:.2f} minutes.".format((time.time() - start_time) / 60)
        )

    # ======================================================================
    # Phase 1: Rollout
    # ======================================================================

    def _rollout_phase(self, rs: RunState) -> list[dict]:
        """Collect one rollout of ``num_rollout_steps`` across all envs.

        Mutates *rs* in place (episode_step, next_obs, done, spawn_pos,
        global_step, global_count).  Returns sampled episode info dicts for
        logging.
        """
        train_conf = self.config.train
        device = self.device
        agent = self.agent
        trajy = self.trajectory
        envs = self.envs

        sampled_episode_infos: list[dict] = []
        trajy.new_rollout()

        for rollout_step in range(train_conf.num_rollout_steps):
            current_obs = rs.next_obs

            with torch.no_grad():
                # (N, grid_w, grid_h)
                fov = np.stack([env.get_visible_cells() for env in self.train_envs])
                fov = torch.as_tensor(fov, device=device)

                # perceived position relative to spawn
                current_agent_positions = torch.stack(
                    [
                        torch.tensor(env.agent_pos, dtype=torch.long, device=device)
                        for env in self.train_envs
                    ]
                )
                rs.perceived_pos = current_agent_positions - rs.spawn_pos  # (N, 2), can be negative

                mem_retrieval_step = rs.global_step - 1
                memory_window = self.memory_handler.get_memory_window_rollout(
                    global_step=mem_retrieval_step,
                    exclude_out_of_episode=self.config.train.exclude_out_of_episode_memories
                )

                if train_conf.apply_memory_masks:
                    apply_memory_masks_(memory_window)

                forward_diagnostics = ForwardDiagnostics()
                action, value, memory_write_record = agent.sample_action(
                    obs=current_obs,
                    episode_step=rs.episode_step.clone(),
                    fov=fov,
                    memory_window=memory_window,
                    perceived_pos=rs.perceived_pos,
                    forward_diagnostics=forward_diagnostics,  # collects data for visualization
                )
                external_action = action.external_action
                external_logprob = action.external_log_probs

            # ── Capture frame (env 0 only) before stepping ────────────────
            rf_scaled_env0 = memory_write_record.receptive_fields_scaled[0]
            rf_binary_env0 = memory_write_record.receptive_fields_binary[0]

            retrieved_fov = None
            cumulative_fov = None
            if self.config.train.use_spatial_memory:
                env0_t = int(rs.episode_step[0].item())
                retrieved_fov = self.agent.spatial_index.get_retrieved_fov_union(
                    0, memory_window.indices_env[0], memory_window.masks[0]
                )
                cumulative_fov = self.agent.spatial_index.get_cumulative_fov(0, env0_t)

            self.env0.capture_frame(
                env_t=int(rs.episode_step[0].item()),
                rf_scaled=rf_scaled_env0[-1],
                rf_binary=rf_binary_env0[-1],
                perceived_pos=rs.perceived_pos[0],
                memory_retrieval_pos=action.retr_positions[0]
                if action.retr_positions is not None
                else None,
                memory_indices_env=memory_window.indices_env[0],
                memory_masks=memory_window.masks[0],
                retrieved_fov=retrieved_fov,
                cumulative_fov=cumulative_fov,
                forward_diagnostics=forward_diagnostics,  # pass diagnostics to env for video overlay
            )

            # ── Step the environments ─────────────────────────────────────
            next_obs, reward, terminations, truncations, infos = envs.step(
                external_action.cpu().numpy()
            )

            rs.next_obs = torch.as_tensor(next_obs, device=device)
            reward_t = torch.as_tensor(reward, device=device)
            rs.done = torch.as_tensor(
                np.logical_or(terminations, truncations), dtype=torch.bool, device=device
            )

            # ── Store trajectory + memory ─────────────────────────────────
            trajy.store_rollout_step(
                global_step=rs.global_step,
                mem_retrieval_step=mem_retrieval_step,
                envs_t=rs.episode_step.clone(),
                rollout_step=rollout_step,
                reward=reward_t,
                done=rs.done,
                value=value,
                obs=current_obs,
                action=external_action,
                logprob=external_logprob,
                perceived_positions=rs.perceived_pos,
                internal_action=action.internal_action,
                internal_logprob=action.internal_log_probs,
                retrieval_hit=action.retrieval_hit,
            )

            with torch.no_grad():
                self.memory_handler.add_memory_write_record(
                    global_step=rs.global_step,
                    write_record=memory_write_record,
                    done_envs=rs.done,
                    envs_t=rs.episode_step.clone(),
                )

            # ── Episode info logging ──────────────────────────────────────
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        sampled_episode_infos.append(info["episode"])

            # ── Time-step bookkeeping ─────────────────────────────────────
            rs.episode_step += 1
            for env_id, done_i in enumerate(rs.done):
                if done_i:
                    rs.episode_step[env_id] = 0
                    rs.spawn_pos[env_id] = torch.tensor(
                        self.train_envs[env_id].agent_pos, dtype=torch.long, device=device
                    )
            self.agent.reset_envs(rs.done)

            rs.global_count += train_conf.num_envs
            rs.global_step += 1

        return sampled_episode_infos

    # ======================================================================
    # Phase 2: Bootstrap value computation
    # ======================================================================

    @torch.no_grad()
    def _compute_bootstrap(self, rs: RunState) -> torch.Tensor:
        """Compute the bootstrap value for GAE at the end of the rollout."""
        train_conf = self.config.train
        device = self.device
        agent = self.agent

        if not train_conf.use_old_bootstrap:
            bootstrap_memory_window = self.memory_handler.get_memory_window_rollout(
                global_step=rs.global_step - 1,
                exclude_out_of_episode=self.config.train.exclude_out_of_episode_memories
            )
            return agent.bootstrap_value(
                rs.next_obs, bootstrap_memory_window, rs.episode_step, rs.perceived_pos
            )

        ####################################################
        # START: Old bootstrap – reproduces the original codebase's bootstrap
        # calculation.  The suspected bug is that the memory window is indexed
        # at episode_step-1 ("stale"), so the frame written at the *current*
        # step is never included.  In contrast, the new path uses
        # get_memory_window_rollout(global_step-1) which already contains the
        # latest write.
        #
        # High-level flow:
        #   1. Fetch the raw memory buffer snapshot from the *previous*
        #      global step (the last step that was written during rollout).
        #   2. Compute a "stale" episode step:
        #        - For envs that just finished (done=True), episode_step was
        #          already reset to 0, so we use that directly.
        #        - For ongoing envs, we use episode_step - 1 (i.e. one step
        #          behind the agent's actual position).
        #   3. Use the stale step to look up which memory-slot indices the
        #      sliding window would select (memory_indices_template) and
        #      which slots are valid (memory_mask_template, based on the
        #      *current* episode step).
        #   4. Gather the corresponding frames from the raw buffer and zero
        #      out any masked (not-yet-filled) positions.
        #   5. Build a MemoryWindow and pass it to the critic for V(s').
        #      gourmet_mode is turned off to not trigger assertions.
        ####################################################
        warnings.warn("Running original bootstrap calculation")

        # 1. Raw buffer snapshot from the last rollout step
        raw_window = self.memory_handler.memory_rollout_buffer.get_step(
            rs.global_step - 1,
            exclude_out_of_episode=self.config.train.exclude_out_of_episode_memories,
        )

        # 2. Stale step: for done envs episode_step is already 0; for
        #    ongoing envs subtract 1 so the indices lag behind by one step.
        stale_step = torch.where(rs.done, rs.episode_step, rs.episode_step - 1)

        # 3a. Which M memory slots to attend to (sliding-window pattern)
        # TODO DEBUG line replace 
        # stale_indices = agent.temporal_memory.memory_indices_template[stale_step].to(device)
        episode_step_at_last_rollout = self.trajectory.ri.envs_t[:, -1]  # (num_envs,)
        stale_indices = agent.temporal_memory.memory_indices_template[episode_step_at_last_rollout].to(device)
        # TODO DEBUG above

        # 3b. Mask: which of those M slots actually contain valid data.
        #     Indexed by the *current* episode step (not stale) so the mask
        #     width matches how many steps the env has truly experienced.
        current_mask = agent.temporal_memory.memory_mask_template[
            torch.clip(rs.episode_step, 0, train_conf.trxl_memory_length - 1)
        ].to(device)

        # 4. Gather frames for the selected memory slots and zero-mask them
        B = raw_window.frames.shape[0]
        batch_idx = torch.arange(B, device=device)[:, None]
        bootstrap_frames = raw_window.frames[batch_idx, stale_indices]
        perceived_positions = raw_window.perceived_positions[batch_idx, stale_indices]

        # TODO:debug outcomment
        # if True:  # always mask; original had a flag but it was always on
        #    bootstrap_frames = bootstrap_frames * current_mask[:, :, None, None]

        # 5. Assemble the MemoryWindow and compute V(s')
        bootstrap_memory_window = MemoryWindow(
            frames=bootstrap_frames,
            indices_env=stale_indices,
            masks=current_mask,
            receptive_fields_scaled=torch.empty(0, device=device),
            receptive_fields_binary=torch.empty(0, device=device),
            perceived_positions=perceived_positions,
            env_ids=raw_window.env_ids,
            envs_t=rs.episode_step,
            global_steps=raw_window.global_steps,
            retrieval_pos=raw_window.retrieval_pos,
        )

        agent.gourmet_mode = False  # skip auxiliary heads for bootstrap
        bootstrap_value = agent.bootstrap_value(
            rs.next_obs,
            bootstrap_memory_window,
            rs.episode_step,
            perceived_pos=rs.perceived_pos,
        )
        agent.gourmet_mode = True
        return bootstrap_value
        # END: Old bootstrap
        ####################################################

    # ======================================================================
    # Phase 3: PPO update
    # ======================================================================

    def _ppo_update(
        self,
        metrics: Metrics,
        ent_coef: float,
        batch_size: int,
        minibatch_size: int,
        global_step: int = 0,
    ) -> None:
        """Run PPO update epochs over the collected trajectory.  Mutates *metrics*."""
        train_conf = self.config.train
        device = self.device
        agent = self.agent
        trajy = self.trajectory

        # Accumulators for iteration-level averages
        sum_pg = 0.0
        sum_v = 0.0
        sum_entropy = 0.0
        sum_int_pg = 0.0
        sum_int_entropy = 0.0
        sum_retr_reinforce = 0.0
        sum_loss = 0.0
        sum_r = 0.0
        sum_old_kl = 0.0
        sum_kl = 0.0
        clipfracs: list[float] = []
        n_updates = 0
        approx_kl = torch.tensor(0.0)

        for epoch in range(train_conf.update_epochs):
            minibatches = torch.cartesian_prod(
                torch.arange(train_conf.num_rollout_steps), torch.arange(train_conf.num_envs)
            )
            b_inds = torch.randperm(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_steps, mb_envs = minibatches[mb_inds].T
                mb = trajy.get_minibatch(mb_envs=mb_envs, mb_steps=mb_steps)

                # TODO also train internal actions
                mb_memory_window = self.memory_handler.get_memory_window_minibatch(
                    env_ids=mb_envs,
                    global_steps=mb.mem_retrieval_steps,
                    exclude_out_of_episode=self.config.train.exclude_out_of_episode_memories,
                )

                if (mb.envs_t == 0).any():
                    pass

                # apply transformations
                if train_conf.hindsight_at_0:
                    add_hindsight_at_0_(mb_memory_window, mb)
                if train_conf.apply_memory_masks:
                    apply_memory_masks_(mb_memory_window)

                # TODO: why not obey the agent facade
                (
                    result,
                    newvalue,
                    _,
                    _,
                ) = agent.module.get_action_and_value(
                    x=mb.obs,
                    memory_window=mb_memory_window,
                    envs_t=mb.envs_t,
                    perceived_pos=mb.perceived_positions,
                    action=Action(
                        external_action=mb.actions,
                        external_log_probs=torch.empty(0, device=device),
                        external_entropy=torch.empty(0, device=device),
                        internal_action=mb.internal_actions,
                        internal_log_probs=torch.empty(0, device=device),
                        internal_entropy=torch.empty(0, device=device),
                    ),
                    global_step=global_step,
                )

                ####
                # External Action Update
                ####

                newlogprob = result.external_log_probs
                entropy = result.external_entropy

                # Policy loss
                mb_advantages = mb.advantages.clone()
                if train_conf.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                mb_advantages = mb_advantages.unsqueeze(1).repeat(
                    1, len(self.action_space_shape)
                )  # Repeat is necessary for multi-discrete action spaces
                logratio = newlogprob - mb.log_probs
                ratio = torch.exp(logratio)
                pgloss1 = -mb_advantages * ratio
                pgloss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - train_conf.clip_coef, 1.0 + train_conf.clip_coef
                )
                pg_loss = torch.max(pgloss1, pgloss2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                external_loss = pg_loss - ent_coef * entropy_loss

                ####
                # Internal Action Update (if enabled)
                ####

                if self.config.train.use_spatial_memory:
                    if (
                        result.internal_log_probs is None
                        or mb.internal_log_probs is None
                        or result.internal_entropy is None
                    ):
                        raise ValueError(
                            "Internal log probs and entropy must be available for internal action"
                            " update."
                        )
                    int_logratio = result.internal_log_probs - mb.internal_log_probs
                    int_ratio = torch.exp(int_logratio)
                    # Use same advantages as external (shared reward signal)
                    int_adv = mb_advantages[:, 0]  # (N,) — collapse branch dim
                    int_pgloss1 = -int_adv * int_ratio
                    int_pgloss2 = -int_adv * torch.clamp(
                        int_ratio, 1.0 - train_conf.clip_coef, 1.0 + train_conf.clip_coef
                    )
                    internal_pg_loss = torch.max(int_pgloss1, int_pgloss2).mean()
                    internal_entropy_loss = result.internal_entropy.mean()
                else:
                    internal_pg_loss = torch.tensor(0.0)
                    internal_entropy_loss = torch.tensor(0.0)

                internal_loss = (
                    internal_pg_loss - train_conf.internal_ent_coef * internal_entropy_loss
                )

                # Retrieval-hit REINFORCE loss (if enabled)

                if (
                    self.config.train.use_spatial_memory
                    and train_conf.retrieval_reward_coef > 0
                    and mb.retrieval_hit is not None
                    and result.internal_log_probs is not None
                ):
                    hit = mb.retrieval_hit  # (N,)
                    # Center reward for variance reduction
                    retrieval_reward = hit - hit.mean()
                    retrieval_reinforce_loss = -(
                        retrieval_reward.detach() * result.internal_log_probs
                    ).mean()
                else:
                    retrieval_reinforce_loss = torch.tensor(0.0)

                ####
                # Critic update
                ####

                # Value loss
                v_loss_unclipped = (newvalue - mb.returns) ** 2
                if train_conf.clip_vloss:
                    v_loss_clipped = mb.values + (newvalue - mb.values).clamp(
                        min=-train_conf.clip_coef, max=train_conf.clip_coef
                    )
                    v_loss = torch.max(v_loss_unclipped, (v_loss_clipped - mb.returns) ** 2).mean()
                else:
                    v_loss = v_loss_unclipped.mean()

                ####
                # Other updates and losses (e.g. reconstruction)
                ####

                # Reconstruction loss
                r_loss = torch.tensor(0.0)
                if train_conf.reconstruction_coef > 0.0:
                    r_loss = self.bce_loss(agent.reconstruct_observation(), mb.obs / 255.0)

                # combine losses
                loss = (
                    external_loss
                    + internal_loss
                    + train_conf.retrieval_reward_coef * retrieval_reinforce_loss
                    + train_conf.vf_coef * v_loss
                    + train_conf.reconstruction_coef * r_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.module.parameters(), max_norm=train_conf.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > train_conf.clip_coef).float().mean().item()
                    )

                # Accumulate for iteration-level averages
                sum_pg += pg_loss.item()
                sum_v += v_loss.item()
                sum_entropy += entropy_loss.item()
                sum_int_pg += internal_pg_loss.item()
                sum_int_entropy += internal_entropy_loss.item()
                sum_retr_reinforce += retrieval_reinforce_loss.item()
                sum_loss += loss.item()
                sum_r += r_loss.item()
                sum_old_kl += old_approx_kl.item()
                sum_kl += approx_kl.item()
                n_updates += 1

            if train_conf.target_kl is not None and approx_kl > train_conf.target_kl:
                break

        # Write iteration-level averages to metrics (triggers EMA update)
        metrics.pg_loss = sum_pg / n_updates
        metrics.v_loss = sum_v / n_updates
        metrics.entropy_loss = sum_entropy / n_updates
        metrics.internal_pg_loss = sum_int_pg / n_updates
        metrics.internal_entropy_loss = sum_int_entropy / n_updates
        metrics.retrieval_reinforce_loss = sum_retr_reinforce / n_updates
        metrics.loss = sum_loss / n_updates
        metrics.r_loss = sum_r / n_updates
        metrics.old_approx_kl = sum_old_kl / n_updates
        metrics.approx_kl = sum_kl / n_updates
        metrics.clipfrac = float(np.mean(clipfracs)) if clipfracs else float("nan")

        # Explained variance
        values_flat = trajy.ri.values.view(-1)
        returns_flat = trajy.pri.returns.view(-1)
        y_pred, y_true = values_flat.cpu().numpy(), returns_flat.cpu().numpy()
        var_y = np.var(y_true)
        metrics.explained_var = float(np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)

    # ======================================================================
    # Logging
    # ======================================================================

    def _log_iteration(
        self,
        iteration: int,
        metrics: Metrics,
        episode_infos: deque[dict],
        lr: float,
        ent_coef: float,
        global_count: int,
        start_time: float,
    ) -> None:
        writer = self.writer
        trajy = self.trajectory

        episode_result: dict[str, float] = {}
        if len(episode_infos) > 0:
            # Aggregate over the union of keys and ignore non-finite values.
            # This keeps conditional metrics (e.g. acc_given_*) loggable even
            # when many episodes intentionally carry NaN for those keys.
            all_episode_keys = set().union(*(info.keys() for info in episode_infos))
            for key in all_episode_keys:
                values = np.asarray(
                    [info[key] for info in episode_infos if key in info], dtype=float
                )
                finite_values = values[np.isfinite(values)]
                if finite_values.size == 0:
                    continue
                episode_result[key + "_mean"] = float(np.mean(finite_values))

        # Update episode-level metrics on Metrics (triggers EMA)
        if "r_mean" in episode_result:
            metrics.episode_return = episode_result["r_mean"]
        if "l_mean" in episode_result:
            metrics.episode_length = episode_result["l_mean"]
        if "steps_above_min_mean" in episode_result:
            metrics.steps_above_min = episode_result["steps_above_min_mean"]

        metrics.pass_additional(episode_result)
        metrics.value_mean = float(torch.mean(trajy.ri.values))
        metrics.advantage_mean = float(torch.mean(trajy.pri.advantages))

        sps = int(global_count / (time.time() - start_time))

        print(
            "{:9} SPS={:4} return={:.2f} length={:.1f} pi_loss={:.3f} v_loss={:.3f} "
            "entropy={:.3f} r_loss={:.3f} value={:.3f} adv={:.3f}".format(
                iteration,
                sps,
                metrics.episode_return,
                metrics.episode_length,
                metrics.pg_loss,
                metrics.v_loss,
                metrics.entropy_loss,
                metrics.r_loss,
                metrics.value_mean,
                metrics.advantage_mean,
            )
        )

        writer.add_scalar("charts/learning_rate", lr, global_count)
        writer.add_scalar("charts/entropy_coefficient", ent_coef, global_count)
        writer.add_scalar("charts/SPS", sps, global_count)

        # ── Raw + smoothed metrics ────────────────────────────────────────
        _raw_ma: list[tuple[str, str, float, float]] = [
            ("losses", "policy_loss", metrics.pg_loss, metrics.pg_loss_ma),
            ("losses", "value_loss", metrics.v_loss, metrics.v_loss_ma),
            ("losses", "loss", metrics.loss, metrics.loss_ma),
            ("losses", "entropy", metrics.entropy_loss, metrics.entropy_loss_ma),
            (
                "losses",
                "internal_policy_loss",
                metrics.internal_pg_loss,
                metrics.internal_pg_loss_ma,
            ),
            (
                "losses",
                "internal_entropy",
                metrics.internal_entropy_loss,
                metrics.internal_entropy_loss_ma,
            ),
            (
                "losses",
                "retrieval_reinforce_loss",
                metrics.retrieval_reinforce_loss,
                metrics.retrieval_reinforce_loss_ma,
            ),
            ("losses", "reconstruction_loss", metrics.r_loss, metrics.r_loss_ma),
            ("losses", "old_approx_kl", metrics.old_approx_kl, metrics.old_approx_kl_ma),
            ("losses", "approx_kl", metrics.approx_kl, metrics.approx_kl_ma),
            ("losses", "clipfrac", metrics.clipfrac, metrics.clipfrac_ma),
            ("losses", "explained_variance", metrics.explained_var, metrics.explained_var_ma),
            ("episode", "return", metrics.episode_return, metrics.episode_return_ma),
            ("episode", "length", metrics.episode_length, metrics.episode_length_ma),
            ("episode", "steps_above_min", metrics.steps_above_min, metrics.steps_above_min_ma),
            ("episode", "value_mean", metrics.value_mean, metrics.value_mean_ma),
            ("episode", "advantage_mean", metrics.advantage_mean, metrics.advantage_mean_ma),
        ]

        for name, values in metrics.additionals.items():
            _raw_ma.append(("additionals", name, values["raw"], values["ma"]))

        for group, name, raw, ma in _raw_ma:
            writer.add_scalar(f"{group}/{name}", raw, global_count)
            writer.add_scalar(f"{group}_ma/{name}", ma, global_count)

    def _log_new_videos_to_wandb(self):
        if not self.config.log.track or not self.config.log.capture_video:
            return

        for vf in self.env0.drain_completed_videos():
            wandb.log({"videos": wandb.Video(vf, format="mp4")})
