from dataclasses import dataclass

import torch


@dataclass
class RolloutItem:
    """
    A dataclass to store the data for a single rollout step, for a single environment.

    In each tensor:
        - the first dimension index correponds to an environment index
        - the second dimension corresponds to a time step index

    n: number of environments of rollout item
    m: number of rollout steps of rollout item

    """

    @staticmethod
    def get_empty(n, m, observation_space_shape, device) -> "RolloutItem":
        # needed when initialising the trajectory at the beginning of a rollout,
        # since we need to create empty tensors to store the rollout data.
        return RolloutItem(
            n,
            m,
            rewards=torch.empty((n, m), dtype=torch.float, device=device),
            dones=torch.empty((n, m), dtype=torch.bool, device=device),
            obs=torch.empty((n, m, *observation_space_shape), dtype=torch.float, device=device),
            values=torch.empty((n, m), dtype=torch.float, device=device),
            global_steps=torch.empty((n, m), dtype=torch.long, device=device),
            mem_retrieval_steps=torch.empty((n, m), dtype=torch.long, device=device),
            envs_t=torch.empty((n, m), dtype=torch.long, device=device),
            perceived_positions=torch.empty((n, m, 2), dtype=torch.long, device=device),
        )

    def __init__(
        self,
        n,
        m,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        obs: torch.Tensor,
        values: torch.Tensor,
        global_steps: torch.Tensor,
        mem_retrieval_steps: torch.Tensor,
        envs_t: torch.Tensor,
        perceived_positions: torch.Tensor,
    ):
        self._n = n
        self._m = m
        self.rewards = rewards  # shape (n,m)
        self.dones = dones  # shape (n,m)
        self.values = values  # shape (n,m)
        self.obs = obs  # shape (n,m,*obs_shape)
        # note that global step will contain much redundant information, since all environments will have the same
        # global step at a given rollout step and global_step[i,j] + 1 = global_step[i,j+1]. (we could reduce it to a
        # single number). But that way we use the same system for global step as for all other data items
        self.global_steps = global_steps  # shape (n,m)
        # shape (n,m) - contains the global step at which the memory window was retrieved for the given
        # rollout step. This is needed for the trainer to know which memory window to retrieve for training.
        self.mem_retrieval_steps = mem_retrieval_steps
        self.envs_t = envs_t
        self.perceived_positions = perceived_positions

        # ── Lazy-init: shape inferred on first store_step ──
        self.actions: torch.Tensor | None = None  # (n, m, num_branches)
        self.log_probs: torch.Tensor | None = None  # (n, m, num_branches)
        self.internal_actions: torch.Tensor | None = None  # (n, m, action_dim) or stays None
        self.internal_log_probs: torch.Tensor | None = None  # (n, m) or stays None
        self.retrieval_hit: torch.Tensor | None = None  # (n, m) or stays None

    def store_step(
        self,
        t: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        global_step: int,
        mem_retrieval_step: int,
        envs_t: torch.Tensor,
        perceived_positions: torch.Tensor | None = None,
        internal_action: torch.Tensor | None = None,
        internal_logprob: torch.Tensor | None = None,
        retrieval_hit: torch.Tensor | None = None,
    ):

        self.obs[:, t] = obs
        self.rewards[:, t] = reward
        self.dones[:, t] = done
        self.values[:, t] = value
        self.global_steps[:, t] = global_step
        self.mem_retrieval_steps[:, t] = mem_retrieval_step
        self.envs_t[:, t] = envs_t
        if perceived_positions is not None:
            self.perceived_positions[:, t] = perceived_positions

        # set ittems and initialise them lazily if needed
        self._set_lazy_item("actions", t, action)
        self._set_lazy_item("log_probs", t, logprob)
        if internal_action is not None:
            self._set_lazy_item("internal_actions", t, internal_action)
        if internal_logprob is not None:
            self._set_lazy_item("internal_log_probs", t, internal_logprob)
        if retrieval_hit is not None:
            self._set_lazy_item("retrieval_hit", t, retrieval_hit)

    def _set_lazy_item(self, attr: str, t: int, item: torch.Tensor):
        if getattr(self, attr) is None:
            setattr(
                self,
                attr,
                torch.empty(
                    (self._n, self._m, *item.shape[1:]), dtype=item.dtype, device=item.device
                ),
            )
        getattr(self, attr)[:, t] = item

    @property
    def all(self):
        items = [self.rewards, self.dones, self.values, self.obs, self.global_steps, self.perceived_positions]
        if self.actions is not None and self.log_probs is not None:
            items.extend([self.actions, self.log_probs])
        if self.internal_actions is not None and self.internal_log_probs is not None:
            items.extend([self.internal_actions, self.internal_log_probs])
        return items

    def has_nan(self):
        return any([x.isnan().any() for x in self.all])

    def nan_report(self):
        report = "NanReport: \n"
        for name, tensor in zip(
            ["rewards", "actions", "dones", "values", "log_probs", "obs", "global_steps"], self.all
        ):
            if tensor.isnan().any():
                report += (
                    f"Detected nan in {name} tensor at indices: {torch.where(tensor.isnan())}. \n"
                )
        return report


@dataclass
class PostRolloutItem:
    """
    A dataclass to store assorted data after a rollout is completed.
    Namely, the advantages and returns computed after the rollout is completed
    """

    advantages: torch.Tensor  # shape (n,m)
    returns: torch.Tensor  # shape (n,m)

    def has_nan(self):
        return self.advantages.isnan().any() or self.returns.isnan().any()

    @staticmethod
    def get_empty(n, m, device) -> "PostRolloutItem":
        return PostRolloutItem(
            advantages=torch.empty((n, m), dtype=torch.float, device=device),
            returns=torch.empty((n, m), dtype=torch.float, device=device),
        )


@dataclass
class Minibatch:
    """
    A dataclass to store a minibatch of rollout data.
    """

    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    obs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    mem_retrieval_steps: torch.Tensor
    envs_t: torch.Tensor
    _global_steps: torch.Tensor
    envs: torch.Tensor

    actions: torch.Tensor
    log_probs: torch.Tensor
    internal_actions: torch.Tensor | None
    internal_log_probs: torch.Tensor | None
    perceived_positions: torch.Tensor
    retrieval_hit: torch.Tensor | None

    @property
    def global_steps(self):
        return self._global_steps.clone()


class Trajectory:
    """
    Storage rule: Trajectory holds data that is collected during rollout and
    needed during training (PPO update), but NOT needed after training.
    Contrast with RolloutBuffer, which stores data that persists across
    rollout boundaries.

    Provides the interface for the Trainer to retrieve the data collected during rollout, by rollout step:
     - advantages, actions, rewards, dones, etc. for each environment and rollout step,
         with respect to the rollout timestep.
     - also stores the global steps with respect to rollout timesteps

     Latter is needed such that the trainer can ask agent

     -----------------

     Definition of at the same time (The "||"'s encapsulte what is related to the same time step):
     before existance of time     rollout time step 0                       rollout time step 1
     env_reset ->              || observation -> action -> reward, done, || observation -> action -> reward, done,

     Dont worry, this is just a convention and not grounded on any deeper phylosophical truth.
     This convention is practical, since it bundles what we need in training.


     TODO:
     - raise errors if methods are called in the wrong order (e.g. forgetting to store rollout data, or trying to retrieve data for a rollout step that has not been stored yet, etc.)

    """

    def __init__(
        self,
        num_envs: int,
        rollout_steps: int,
        observation_space_shape: tuple[int],
        device: torch.device,
    ):

        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.observation_space_shape = observation_space_shape
        self.device = device

        self.ri: RolloutItem
        self.pri: PostRolloutItem

        self.next_input_pointer: int  # pointer to the next rollout step to store data for, in the current rollout iteration.

    # -- Before Rollout ------------------------------------------------------------

    def new_rollout(self):
        """
        Resets the rollout buffers for a new rollout, and sets the global step offset for the rollout.
        """
        self.next_input_pointer = 0

        self.ri = RolloutItem.get_empty(
            n=self.num_envs,
            m=self.rollout_steps,
            observation_space_shape=self.observation_space_shape,
            device=self.device,
        )

        self.pri = PostRolloutItem.get_empty(
            n=self.num_envs, m=self.rollout_steps, device=self.device
        )

    # -- During Rollout ------------------------------------------------------------

    def store_rollout_step(
        self,
        rollout_step: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        global_step: int,
        mem_retrieval_step: int,
        envs_t: torch.Tensor,
        perceived_positions: torch.Tensor | None = None,
        internal_action: torch.Tensor | None = None,
        internal_logprob: torch.Tensor | None = None,
        retrieval_hit: torch.Tensor | None = None,
    ):
        """
        Stores the data for a single rollout step.
        """

        if self.next_input_pointer != rollout_step:
            raise ValueError(
                f"Rollout step {rollout_step} does not match the expected next input pointer "
                f"{self.next_input_pointer}."
            )

        self.ri.store_step(
            t=rollout_step,
            obs=obs,
            action=action,
            logprob=logprob,
            reward=reward,
            done=done,
            value=value,
            global_step=global_step,
            mem_retrieval_step=mem_retrieval_step,
            envs_t=envs_t,
            perceived_positions=perceived_positions,
            internal_action=internal_action,
            internal_logprob=internal_logprob,
            retrieval_hit=retrieval_hit,
        )
        self.next_input_pointer += 1

    # -- Post Rollout ------------------------------------------------------------

    def calculate_advantages_and_returns(
        self, bootstrap_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        """
        Calculates the advantages and returns for the rollout, using the rewards, values, and dones stored during rollout, and the last value estimates for bootstrapping.

        Args:
            last_values: value estimates for the last observations of the rollout, shape (num_envs,)
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        """

        # Make sure, rollout has concluded from this classes point of view
        if self.next_input_pointer != self.rollout_steps:
            raise ValueError(
                f"Expected next input pointer to be rollout_steps. Got {self.next_input_pointer}. "
                "Did you add all rollout items?"
            )
        if self.ri.has_nan():
            raise ValueError(f"Detected nan in the current rollout item: {self.ri.nan_report()}")

        # GAE(lambda) backward pass over rollout steps.
        # Convention at step t: (obs_t, action_t, reward_t, done_t, value_t)
        # where done_t belongs to the transition obs_t -> obs_{t+1}.
        with torch.no_grad():
            advantages = torch.zeros_like(self.ri.rewards, device=self.device)  # (env, t)
            gae_next = torch.zeros(
                self.num_envs, device=self.device
            )  # A_{t+1} during backward recursion

            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    value_tp1 = bootstrap_value
                else:
                    value_tp1 = self.ri.values[:, t + 1]

                not_done_t = (~self.ri.dones[:, t]).float()

                # delta_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
                delta_t = (
                    self.ri.rewards[:, t] + gamma * value_tp1 * not_done_t - self.ri.values[:, t]
                )

                # A_t = delta_t + gamma * gae_lambda * (1 - done_t) * A_{t+1}
                gae_next = delta_t + gamma * gae_lambda * not_done_t * gae_next
                advantages[:, t] = gae_next

            returns = advantages + self.ri.values

        self.pri.advantages = advantages
        self.pri.returns = returns

    # -- At training time ------------------------------------------------------------

    def get_minibatch(self, mb_envs: torch.Tensor, mb_steps: torch.Tensor) -> Minibatch:
        """
        Retrieves the data for a given minibatch index, which specifies the environment and rollout step
        indices to retrieve.

        Lets call
            - n_m: minibatch size (number of environments per minibatch)
            - 1: number of rollout steps per minibatch per environment

        Args:
            mb_envs: tensor of shape [n_m] containing the environment indices to retrieve for the minibatch
            mb_steps: tensor of shape [n_m] containing the rollout step indices to retrieve for the minibatch
        Returns:
            A RolloutItem containing the data for the specified minibatch, with tensors of shape [n_m, 1, ...]
        """
        # TODO
        if self.pri.has_nan():
            raise ValueError(
                "Detected nan in the current post-rollout item. Did you forget to "
                "call calculate_advantages_and_returns?"
            )

        if self.ri.actions is None or self.ri.log_probs is None:
            raise ValueError("Rollout item actions or logprobs where not collected.")

        def opt_inter(optional) -> torch.Tensor | None:
            return optional[mb_envs, mb_steps] if optional is not None else None

        minibatch = Minibatch(
            rewards=self.ri.rewards[mb_envs, mb_steps],
            dones=self.ri.dones[mb_envs, mb_steps],
            values=self.ri.values[mb_envs, mb_steps],
            obs=self.ri.obs[mb_envs, mb_steps],
            advantages=self.pri.advantages[mb_envs, mb_steps],
            returns=self.pri.returns[mb_envs, mb_steps],
            envs_t=self.ri.envs_t[mb_envs, mb_steps],
            mem_retrieval_steps=self.ri.mem_retrieval_steps[mb_envs, mb_steps],
            _global_steps=self.ri.global_steps[mb_envs, mb_steps],
            envs=mb_envs.clone(),
            perceived_positions=self.ri.perceived_positions[mb_envs, mb_steps],
            actions=self.ri.actions[mb_envs, mb_steps],  # (512, 1) — minibatch slice
            log_probs=self.ri.log_probs[mb_envs, mb_steps],  # (512, 1) — minibatch slice
            internal_actions=opt_inter(self.ri.internal_actions),
            internal_log_probs=opt_inter(self.ri.internal_log_probs),
            retrieval_hit=opt_inter(self.ri.retrieval_hit),
        )

        return minibatch

    def get_episode_end_steps(self) -> torch.Tensor:
        """
        Retrieves the episode end steps for each environment in the rollout, which is needed for memory trimming.
        Returns:
            A tensor of shape [num_envs] containing the episode end step for each environment in the rollout.
        """
        # episode end step is the first step where done is True, or rollout_steps if no done is True
        done_mask = self.ri.dones  # (num_envs, rollout_steps)
        done_indices = torch.where(
            done_mask, torch.arange(self.rollout_steps, device=self.device), self.rollout_steps
        )
        episode_end_steps, _ = done_indices.min(dim=1)  # (num_envs,)
        return episode_end_steps
