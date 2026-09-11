"""MiniGrid backend for TrainEnv.

Handles MiniGrid-specific env wrapping and exposes agent_pos, grid dims,
and visible-cell computation through the EnvBackend protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from minigrid.envs.memory import MemoryEnv
from minigrid.minigrid_env import MiniGridEnv

from src_new.env.facade import EnvBackend


class MinigridBackend(EnvBackend):
    """EnvBackend implementation for MiniGrid environments.

    Holds a reference to the unwrapped ``MiniGridEnv`` and provides typed
    access to agent position, grid dimensions, and field-of-view.
    """

    def __init__(self, minigrid_env: MiniGridEnv, max_episode_steps: int) -> None:
        self._env = minigrid_env
        self._max_episode_steps = max_episode_steps

    # -- EnvBackend protocol ------------------------------------------------

    @property
    def agent_pos(self) -> tuple[int, int]:
        pos = self._env.agent_pos
        return (int(pos[0]), int(pos[1]))

    @property
    def grid_width(self) -> int:
        return self._env.width

    @property
    def grid_height(self) -> int:
        return self._env.height

    @property
    def tile_size(self) -> int:
        return self._env.tile_size

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def get_visible_cells(self) -> np.ndarray:
        """Boolean mask ``(grid_w, grid_h)`` of cells visible to the agent."""
        env = self._env
        grid_w, grid_h = env.width, env.height
        view_size = env.agent_view_size

        _, vis_mask = env.gen_obs_grid(view_size)

        ax, ay = env.agent_pos
        dx, dy = env.dir_vec
        rx, ry = env.right_vec
        hs = view_size // 2

        tx = ax + dx * (view_size - 1) - rx * hs
        ty = ay + dy * (view_size - 1) - ry * hs

        locations = (
            np.array([[rx, -dx], [ry, -dy]]) @ np.argwhere(vis_mask).T + np.array([[tx], [ty]])
        ).T

        mask = np.zeros((grid_w, grid_h), dtype=np.bool)
        mask[locations[:, 0], locations[:, 1]] = True
        return mask

    def lower_bound_min_steps(self) -> int:
        return 0

    def reset(self) -> dict[str, Any]:
        """Reset backend state. Call from TrainEnv.reset().

        Returns:
            Dict of info to add to the ``info`` dict returned by TrainEnv.reset().
        """
        return {}

    def step(self, info: dict[str, Any]) -> dict[str, Any]:
        """Update backend state after env.step(). Call from TrainEnv.step().

        Returns:
            Dict of info to add to the ``info`` dict returned by TrainEnv.step().
        """
        return {}


class MinigridMemoryBackend(MinigridBackend):
    """Backend for MiniGrid-Memory environments.

    Adds :meth:`lower_bound_min_steps` which computes the theoretical minimum
    number of actions to solve the current episode.
    """

    def __init__(self, memory_env: MemoryEnv, max_episode_steps: int) -> None:
        super().__init__(memory_env, max_episode_steps)
        self._memory_env: MemoryEnv = memory_env

        # position of the agent at the end of the most recent reset,
        # used to detect if the agent has been respawned at the cue
        self._last_spawn_pos: tuple[int, int] = (-1, -1)

    def lower_bound_min_steps(self) -> int:
        """Minimum actions to solve the current episode.  Call right after reset().

        The agent starts at ``(ax, h//2)`` facing right with ``agent_view_size=3``.
        It must first observe the cue object near ``x=1``, then navigate to the
        correct goal at the T-junction.

        With view_size 3 the cue becomes visible (without moving) when the agent
        faces left from ``x <= 3``, or faces up from ``x <= 2``, or faces right
        at ``x = 1``.  Combining detour + traversal yields:

            lower_bound = ax + hallway_end + 1

        where ``hallway_end = success_pos[0] - 1``.
        """
        ax = int(self._memory_env.agent_pos[0])
        hallway_end = int(self._memory_env.success_pos[0]) - 1
        return ax + hallway_end + 1

    def reset(self) -> dict[str, Any]:
        """Reset backend state. Call from TrainEnv.reset(). Called right after env.reset().

        Returns:
            Dict of info to add to the ``info`` dict returned by TrainEnv.reset().
        """
        self._last_spawn_pos = self.agent_pos
        return {"spawned_at_cue": self._last_spawn_pos[0] == 1}

    def step(self, info: dict[str, Any]) -> dict[str, Any]:
        """Update backend state after env.step(). Call from TrainEnv.step().

        Returns:
            Dict of info to add to the ``info`` dict returned by TrainEnv.step().
        """

        if "episode" in info:
            r = info["episode"]["r"]
            assert len(r) == 1, "Expected episode reward to be a single scalar"
            # reward is success -  0.9 * (step_count / max_steps),
            # but we only care about whether the agent succeeded
            s = float(r[0] > 1e-6) 
            spawned_at_cue = self._last_spawn_pos[0] == 1
            # TODO correctly return reward given spawned_at_cue, without nan
            s_given_spawned_at_cue = s if spawned_at_cue else float("nan")
            s_given_not_spawned_at_cue = s if not spawned_at_cue else float("nan")
            
            
            # we change success to accuracy, because later success will be averaged across episodes.
            return {
                "spawned_at_cue": spawned_at_cue,
                "acc": s,
                "acc_given_spawned_at_cue": s_given_spawned_at_cue,
                "acc_given_not_spawned_at_cue": s_given_not_spawned_at_cue
            }

        return {}
