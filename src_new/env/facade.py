"""TrainEnv — the only env type the Trainer sees.

Wraps a gymnasium environment and delegates env-family-specific queries
(agent position, visible cells, grid dimensions) to an :class:`EnvBackend`.
Optionally owns a :class:`VideoRecorder` for frame capture.
"""

from __future__ import annotations

from typing import Any, Dict, SupportsFloat
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ActType, ObsType

from src_new.env.video.recorder import VideoRecorder
from src_new.trainer.forward_diagnostics import ForwardDiagnostics

# ---------------------------------------------------------------------------
# Backend protocol — one implementation per env family
# ---------------------------------------------------------------------------



class EnvBackend(ABC):
    """What a backend must provide so TrainEnv can expose a uniform API."""

    @property
    @abstractmethod
    def agent_pos(self) -> tuple[int, int]:
        """Current (x, y) position of the agent."""
        ...

    @property
    @abstractmethod
    def grid_width(self) -> int: ...

    @property
    @abstractmethod
    def grid_height(self) -> int: ...

    @property
    @abstractmethod
    def max_episode_steps(self) -> int: ...

    @abstractmethod
    def get_visible_cells(self) -> np.ndarray:
        """Boolean mask ``(grid_w, grid_h)`` of currently visible cells."""
        ...
    
    @abstractmethod
    def lower_bound_min_steps(self) -> int:
        """Theoretical minimum actions to solve the current episode (0 if unknown)."""
        ...


    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the backend's internal state. Returns additional info for Trainer. 
        Called right after env.reset()."""
        ...

    @abstractmethod
    def step(self, info: dict[str, Any]) -> Dict[str, Any]:
        """Update backend state based on the taken action. Returns additional info for Trainer.
        Called right after env.step()."""
        ...


# ---------------------------------------------------------------------------
# TrainEnv — thin gym.Wrapper facade
# ---------------------------------------------------------------------------


class TrainEnv(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Uniform env interface for the Trainer.

    The Trainer never touches ``env.unwrapped`` — it uses the typed
    properties below instead.  Video recording is handled via
    :meth:`capture_frame`, which no-ops when no recorder is attached.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        backend: EnvBackend,
        video_recorder: VideoRecorder | None = None,
    ) -> None:
        super().__init__(env)
        self._backend = backend
        self._recorder = video_recorder
        self._current_lower_bound: int = 0

    # ------------------------------------------------------------------
    # Backend-delegated properties
    # ------------------------------------------------------------------

    @property
    def agent_pos(self) -> tuple[int, int]:
        return self._backend.agent_pos

    @property
    def grid_width(self) -> int:
        return self._backend.grid_width

    @property
    def grid_height(self) -> int:
        return self._backend.grid_height

    @property
    def max_episode_steps(self) -> int:
        return self._backend.max_episode_steps

    def get_visible_cells(self) -> np.ndarray:
        return self._backend.get_visible_cells()

    @property
    def current_lower_bound(self) -> int:
        """Lower-bound min steps captured at the most recent reset."""
        return self._current_lower_bound

    # ------------------------------------------------------------------
    # gym.Wrapper overrides (video hooks)
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if self._recorder is not None:
            self._recorder.on_pre_reset(render_fn=self.env.render)
        obs, info = super().reset(seed=seed, options=options)
        self._current_lower_bound = self._backend.lower_bound_min_steps()

        backend_info = self._backend.reset()
        if 'episode' in info:
            info['episode'].update(backend_info)
        else:
            info['episode'] = backend_info

        if self._recorder is not None:
            self._recorder.on_post_reset(render_fn=self.env.render)
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self._recorder is not None:
            self._recorder.on_step()
        
        # update backend and get info from backend
        backend_info = self._backend.step(info)

        # TODO: backend step, backend adds additional info (e.g. retrieval success), update _current_lower_bound if new lower bound info is available
        if "episode" in info:
            if self._current_lower_bound > 0:
                info["episode"]["steps_above_min"] = (
                    int(info["episode"]["l"]) - self._current_lower_bound
                )

            info["episode"].update(backend_info)


        return obs, rew, terminated, truncated, info

    def close(self) -> None:
        super().close()
        if self._recorder is not None:
            self._recorder.close()

    def drain_completed_videos(self) -> list[str]:
        """Return and clear completed video paths. Empty list if no recorder."""
        if self._recorder is None:
            return []
        return self._recorder.drain_completed_videos()

    # ------------------------------------------------------------------
    # Video capture (no-ops when recorder is None)
    # ------------------------------------------------------------------

    def capture_frame(
        self,
        env_t: int,
        rf_scaled: torch.Tensor,
        rf_binary: torch.Tensor,
        perceived_pos: torch.Tensor | None = None,
        memory_retrieval_pos: torch.Tensor | None = None,
        memory_indices_env: torch.Tensor | None = None,
        memory_masks: torch.Tensor | None = None,
        retrieved_fov: torch.Tensor | None = None,
        cumulative_fov: torch.Tensor | None = None,
        forward_diagnostics: ForwardDiagnostics | None = None,
    ) -> None:
        """Capture the current (pre-step) env state with RF overlay.

        Must be called BEFORE ``env.step()`` so the rendered frame shows the
        state the agent actually observed.  No-ops when no recorder is attached.
        """
        if self._recorder is None:
            return
        self._recorder.record_frame(
            render_fn=self.env.render,
            env_t=env_t,
            rf_scaled=rf_scaled,
            rf_binary=rf_binary,
            perceived_pos=perceived_pos,
            memory_retrieval_pos=memory_retrieval_pos,
            memory_indices_env=memory_indices_env,
            memory_masks=memory_masks,
            retrieved_fov=retrieved_fov,
            cumulative_fov=cumulative_fov,
            forward_diagnostics=forward_diagnostics,
            lower_bound_min_steps=self._current_lower_bound,
        )
