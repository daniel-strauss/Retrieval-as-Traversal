"""Environment factory — the only thing the Trainer imports to create envs.

``make_env`` returns a thunk suitable for ``gym.vector.SyncVectorEnv``.
Each thunk produces a :class:`TrainEnv` with the correct backend and
optional video recorder already wired in.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.time_limit import TimeLimit
from minigrid.envs.memory import MemoryEnv
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from src_new.env.facade import TrainEnv
from src_new.env.minigrid_backend import MinigridBackend, MinigridMemoryBackend
from src_new.env.video.activation_renderer import ActivationRenderer
from src_new.env.video.agent_view_renderer import AgentViewRenderer
from src_new.env.video.overlay_render import OverlayRender
from src_new.env.video.recorder import VideoRecorder

# TODO: to many args, maybe create a render init pack and a env pack or some shit like that


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    trxl_layers: int,
    trxl_dim: int,
    num_vids_per_trigger: int = 10,
    render_mode: str = "rgb_array",
) -> Callable[[], TrainEnv]:
    """Return a thunk that creates a fully-wrapped :class:`TrainEnv`.

    The thunk is intended for ``gym.vector.SyncVectorEnv([make_env(...), ...])``.
    """

    def thunk() -> TrainEnv:
        if "MiniGrid" in env_id:
            return _make_minigrid_env(
                env_id,
                idx,
                capture_video,
                run_name,
                num_vids_per_trigger,
                render_mode,
                trxl_layers,
                trxl_dim,
            )
        else:
            raise NotImplementedError(
                f"Environment family not yet supported for env_id={env_id!r}. "
                "See memory_gym_backend.py for the next planned backend."
            )

    return thunk


# ---------------------------------------------------------------------------
# MiniGrid-specific construction
# ---------------------------------------------------------------------------


def _make_minigrid_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    num_vids_per_trigger: int,
    render_mode: str,
    trxl_layers: int,
    trxl_dim,
) -> TrainEnv:

    env: gym.Env = gym.make(env_id, agent_view_size=3, tile_size=28, render_mode=render_mode)
    env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=28))
    env = TimeLimit(env, 96)

    # Resolve max_episode_steps (guaranteed by TimeLimit above)
    max_episode_steps = env.spec.max_episode_steps if env.spec else None
    if max_episode_steps is None:
        raise ValueError(
            f"Could not determine max_episode_steps for {env_id}. "
            "Ensure the env is wrapped with TimeLimit or registered with max_episode_steps."
        )

    # Backend (holds ref to unwrapped MiniGridEnv)
    minigrid_env = env.unwrapped
    if not isinstance(minigrid_env, MiniGridEnv):
        raise TypeError(f"Expected MiniGridEnv, got {type(minigrid_env)}")

    if isinstance(minigrid_env, MemoryEnv):
        backend = MinigridMemoryBackend(minigrid_env, max_episode_steps=max_episode_steps)
    else:
        backend = MinigridBackend(minigrid_env, max_episode_steps=max_episode_steps)

    # Stats wrapper
    env = RecordEpisodeStatistics(env)

    # Optional video recorder (only for env 0)
    recorder: VideoRecorder | None = None
    if capture_video and idx == 0:
        recorder = VideoRecorder(
            video_folder=f"videos/trashy_videos/{run_name}",
            overlay_renderer=OverlayRender(
                tile_size=backend.tile_size,
                width=backend.grid_width,
                height=backend.grid_height,
            ),
            agent_view_renderer=AgentViewRenderer(
                grid_w=backend.grid_width,
                grid_h=backend.grid_height,
                res=backend.tile_size,
                max_episode_steps=max_episode_steps,
            ),
            activation_renderer=ActivationRenderer(num_layers=trxl_layers, dim=trxl_dim),
            num_vids_per_trigger=num_vids_per_trigger,
            grid_w=backend.grid_width,
            grid_h=backend.grid_height,
        )

    return TrainEnv(env, backend=backend, video_recorder=recorder)
