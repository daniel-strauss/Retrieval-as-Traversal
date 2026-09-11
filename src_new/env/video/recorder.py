"""VideoRecorder — owns renderers, frame buffers, trigger logic, and h264 encoding.

Extracted from CustomRecordVideo so that video concerns are decoupled from the
gym.Wrapper env interface.  CustomRecordVideo (and later TrainEnv) delegates all
recording to an instance of this class.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from fractions import Fraction

import av
import numpy as np
import torch
from gymnasium import logger
from PIL import Image

from src_new.env.video.activation_renderer import ActivationRenderer
from src_new.env.video.agent_view_renderer import AgentViewRenderer
from src_new.env.video.overlay_render import OverlayRender
from src_new.trainer.forward_diagnostics import ForwardDiagnostics


def _to_numpy(val: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    """Convert a tensor to numpy; pass through ndarray/None unchanged."""
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.cpu().numpy()
    return val


def _to_pos(val: torch.Tensor | np.ndarray | tuple | None) -> tuple[int, int] | None:
    """Coerce a 2-element position to tuple[int, int]."""
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return (int(val[0].item()), int(val[1].item()))
    if isinstance(val, np.ndarray):
        return (int(val[0]), int(val[1]))
    return (int(val[0]), int(val[1]))


class VideoRecorder:
    """Records two-panel videos (main overlay + agent-view timeline).

    Lifecycle managed by the owning env wrapper:
        - Construction: pass renderers config + trigger callbacks
        - ``on_reset(render_fn)``  — called every env reset
        - ``record_frame(...)``    — called every step (no-ops when not recording)
        - ``on_step()``            — per-step trigger check
        - ``close()``              — flush last video if recording
    """

    def __init__(
        self,
        *,
        video_folder: str,
        overlay_renderer: OverlayRender,
        agent_view_renderer: AgentViewRenderer,
        activation_renderer: ActivationRenderer,
        episode_trigger: Callable[[int], bool] | None = None,
        num_vids_per_trigger: int = 10,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int = 3,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
        # grid dims for zero-RF frames (reset / final frame)
        grid_w: int = 1,
        grid_h: int = 1,
    ) -> None:
        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        self.overlay_renderer = overlay_renderer
        self.agent_view_renderer = agent_view_renderer
        self.activation_renderer = activation_renderer

        self.episode_trigger = episode_trigger
        self.num_vids_per_trigger = num_vids_per_trigger
        self.step_trigger = step_trigger
        self.gc_trigger = gc_trigger

        self.video_length: int = video_length if video_length != 0 else float("inf")  # type: ignore
        self.name_prefix = name_prefix
        self.frames_per_sec = fps

        self.grid_w = grid_w
        self.grid_h = grid_h

        # ── mutable state ──
        self.recording: bool = False
        self._video_name: str | None = None
        self.recorded_main_frames: list[np.ndarray] = []
        self.recorded_activation_frames: list[np.ndarray] = []
        self.recorded_agent_frames: list[np.ndarray] = []
        self.step_id: int = -1
        self.episode_id: int = -1
        self._completed_videos: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by the env wrapper)
    # ------------------------------------------------------------------

    def on_pre_reset(self, render_fn: Callable[[], object]) -> None:
        """Called BEFORE env.reset().  Stops the previous recording if active."""
        if self.recording and self.video_length == float("inf"):
            self._record_blank_frame(render_fn)
            self.stop_recording()

    def on_post_reset(self, render_fn: Callable[[], object]) -> None:
        """Called AFTER env.reset().  Handles episode trigger + first frame."""
        self.episode_id += 1
        self.agent_view_renderer.reset()
        self.activation_renderer.reset()

        if self.episode_trigger and any(
            self.episode_id - i >= 0 and self.episode_trigger(self.episode_id - i)
            for i in range(self.num_vids_per_trigger)
        ):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
            self._record_blank_frame(render_fn)

    def on_step(self) -> None:
        """Called after every env step for step-trigger bookkeeping."""
        self.step_id += 1
        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")

    def record_frame(
        self,
        render_fn: Callable[[], object],
        *,
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
        lower_bound_min_steps: int | None = None,
    ) -> None:
        """Capture one frame with overlays.  No-ops when not recording.

        Conversion to numpy happens here (only when actually recording)
        to avoid unnecessary GPU→CPU transfers on non-recording steps.
        """
        if not self.recording:
            return

        expected = len(self.recorded_main_frames) - 1
        if env_t != expected:
            raise ValueError(
                f"record_frame step mismatch: caller says env_t={env_t}, "
                f"but recorder expects {expected} "
                f"(recorded_frames={len(self.recorded_main_frames)})"
            )

        self._capture(
            render_fn,
            rf_binary=_to_numpy(rf_binary),  # type: ignore[arg-type]  # guaranteed non-None
            rf_scaled=_to_numpy(rf_scaled),  # type: ignore[arg-type]  # guaranteed non-None
            perceived_pos=_to_pos(perceived_pos),
            memory_retrieval_pos=_to_pos(memory_retrieval_pos),
            current_env_t=env_t,
            memory_indices_env=_to_numpy(memory_indices_env),
            memory_masks=_to_numpy(memory_masks),
            retrieved_fov=_to_numpy(retrieved_fov),
            cumulative_fov=_to_numpy(cumulative_fov),
            forward_diagnostics=forward_diagnostics,
            lower_bound_min_steps=lower_bound_min_steps
        )

        if len(self.recorded_main_frames) > self.video_length:
            self.stop_recording()

    def close(self) -> None:
        if self.recording:
            self.stop_recording()

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def drain_completed_videos(self) -> list[str]:
        """Return and clear the list of completed video paths."""
        paths = self._completed_videos
        self._completed_videos = []
        return paths

    def start_recording(self, video_name: str) -> None:
        if self.recording:
            self.stop_recording()
        self.recording = True
        self._video_name = video_name

    def stop_recording(self) -> None:
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_main_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            if len(self.recorded_main_frames) != len(self.recorded_agent_frames):
                raise ValueError(
                    f"Recorded frames and agent view frames have different lengths: "
                    f"main={len(self.recorded_main_frames)} agent={len(self.recorded_agent_frames)} "
                    f"activation={len(self.recorded_activation_frames)}. "
                    "This should never happen, please debug."
                )
            self._encode_video()
            self._completed_videos.append(
                os.path.join(self.video_folder, f"{self._video_name}.mp4")
            )

        self.recorded_main_frames = []
        self.recorded_activation_frames = []
        self.recorded_agent_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_blank_frame(self, render_fn: Callable[[], object]) -> None:
        """Record a frame with zero RF (used for reset first-frame and final frame)."""
        self._capture(
            render_fn,
            rf_binary=np.zeros((self.grid_h, self.grid_w), dtype=bool),
            rf_scaled=np.zeros((self.grid_h, self.grid_w), dtype=float),
            perceived_pos=None,
            memory_retrieval_pos=None,
            forward_diagnostics=None,
        )

    def _capture(
        self,
        render_fn: Callable[[], object],
        *,
        rf_binary: np.ndarray,
        rf_scaled: np.ndarray,
        perceived_pos: tuple[int, int] | None = None,
        memory_retrieval_pos: tuple[int, int] | None = None,
        current_env_t: int | None = None,
        memory_indices_env: np.ndarray | None = None,
        memory_masks: np.ndarray | None = None,
        retrieved_fov: np.ndarray | None = None,
        cumulative_fov: np.ndarray | None = None,
        forward_diagnostics: ForwardDiagnostics | None = None,
        lower_bound_min_steps: int | None = None,
    ) -> None:
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        base_frame = render_fn()
        if not isinstance(base_frame, np.ndarray):
            raise TypeError(
                f"render_fn must return np.ndarray, got {type(base_frame)}. "
                "Did you set render_mode to rgb_array?"
            )

        main_frame = self.overlay_renderer.add_image_overlay(
            base=base_frame,
            rf_binary=rf_binary,
            rf_scaled=rf_scaled,
        )

        self.agent_view_renderer.add_frame_info(
            agent_pos=perceived_pos,
            memory_retrieval_pos=memory_retrieval_pos,
            current_env_t=current_env_t,
            memory_indices_env=memory_indices_env,
            memory_masks=memory_masks,
            retrieved_fov=retrieved_fov,
            cumulative_fov=cumulative_fov,
            lower_bound_min_steps=lower_bound_min_steps
        )

        self.activation_renderer.add_frame_info(
            activations=_to_numpy(forward_diagnostics.layer_activations)
            if forward_diagnostics
            else None,
            pe_current_token=_to_numpy(forward_diagnostics.pe_current_token)
            if forward_diagnostics
            else None,
        )

        activation_frame = self.activation_renderer.generate_frame()

        agent_view_frame = self.agent_view_renderer.generate_frame()

        if (
            not isinstance(main_frame, np.ndarray)
            or not isinstance(agent_view_frame, np.ndarray)
            or not isinstance(activation_frame, np.ndarray)
        ):
            raise TypeError(
                f"Expected np.ndarray frames, got {type(main_frame)} and {type(agent_view_frame)} "
                f"and {type(activation_frame)}"
            )

        self.recorded_main_frames.append(main_frame)
        self.recorded_activation_frames.append(activation_frame)
        self.recorded_agent_frames.append(agent_view_frame)

    def _encode_video(self) -> None:
        path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
        container = av.open(path, mode="w")
        stream = container.add_stream("libx264", rate=self.frames_per_sec)
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "preset": "veryfast",
            "crf": "23",
            "profile": "baseline",
            "level": "3.1",
        }
        stream.sample_aspect_ratio = Fraction(1, 1)

        # Pre-compute even dimensions from first frame and pad all frames
        first_combined = self._combine_panels(0)
        h, w = first_combined.shape[:2]
        # yuv420p requires even width and height
        target_w = w + (w % 2)
        target_h = h + (h % 2)
        stream.width = target_w
        stream.height = target_h

        for i in range(len(self.recorded_main_frames)):
            combined = self._combine_panels(i) if i > 0 else first_combined
            # Pad frame to even dimensions if needed
            ch, cw = combined.shape[:2]
            if cw != target_w or ch != target_h:
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                padded[:ch, :cw] = combined
                combined = padded
            video_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
            video_frame.pts = i
            video_frame.time_base = Fraction(1, self.frames_per_sec)
            packet = stream.encode(video_frame)
            container.mux(packet)

        container.mux(stream.encode())
        container.close()

    def _combine_panels(self, idx: int) -> np.ndarray:
        """Combine panels for frame *idx*: main | activation | agent_view.

        All panels are scaled to the same height (the tallest) while
        preserving each panel's aspect ratio, then stacked horizontally.
        """
        panels = [
            self.recorded_main_frames[idx],
            self.recorded_activation_frames[idx],
            self.recorded_agent_frames[idx],
        ]

        target_h = max(p.shape[0] for p in panels)
        scaled = []
        for p in panels:
            if p.shape[0] == target_h:
                scaled.append(p)
            else:
                w = max(1, int(p.shape[1] * target_h / p.shape[0]))
                scaled.append(
                    np.array(Image.fromarray(p).resize((w, target_h), Image.Resampling.NEAREST))
                )

        return np.hstack(scaled)

    def __del__(self) -> None:
        if len(self.recorded_main_frames) > 0:
            logger.warn("Unable to save last video! Did you call close()?")


# ----------------------------------------------------------------------
# Module-level helper (no state needed)
# ----------------------------------------------------------------------


def _synch_frames(
    frame_a: np.ndarray, frame_b: np.ndarray, mode: str = "upscale", v_stack: bool = False
) -> np.ndarray:
    """Combine two frames side-by-side (or stacked), rescaling if needed."""
    stack_dim, glue_dim = (0, 1) if v_stack else (1, 0)

    if frame_a.shape[glue_dim] == frame_b.shape[glue_dim]:
        return np.vstack([frame_a, frame_b]) if v_stack else np.hstack([frame_a, frame_b])

    if mode == "upscale":
        fs = [frame_a, frame_b]
        smaller = 0 if fs[0].shape[glue_dim] < fs[1].shape[glue_dim] else 1
        glued_dim_size = fs[1 - smaller].shape[glue_dim]
        smaller_img = Image.fromarray(fs[smaller])
        new_h_sm = int(fs[smaller].shape[stack_dim] / fs[smaller].shape[glue_dim] * glued_dim_size)
        fs[smaller] = np.array(
            smaller_img.resize((glued_dim_size, new_h_sm), Image.Resampling.NEAREST)
        )
        return np.vstack(fs) if v_stack else np.hstack(fs)

    if mode == "pad":
        target_w = max(frame_a.shape[stack_dim], frame_b.shape[stack_dim])
        frame_a = np.pad(frame_a, ((0, 0), (0, target_w - frame_a.shape[stack_dim]), (0, 0)))
        frame_b = np.pad(frame_b, ((0, 0), (0, target_w - frame_b.shape[stack_dim]), (0, 0)))
        return np.vstack([frame_a, frame_b]) if v_stack else np.hstack([frame_a, frame_b])

    raise ValueError(f"Unknown mode {mode}. Supported: 'upscale', 'pad'.")
