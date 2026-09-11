import numpy as np


class OverlayRender:
    """
    Rendering overlay interface for CustomRecordVideo environment wrapper.

    Whenever CustomRecorVideo renders a frame from the minigrid env, it calls this class afterwards to
    generate an overlay, with additional information, such as
        - curcles on each cell whos area relect the attention weight of the output layer (scaled mode)
        - borders on each cell where the attention weight is nonzero (binary mode)

    Args:
        --- values from env.unrappwed ---
        tile_size: Size of a single cell in pixels (e.g. 32).
        width: Number of cells in the horizontal direction.
        height: Number of cells in the vertical direction.
    TODO stop monkey patching that shit.

    """

    def __init__(
        self,
        tile_size: int,
        width: int,
        height: int,
    ) -> None:

        self.tile_size: int = tile_size
        self.grid_w: int = width
        self.grid_h: int = height

        # ── cached static arrays for add_shape (computed once per instance) ──
        ts = tile_size
        _fill_tile = np.ones((ts, ts), dtype=bool)
        _t = 0.08  # default border thickness
        _ys, _xs = np.mgrid[0:ts, 0:ts]
        _xn = (_xs + 0.5) / ts
        _yn = (_ys + 0.5) / ts
        _border_tile = ~((_xn >= _t) & (_xn <= 1 - _t) & (_yn >= _t) & (_yn <= 1 - _t))
        self._fill_shape_px: np.ndarray = np.tile(_fill_tile, (height, width))
        self._border_shape_px: np.ndarray = np.tile(_border_tile, (height, width))
        _cy = _cx = (ts - 1) / 2.0
        _dist_tile = np.sqrt((_xs - _cx) ** 2 + (_ys - _cy) ** 2).astype(np.float32)
        self._dist_px: np.ndarray = np.tile(_dist_tile, (height, width))

    # ------------------------------------------------------------------
    # Static helper – must be called *before* gym.make() because
    # gymnasium locks render_mode at environment creation time.
    # ------------------------------------------------------------------
    @staticmethod
    def get_render_mode(env_id: str, idx: int, render_debug: bool) -> str:
        """Return the appropriate ``render_mode`` for a given env index.

        Args:
            env_id: The gymnasium environment id (e.g. ``"MiniGrid-MemoryS9-v0"``).
            idx: Index of this environment inside the vectorised set.
            render_debug: Whether debug rendering is enabled.

        Returns:
            ``"human"`` for env 0 when *render_debug* is ``True``,
            otherwise ``"rgb_array"``.
        """
        raise NotImplementedError("Became Obsolete")

        if render_debug and idx == 0:
            return "human"
        return "rgb_array"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_render(self) -> None:
        """Trigger an initial render (opens the human window if *render_debug*)
        and discard any stale overlay layers."""
        raise NotImplementedError("Became Obsolete")
        self._overlay_layers.clear()
        # if self.render_debug:
        #    self.env.render()

    def reset_debug_overlays(self) -> None:
        """Discard queued overlays and restore the original render path."""
        raise NotImplementedError("Became Obsolete")
        self._overlay_layers.clear()
        self.unwrapped.get_full_render = self._original_get_full_render

    def add_image_overlay(
        self, base: np.ndarray, rf_scaled: np.ndarray, rf_binary: np.ndarray
    ) -> np.ndarray:
        """
        Creates an overlay on the image based on the previded receptive fields and returns it.
        Args:
            image: The original image to be overlaid.
            rf_scaled: A 2D array representing the scaled receptive field values for each cell.
            rf_binary: A 2D array representing the binary receptive field values for each cell.
        Returns:
            A new image with the overlay applied.
        """

        ##
        # I. create overlay layer
        ##

        rf_max = rf_scaled.max()
        rf_scaled_norm = (rf_scaled / rf_max if rf_max > 0 else rf_scaled).astype(np.float32)

        # binary: uniform alpha where rf > 0
        rf_bin_norm = (rf_binary > 0).astype(np.float32)

        # Build both alpha masks, then composite in a single pass
        circle_alpha = self._alpha_scaled_circle(rf_scaled_norm)
        border_alpha = self._alpha_border(rf_bin_norm)

        img = base.astype(np.float32)

        # Circle overlay (purple)
        a1 = circle_alpha[:, :, np.newaxis]  # (H, W, 1)
        c1 = np.float32([125, 0, 255])
        img = (1.0 - a1) * img + a1 * c1

        # Border overlay (red)
        a2 = border_alpha[:, :, np.newaxis]  # (H, W, 1)
        c2 = np.float32([255, 0, 0])
        img = (1.0 - a2) * img + a2 * c2

        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Fast alpha-only helpers (skip 4-channel RGBA allocation)
    # ------------------------------------------------------------------
    def _alpha_scaled_circle(
        self, scaled_grid_mask: np.ndarray, max_alpha: float = 0.3
    ) -> np.ndarray:
        """Return (height_px, width_px) float32 alpha for scaled-circle overlay."""
        ts = self.tile_size
        r_px = np.repeat(
            np.repeat(
                (np.sqrt(scaled_grid_mask) * (ts / 2.0)).astype(np.float32),
                ts,
                axis=0,
            ),
            ts,
            axis=1,
        ).T
        return (self._dist_px <= r_px).astype(np.float32) * max_alpha

    def _alpha_border(self, scaled_grid_mask: np.ndarray, max_alpha: float = 0.3) -> np.ndarray:
        """Return (height_px, width_px) float32 alpha for border overlay."""
        ts = self.tile_size
        alpha_px = np.repeat(np.repeat(scaled_grid_mask, ts, axis=0), ts, axis=1).T
        return alpha_px * self._border_shape_px * max_alpha

    # ------------------------------------------------------------------
    # Core drawing primitive
    # ------------------------------------------------------------------
    def add_shape(
        self,
        scaled_grid_mask: np.ndarray,
        *,
        shape: str = "fill",
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: float = 0.08,
        max_alpha: float = 0.3,
    ) -> np.ndarray:
        """Queue an overlay shape on grid cells weighted by *scaled_grid_mask*.

        All pixel-level calculations are confined to this method.  The
        resulting RGBA overlay is appended to an internal queue and
        composited onto the rendered frame only when :meth:`flush` is
        called.

        Args:
            scaled_grid_mask: Float array of shape ``(grid_w, grid_h)``
                with values in ``[0, 1]`` controlling per-cell intensity.
                Cells with value ``0`` are not drawn.
            shape: ``"fill"`` fills entire cells; ``"border"`` draws an
                inset rectangular border; ``"scaled_circle"`` draws a
                filled circle whose *area* is proportional to the mask
                value (mask=1 → inscribed circle, mask=0.5 → half area).
            color: RGB tuple ``(R, G, B)`` for the overlay colour.
            thickness: Border width as a fraction of tile size, range
                ``(0, 0.5)``.  Only used when *shape* is ``"border"``.
            max_alpha: Maximum alpha value for the overlay.  For
                ``"scaled_circle"`` this is a flat opacity (intensity is
                encoded in circle area, not transparency).
        """

        tile_size = self.tile_size
        grid_w = self.grid_w
        grid_h = self.grid_h

        height_px = grid_h * tile_size
        width_px = grid_w * tile_size

        if shape in ("fill", "border"):
            # -- use cached shape mask; recompute only for non-default thickness -
            if shape == "fill":
                shape_px = self._fill_shape_px
            elif thickness == 0.08:
                shape_px = self._border_shape_px
            else:
                t = thickness
                ys, xs = np.mgrid[0:tile_size, 0:tile_size]
                xn = (xs + 0.5) / tile_size
                yn = (ys + 0.5) / tile_size
                inner = (xn >= t) & (xn <= 1 - t) & (yn >= t) & (yn <= 1 - t)
                shape_px = np.tile(~inner, (grid_h, grid_w))

            # -- upscale grid mask → pixel resolution (no large ones allocation) -
            # scaled_grid_mask is (grid_w, grid_h); repeat+T → (height_px, width_px).
            alpha_px = np.repeat(
                np.repeat(scaled_grid_mask, tile_size, axis=0), tile_size, axis=1
            ).T  # (height_px, width_px)

            # -- combine: per-cell intensity × shape mask ---------------
            alpha_combined = alpha_px * shape_px * max_alpha

        elif shape == "scaled_circle":
            # Intensity is encoded in circle area: A ∝ mask, so r ∝ √mask.
            # At mask=1 the circle is inscribed in the tile (r = tile_size/2).

            # -- upscale per-cell radius to pixel resolution (no large ones allocation) -
            r_px = np.repeat(
                np.repeat(
                    (np.sqrt(scaled_grid_mask) * (tile_size / 2.0)).astype(np.float32),
                    tile_size,
                    axis=0,
                ),
                tile_size,
                axis=1,
            ).T  # (height_px, width_px)

            # -- circle mask using cached tiled distance map ------------
            alpha_combined = (self._dist_px <= r_px).astype(np.float32) * max_alpha

        else:
            raise ValueError(
                f"Unknown shape {shape!r}. Supported: 'fill', 'border', 'scaled_circle'."
            )

        # -- build RGBA overlay and enqueue -----------------------------
        overlay = np.zeros((height_px, width_px, 4), dtype=np.float32)
        overlay[:, :, 0] = color[0]
        overlay[:, :, 1] = color[1]
        overlay[:, :, 2] = color[2]
        overlay[:, :, 3] = alpha_combined

        return overlay
