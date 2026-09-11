"""ActivationRenderer — renders transformer layer activations as heatmap grids."""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Baked-in inferno-style 256-entry colour LUT (RGB) ──────────────────────
# Interpolates from black → purple → orange → yellow.
_LUT = np.zeros((256, 3), dtype=np.uint8)
for _i in range(256):
    _t = _i / 255.0
    _r = int(255 * min(1.0, max(0.0, 1.5 * _t - 0.15)))
    _g = int(255 * max(0.0, min(1.0, 1.5 * _t - 0.5)))
    _b = int(255 * max(0.0, min(1.0, 1.0 - 2.0 * abs(_t - 0.35))))
    _LUT[_i] = (_r, _g, _b)


class ActivationRenderer:
    """Renders input embedding + per-layer activations as coloured grids.

    Each activation vector of length *dim* is reshaped into a
    ``ceil(sqrt(dim)) x ceil(sqrt(dim))`` grid, coloured via a LUT,
    and stacked vertically with subtitles.
    """

    _PAD = 4  # px between sections
    _SIDE_PAD = 6  # px left/right margin
    _BG = (30, 30, 40)
    _CELL_PX = 6  # pixels per neuron in the heatmap grid
    _FONT_SIZE = 10  # truetype size for subtitles

    def __init__(self, num_layers: int, dim: int) -> None:
        self._num_layers = num_layers
        self._dim = dim
        self._side = math.ceil(math.sqrt(dim))
        self._font = ImageFont.load_default(size=self._FONT_SIZE)

    def reset(self) -> None:
        self.last_pe = None

    def add_frame_info(
        self,
        activations: np.ndarray | None = None,
        pe_current_token: np.ndarray | None = None,
    ):

        if activations is None:
            self.last_activations = np.zeros((self._num_layers + 1, self._dim), dtype=np.float32)
        else:
            if not activations.shape == (self._num_layers + 1, self._dim):
                raise ValueError(
                    f"Expected activations of shape {(self._num_layers + 1, self._dim)}, got {activations.shape}"
                )
            self.last_activations = activations
        if pe_current_token is None:
            self.last_pe = np.zeros(self._dim, dtype=np.float32)
        else:
            if not pe_current_token.shape == (self._dim,):
                raise ValueError(
                    f"Expected pe_current_token of shape {(self._dim,)}, got {pe_current_token.shape}"
                )
            self.last_pe = pe_current_token

    def generate_frame(self) -> np.ndarray:
        """Return an RGB frame with activation heatmaps for each section."""
        sections: list[tuple[str, np.ndarray]] = []
        sections.append(("Input Embed", self.last_activations[0]))
        for layer in range(self._num_layers):
            sections.append((f"Layer {layer}", self.last_activations[layer + 1]))

        if self.last_pe is not None:
            sections.append(("Pos Encoding", self.last_pe))

        return self._render(sections)

    # ------------------------------------------------------------------

    def _activation_to_rgb(self, activation: np.ndarray) -> np.ndarray:
        """Map a 1-D activation vector to an ``(side, side, 3)`` RGB image."""
        padded = np.zeros(self._side * self._side, dtype=np.float32)
        padded[: len(activation)] = activation

        # normalize to 0..255
        lo, hi = padded.min(), padded.max()
        if hi - lo > 1e-8:
            normed = ((padded - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            normed = np.full_like(padded, 128, dtype=np.uint8)

        rgb = _LUT[normed].reshape(self._side, self._side, 3)
        return rgb

    def _render(self, sections: list[tuple[str, np.ndarray]]) -> np.ndarray:
        """Stack sections vertically: [heatmap + subtitle] per section."""
        pad = self._PAD
        side_pad = self._SIDE_PAD
        cpx = self._CELL_PX
        grid_px = self._side * cpx  # upscaled grid size
        text_h = self._FONT_SIZE + 2
        section_h = grid_px + text_h + pad
        total_h = len(sections) * section_h + pad
        total_w = grid_px + 2 * side_pad

        img = Image.new("RGB", (total_w, total_h), self._BG)
        draw = ImageDraw.Draw(img)

        y = pad
        for title, activation in sections:
            rgb_grid = self._activation_to_rgb(activation)
            grid_img = Image.fromarray(rgb_grid, "RGB").resize(
                (grid_px, grid_px), Image.Resampling.NEAREST
            )
            img.paste(grid_img, (side_pad, y))
            # subtitle centred below the grid
            bbox = self._font.getbbox(title)
            tw = bbox[2] - bbox[0]
            tx = side_pad + (grid_px - tw) // 2
            draw.text((tx, y + grid_px + 1), title, fill=(200, 200, 200), font=self._font)
            y += section_h

        return np.array(img)
