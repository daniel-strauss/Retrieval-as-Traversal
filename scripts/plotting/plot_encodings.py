import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from src_new.model.trxl import GridCellEncoding, PositionalEncoding

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "plots", "trashy_plots")


def plot_positional_encoding():
    T = 10000
    dim = 2**8
    pe = PositionalEncoding(dim=dim, max_seq_len=T, min_timescale=2.0, max_timescale=1e4)

    # pe(T) returns the full (T, dim) buffer
    pe_values = pe(T).numpy()  # (T, dim)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pe_values.T,  # (dim, T) → x=timestep, y=dimension
        ax=ax,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "activation"},
    )
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Dimension index")
    ax.set_title("Sinusoidal Positional Encoding (TrXL)")
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "positional_encoding.png"), dpi=150)
    plt.show()


def _make_cell_positions(grid_w: int, grid_h: int) -> torch.Tensor:
    """Return (grid_w*grid_h, 2) integer grid positions."""
    xs, ys = torch.meshgrid(torch.arange(grid_w), torch.arange(grid_h), indexing="ij")
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)


def _heatmap(ax, data, title=None, xlabel=None, ylabel=None, cbar=True):
    sns.heatmap(
        data,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar=cbar,
        xticklabels=False,
        yticklabels=False,
    )
    if title:
        ax.set_title(title, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def plot_grid_cell_encoding_on_map():
    """One subplot per (direction, frequency) pair showing sin activation over the 2D grid.
    Rows 0-(n_angles-1): individual directions. Last row: sum of sin across all directions.
    """
    s = 200
    grid_w, grid_h = s, s
    num_frequencies_to_plot = 4

    encoding = GridCellEncoding(dim=256, min_scale=10, max_scale=s)
    # read shape metadata directly from the encoding object
    n_angles = encoding.n_angles
    n_rows = n_angles + 1  # +1 for combined row
    row_labels = encoding.angle_labels + ["combined (sum)"]
    freqs = encoding.freqs.tolist()  # already computed in GridCellEncoding.__init__
    freq_indices = torch.linspace(0, encoding.n_scales - 1, num_frequencies_to_plot).long().tolist()

    cell_positions = _make_cell_positions(grid_w, grid_h)
    encodings = encoding(cell_positions)  # (W*H, dim)

    fig, axes = plt.subplots(
        n_rows,
        num_frequencies_to_plot,
        figsize=(4 * num_frequencies_to_plot, 4 * n_rows),
    )

    total_steps = num_frequencies_to_plot * n_rows
    pbar = tqdm(total=total_steps, desc="plot_grid_cell_encoding_on_map")

    for col, fi in enumerate(freq_indices):
        for ai in range(n_angles):
            idx = encoding.get_index_for_angle_and_frequency(ai, fi)
            activation = encodings[:, idx].reshape(grid_w, grid_h).detach().numpy()
            _heatmap(
                axes[ai, col],
                activation,
                title=f"{row_labels[ai]}, freq={freqs[fi]:.3f}",
                ylabel="y" if col == 0 else None,
            )
            pbar.update(1)

        # combined row: sum of sin**2 + cos**2 across all directions at this frequency
        combined = (
            sum(
                (
                    - encodings[:, encoding.get_index_for_angle_and_frequency(ai, fi)]**2 # sin
                    + encodings[:, encoding.get_index_for_angle_and_frequency(ai, fi)+1]**2  # cos
                )
  
                for ai in range(n_angles)  # without *2
            )
            .reshape(grid_w, grid_h)
            .detach()
            .numpy()
        )
        _heatmap(
            axes[n_angles, col],
            combined,
            title=f"{row_labels[n_angles]}, freq={freqs[fi]:.3f}",
            xlabel="x",
            ylabel="y" if col == 0 else None,
        )
        pbar.update(1)

    pbar.close()
    fig.suptitle(
        "Grid Cell Encoding — sin(⟨x, aⱼ⟩·freq) per direction × scale (Space2Vec Eq. 3)",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "grid_cell_encoding_map.png"), dpi=150)
    plt.show()


def plot_grid_cell_sheet():
    """Full sheet: rows = directions + combined, columns = all scales + combined column."""
    grid_w, grid_h = 10, 10

    encoding = GridCellEncoding(dim=256, min_scale=1, max_scale=10)
    n_scales = encoding.n_scales
    n_angles = encoding.n_angles
    angle_labels = [f"dir {l}" for l in encoding.angle_labels] + ["combined (sum)"]
    freqs = encoding.freqs.tolist()
    n_rows = n_angles + 1
    n_cols = n_scales + 1

    cell_positions = _make_cell_positions(grid_w, grid_h)
    with torch.no_grad():
        encodings = encoding(cell_positions)  # (W*H, used_dim)

    # Vectorised extraction: exploit buffer layout (N, n_angles, n_scales, 2) in C order.
    # sin component at [..., 0], cos at [..., 1].
    raw = encodings[:, : n_angles * n_scales * 2].reshape(-1, n_angles, n_scales, 2)  # (N, 3, S, 2)
    acts_sin = (
        raw[..., 0].reshape(grid_w, grid_h, n_angles, n_scales).permute(2, 3, 0, 1)
    )  # (3, S, W, H)

    # Combined panels also use sin-only: sin+cos sums to √2·sin(θ+π/4) which adds no information,
    # and summing across many scales with both components causes destructive multi-scale interference
    # that washes out all spatial structure.
    combined_over_dirs = acts_sin.sum(dim=0)  # (S, W, H) — sum of sin over 3 directions
    combined_over_scales = acts_sin.sum(dim=1)  # (3, W, H) — sum of sin over all S scales
    combined_all = acts_sin.sum(dim=(0, 1))  # (W, H)    — sum of sin over all 3×S

    vmin_sin, vmax_sin = acts_sin.min().item(), acts_sin.max().item()
    vmin_comb, vmax_comb = combined_over_dirs.min().item(), combined_over_dirs.max().item()
    cmap = plt.cm.RdBu_r  # type: ignore[attr-defined]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(12, 1.5 * n_cols), 4 * n_rows),
    )
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    def _imshow(ax, data, vmin, vmax, title=None, ylabel=None, xlabel=None):
        ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8)

    pbar = tqdm(total=n_rows * n_cols, desc="plot_grid_cell_sheet")

    for ai in range(n_angles):
        for fi in range(n_scales):
            _imshow(
                axes[ai, fi],
                acts_sin[ai, fi].numpy(),
                vmin=vmin_sin,
                vmax=vmax_sin,
                title=f"f={freqs[fi]:.3f}" if ai == 0 else None,
                ylabel=angle_labels[ai] if fi == 0 else None,
            )
            pbar.update(1)
        _imshow(
            axes[ai, n_scales],
            combined_over_scales[ai].numpy(),
            vmin=vmin_comb,
            vmax=vmax_comb,
            title="sum(sin, scales)" if ai == 0 else None,
            ylabel=angle_labels[ai],
        )
        pbar.update(1)

    for fi in range(n_scales):
        _imshow(
            axes[n_angles, fi],
            combined_over_dirs[fi].numpy(),
            vmin=vmin_comb,
            vmax=vmax_comb,
            ylabel=angle_labels[n_angles] if fi == 0 else None,
            xlabel="x",
        )
        pbar.update(1)
    _imshow(
        axes[n_angles, n_scales],
        combined_all.numpy(),
        vmin=vmin_comb,
        vmax=vmax_comb,
        ylabel=angle_labels[n_angles],
        xlabel="x",
    )
    pbar.update(1)

    pbar.close()
    fig.suptitle(
        "Grid Cell Encoding sheet — sin component (rows=directions+combined, cols=scales+combined)",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "grid_cell_encoding_sheet.png"), dpi=150)
    plt.show()


def plot_grid_cell_sheet_per_position():
    """Per-position view: outer grid = spatial positions, inner matrix = (n_scales × n_angles)
    showing the sin activation of every neuron for that one position.
    """
    sample_grid_w, sample_grid_h = 5, 5  # positions to sample from the map
    map_w, map_h = 10, 10

    encoding = GridCellEncoding(dim=256, min_scale=1, max_scale=10)
    # read shape metadata directly from the encoding object
    n_scales = encoding.n_scales
    n_angles = encoding.n_angles
    freqs = encoding.freqs.tolist()

    # sample positions evenly across the map
    px = torch.linspace(0, map_w - 1, sample_grid_w).long()
    py = torch.linspace(0, map_h - 1, sample_grid_h).long()
    gx, gy = torch.meshgrid(px, py, indexing="ij")
    sample_pos = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (P, 2)
    encodings = encoding(sample_pos)  # (P, dim)
    P = sample_pos.shape[0]

    # Vectorised: buffer layout (P, n_angles, n_scales, 2) in C order;
    # permute to (P, n_scales, n_angles, 2) then flatten last 2 dims →
    # columns interleave sin/cos per direction: [dir0·sin, dir0·cos, dir1·sin, ...]
    raw = encodings[:, : n_angles * n_scales * 2].detach().reshape(P, n_angles, n_scales, 2)
    act = (
        raw.permute(0, 2, 1, 3).reshape(P, n_scales, n_angles * 2).numpy()
    )  # (P, n_scales, n_angles*2)

    col_labels = [part for lbl in encoding.angle_labels for part in (f"{lbl}·sin", f"{lbl}·cos")]

    fig, axes = plt.subplots(
        sample_grid_h,
        sample_grid_w,
        figsize=(3 * sample_grid_w, 3 * sample_grid_h),
    )

    total_steps = sample_grid_w * sample_grid_h
    pbar = tqdm(total=total_steps, desc="plot_grid_cell_sheet_per_position")

    vmin, vmax = act.min(), act.max()
    for idx in range(total_steps):
        xi, yi = divmod(idx, sample_grid_h)
        pos = sample_pos[idx].tolist()
        ax = axes[yi, xi]  # yi=row so y increases downward matches spatial layout
        # only show tick labels on the edges to avoid overlap
        show_xticklabels = col_labels if yi == sample_grid_h - 1 else False
        sns.heatmap(
            act[idx],  # (n_scales, n_angles*2)
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=show_xticklabels,
            yticklabels=False,
        )
        if show_xticklabels:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right")
        if xi == 0:
            # show ~5 evenly spaced frequency ticks instead of all of them
            n_ticks = 5
            tick_pos = [int(round(i * (n_scales - 1) / (n_ticks - 1))) for i in range(n_ticks)]
            ax.set_yticks([t + 0.5 for t in tick_pos])
            ax.set_yticklabels([f"{freqs[t]:.2f}" for t in tick_pos], fontsize=6, rotation=0)
        ax.set_title(f"pos=({pos[0]},{pos[1]})", fontsize=7)
        pbar.update(1)

    pbar.close()
    fig.suptitle(
        "Grid Cell activations per position — each panel: rows=scales, cols=directions",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "grid_cell_sheet_per_position.png"), dpi=150)
    plt.show()


def plot_encoding_distance_matrix():
    """Pairwise cosine-similarity matrix between flattened grid-cell encodings at every
    position on a grid.  Reveals the spatial periodicity of the representation.
    """
    grid_w, grid_h = 15, 15
    encoding = GridCellEncoding(dim=256, min_scale=1, max_scale=10)
    cell_positions = _make_cell_positions(grid_w, grid_h)  # (N, 2)
    with torch.no_grad():
        vecs = encoding(cell_positions)  # (N, dim)
    # cosine similarity: normalise then dot-product
    vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
    sim = (vecs @ vecs.T).numpy()  # (N, N)

    fig, ax = plt.subplots(figsize=(8, 7))
    img = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")

    n = grid_w * grid_h
    ticks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    labels = [str(cell_positions[t].tolist()) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(
        f"Pairwise cosine similarity of grid-cell encodings ({grid_w}×{grid_h} grid)",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "encoding_distance_matrix.png"), dpi=150)
    plt.show()


def plot_radial_activation_profile():
    """Sin activation for direction 0° as a function of position along the x-axis.

    Uses continuous float positions (no integer grid), so every curve is a smooth
    sinusoid.  The wavelength λ = 1/freq is immediately readable as the period.
    """
    n_points = 1000
    max_x = 20.0

    encoding = GridCellEncoding(dim=256, min_scale=1, max_scale=10)
    n_scales = encoding.n_scales
    n_angles = encoding.n_angles
    freqs = encoding.freqs.tolist()

    # Positions along x-axis (y=0): dot-product with direction 0° (angle=0) equals x directly.
    x_vals = torch.linspace(0.0, max_x, n_points)
    positions = torch.stack([x_vals, torch.zeros_like(x_vals)], dim=-1)  # (n_points, 2)

    with torch.no_grad():
        encodings = encoding(positions)  # (n_points, dim)

    # Vectorised extraction: (n_points, n_angles, n_scales, 2)
    raw = encodings[:, : n_angles * n_scales * 2].reshape(n_points, n_angles, n_scales, 2)
    sin_dir0 = raw[:, 0, :, 0].numpy()  # (n_points, n_scales) — sin for direction 0°

    n_show = min(8, n_scales)
    fi_show = [int(round(i * (n_scales - 1) / (n_show - 1))) for i in range(n_show)]

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.cm.plasma  # type: ignore[attr-defined]
    for k, fi in enumerate(fi_show):
        color = cmap(k / max(n_show - 1, 1))
        ax.plot(
            x_vals.numpy(),
            sin_dir0[:, fi],
            color=color,
            lw=1.2,
            label=f"f={freqs[fi]:.3f}  λ={1 / freqs[fi]:.1f}",
        )

    ax.set_xlabel("position x")
    ax.set_ylabel("sin(x · freq)   [direction 0°]")
    ax.set_title("Grid cell activation profile along x-axis — one sinusoid per frequency band")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "radial_activation_profile.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_positional_encoding()
    plot_grid_cell_encoding_on_map()
    plot_grid_cell_sheet()
    plot_grid_cell_sheet_per_position()
    plot_encoding_distance_matrix()
    plot_radial_activation_profile()
