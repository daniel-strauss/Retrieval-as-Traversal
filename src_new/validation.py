"""Stateless validation functions for memory windows, minibatches, and attention weights.

Every function here is a pure assertion: it only reads the data it is passed
and raises ``ValueError`` on violations.  No mutable state is required.
"""

import torch

from src_new.memory.types import MemoryWindow
from src_new.trainer.trajectory import Minibatch

# ── Memory window (used by MemoryHandler) ────────────────────────────────────


def validate_memory_window(memory_window: MemoryWindow, *, history_span: int) -> None:
    """Validate shapes, dtypes, bounds, duplicates, and content of a MemoryWindow.

    Args:
        memory_window: The window to validate.
        history_span: Upper-bound (exclusive) for ``memory_indices_env`` values.
    """
    pure_memory_frames = memory_window.frames  # (B, M, L, D)
    rfs_scaled = memory_window.receptive_fields_scaled  # (B, M, L+1, W, H)
    rfs_binary = memory_window.receptive_fields_binary  # (B, M, L+1, W, H)

    memory_masks = memory_window.masks  # (B, M)
    memory_indices = memory_window.indices_env  # (B, M)

    # Shape sanity
    B, M = pure_memory_frames.shape[0], pure_memory_frames.shape[1]
    if memory_masks.ndim != 2 or memory_indices.ndim != 2:
        raise ValueError(
            f"memory_masks/memory_indices must be 2D, got {memory_masks.shape} and {memory_indices.shape}."
        )
    if memory_masks.shape != memory_indices.shape:
        raise ValueError(
            f"memory_masks and memory_indices shape mismatch: {memory_masks.shape} vs {memory_indices.shape}."
        )
    if memory_masks.shape[0] != B:
        raise ValueError(f"Batch mismatch: memory window B={B}, masks B={memory_masks.shape[0]}.")
    if memory_masks.shape[1] != M:
        raise ValueError(f"Window mismatch: memory window M={M}, masks M={memory_masks.shape[1]}.")
    if rfs_scaled.shape[:2] != (B, M):
        raise ValueError(
            f"receptive_fields_scaled must start with (B, M)=({B}, {M}), got {rfs_scaled.shape[:2]}."
        )
    if rfs_binary.shape != rfs_scaled.shape:
        raise ValueError(
            f"receptive_fields_binary shape mismatch: {rfs_binary.shape} vs {rfs_scaled.shape}."
        )

    # Dtype sanity
    if memory_masks.dtype != torch.bool:
        raise ValueError(f"memory_masks must be bool, got {memory_masks.dtype}.")
    if memory_indices.dtype != torch.long:
        raise ValueError(f"memory_indices_env must be long, got {memory_indices.dtype}.")
    if rfs_binary.dtype != torch.bool:
        raise ValueError(f"receptive_fields_binary must be bool, got {rfs_binary.dtype}.")

    # Bounds sanity
    if (memory_indices < 0).any() or (memory_indices >= history_span).any():
        bad = memory_indices[(memory_indices < 0) | (memory_indices >= history_span)][0].item()
        raise ValueError(f"memory index out of range: {bad}, valid range [0, {history_span}).")

    # no overlap sanity
    sorted_indices, _ = torch.sort(memory_indices, dim=1)
    has_duplicates = (
        (sorted_indices[:, 1:] == sorted_indices[:, :-1]) & (sorted_indices[:, :-1] != 0)
    ).any(dim=1)
    if has_duplicates.any():
        bad_row = has_duplicates.nonzero(as_tuple=True)[0][0]
        # TODO fix: think if no duplicae memories should be allowed
        # raise ValueError(
        #    f"Duplicate memory indices in row {bad_row.item()}: {memory_indices[bad_row]}"
        #    f"Duplicate memory indices handling is not yet implemented."
        # )  # type: ignore

    # Content sanity: unmasked tokens should reference non-empty memory frames.
    good_row, good_col = memory_masks.nonzero(as_tuple=True)
    if good_row.numel() > 0:
        unmasked_frames = pure_memory_frames[good_row, good_col]
        all_zero = (unmasked_frames == 0).all(dim=(1, 2))
        if all_zero.any():
            bad_i = all_zero.nonzero(as_tuple=True)[0][0]
            print(
                "Unmasked memory frame is entirely zero at "
                f"row={good_row[bad_i].item()}, token={good_col[bad_i].item()}."
            )


# ── Memory window (used by Agent) ────────────────────────────────────────────


def validate_agent_memory_window(
    memory_window: MemoryWindow,
    episode_step: torch.Tensor,
    *,
    memory_length: int,
    num_layers: int,
    grid_w: int,
    grid_h: int,
    max_episode_steps: int,
) -> None:
    """Validate a MemoryWindow before it enters the agent forward pass.

    Args:
        memory_window: The window to validate.
        episode_step: (B,) per-env current episode step.
        memory_length: Expected M dimension (trxl_memory_length).
        num_layers: Number of transformer layers (L).
        grid_w: Grid width for RF shape checks.
        grid_h: Grid height for RF shape checks.
        max_episode_steps: Upper bound for memory_indices_env values.
    """
    B, M_actual, L, D = memory_window.frames.shape
    M = memory_length
    if memory_window.indices_env.shape != (B, M):
        raise ValueError(
            f"Expected memory_indices_env shape {(B, M)}, got {memory_window.indices_env.shape}"
        )
    if memory_window.masks.shape != (B, M):
        raise ValueError(f"Expected memory_masks shape {(B, M)}, got {memory_window.masks.shape}")
    if memory_window.receptive_fields_scaled.shape != (B, M, L + 1, grid_w, grid_h):
        raise ValueError(
            f"Expected receptive_fields_scaled shape "
            f"{(B, M, L + 1, grid_w, grid_h)}, "
            f"got {memory_window.receptive_fields_scaled.shape}"
        )
    if memory_window.receptive_fields_binary.shape != (B, M, L + 1, grid_w, grid_h):
        raise ValueError(
            f"Expected receptive_fields_binary shape "
            f"{(B, M, L + 1, grid_w, grid_h)}, "
            f"got {memory_window.receptive_fields_binary.shape}"
        )
    if (memory_window.indices_env < 0).any() or (
        memory_window.indices_env >= max_episode_steps
    ).any():
        raise ValueError(
            f"memory_indices_env must be in [0, {max_episode_steps}), got values outside that range."
        )

    # Current and future time steps should be masked out
    if (
        memory_window.masks[
            tuple(torch.argwhere(episode_step[:, None].expand(B, M) <= memory_window.indices_env).T)
        ]
    ).any():
        raise ValueError("Current and future time steps should be masked out in memory_masks.")

    # Masked-out tokens should have zero content
    bad_row, bad_col = (~memory_window.masks.bool()).nonzero(as_tuple=True)
    if bad_row.numel() > 0:
        masked_out_frames = memory_window.frames[bad_row, bad_col]
        masked_out_rfs_scaled = memory_window.receptive_fields_scaled[bad_row, bad_col]
        masked_out_rfs_binary = memory_window.receptive_fields_binary[bad_row, bad_col]
        if masked_out_frames.abs().max().item() > 0:
            raise ValueError("Error: masked out memory frames are not zero.")
        if masked_out_rfs_scaled.abs().max().item() > 0:
            raise ValueError("Error: masked out scaled receptive fields are not zero.")
        if masked_out_rfs_binary.any():
            raise ValueError("Error: masked out binary receptive fields are not zero.")

    # Unmasked tokens should not be entirely zero
    good_row, good_col = memory_window.masks.bool().nonzero(as_tuple=True)
    if good_row.numel() > 0:
        unmasked_frames = memory_window.frames[good_row, good_col]
        all_zero = (unmasked_frames == 0).all(dim=(1, 2))
        if all_zero.any():
            bad_i = all_zero.nonzero(as_tuple=True)[0][0]
            raise ValueError(
                "Error: unmasked memory frame is entirely zero at "
                f"row={good_row[bad_i].item()}, token={good_col[bad_i].item()}."
            )


# ── Attention weights (used by Agent) ────────────────────────────────────────


def validate_attention_weights(
    attention_weights: list[torch.Tensor],
    memory_window: MemoryWindow,
    *,
    num_layers: int,
    memory_length: int,
) -> None:
    """Validate attention weight shapes and values against the memory window.

    Args:
        attention_weights: List of per-layer attention tensors.
        memory_window: The memory window used in the forward pass.
        num_layers: Expected number of transformer layers.
        memory_length: Expected M dimension (trxl_memory_length).
    """
    B = memory_window.masks.shape[0]
    M = memory_length
    L = num_layers

    if len(attention_weights) != L:
        raise ValueError(
            f"Expected {L} attention tensors (one per layer), got {len(attention_weights)}"
        )

    memory_masks = memory_window.masks.bool()  # (B, M)
    has_key = memory_masks.any(dim=1)

    for layer_idx, layer_attn in enumerate(attention_weights):
        if layer_attn.ndim != 4:
            raise ValueError(
                f"Layer {layer_idx}: expected attention shape (B, H, 1, M), got {tuple(layer_attn.shape)}"
            )
        if layer_attn.shape[0] != B or layer_attn.shape[2] != 1 or layer_attn.shape[3] != M:
            raise ValueError(
                f"Layer {layer_idx}: expected shape (B, H, 1, M) with B={B}, M={M}, got {tuple(layer_attn.shape)}"
            )

        attn = layer_attn[:, :, 0, :]  # (B, H, M)
        invalid = (~memory_masks).unsqueeze(1).expand_as(attn)  # noqa: F841
        # TODO
        # if (attn[invalid] > tol).any():
        #    raise ValueError(f"Layer {layer_idx}: non-zero attention mass on masked memory positions.")

        sums = attn.sum(dim=-1)  # (B, H)
        has_key_expanded = has_key.unsqueeze(1).expand_as(sums)
        # if has_key_expanded.any() and not torch.allclose(
        #    sums[has_key_expanded], torch.ones_like(sums[has_key_expanded]), atol=tol, rtol=0
        # ):
        #    raise ValueError(f"Layer {layer_idx}: attention does not sum to 1 for rows with valid keys.")
        # if (~has_key_expanded).any() and not torch.allclose(
        #    sums[~has_key_expanded], torch.zeros_like(sums[~has_key_expanded]), atol=tol, rtol=0
        # ):
        #    raise ValueError(f"Layer {layer_idx}: attention should sum to 0 for fully-masked rows.")

        # debug just debug TODO revert to one check
        if (
            has_key_expanded.any()
            and not (sums[has_key_expanded] > torch.zeros_like(sums[has_key_expanded])).all()
        ):
            raise ValueError(
                f"Layer {layer_idx}: attention does not sum to > 0 for rows with valid keys."
            )


# ── Minibatch (used by Trainer) ──────────────────────────────────────────────


def validate_minibatch(mb: Minibatch, memory_window: MemoryWindow) -> None:
    """Check that a minibatch and its memory window are internally consistent.

    Validates shape agreement, global-step passthrough, memory-slot ordering,
    frame content, RF monotonicity, and NaN/Inf absence.
    """
    B = mb.envs_t.shape[0]
    masks = memory_window.masks  # (B, M)
    frames = memory_window.frames  # (B, M, L, D)
    rf_binary = memory_window.receptive_fields_binary  # (B, M, L+1, W, H)

    # ── Shape sanity ──────────────────────────────────────────────────────
    if memory_window.envs_t.shape != (B,):
        raise ValueError(
            f"memory_window.envs_t shape {memory_window.envs_t.shape} != expected ({B},)"
        )
    if memory_window.global_steps.shape != (B,):
        raise ValueError(
            f"memory_window.global_steps shape {memory_window.global_steps.shape} != expected ({B},)"
        )

    # ── global_steps passthrough ──────────────────────────────────────────
    if not torch.equal(memory_window.global_steps, mb.mem_retrieval_steps):
        raise ValueError(
            "global_steps mismatch: memory_window.global_steps != mb.mem_retrieval_steps"
        )

    # ── envs_t consistency (per-slot, vectorized) ─────────────────────────
    mem_slot_steps = memory_window.indices_env  # (B, M)

    filled = mem_slot_steps.clone()
    filled[~masks] = -1

    # 1. Monotonicity via cummax
    cummax_vals, _ = filled.cummax(dim=1)  # (B, M)
    violations = masks & (filled < cummax_vals)  # (B, M)
    if violations.any():
        bad_rows = violations.any(dim=1)
        bad_idx = bad_rows.nonzero(as_tuple=False).squeeze(-1)
        b0 = bad_idx[0].item()
        raise ValueError(
            f"Memory slot env-steps not in order for {bad_rows.sum().item()} "
            f"minibatch elements, e.g. element {b0}: "
            f"{mem_slot_steps[b0][masks[b0]].tolist()}"  # type: ignore
        )  # type: ignore

    # 2. max(active slot env-step) + 1 == mb.envs_t
    has_active = masks.any(dim=1)  # (B,)
    non_boundary = mb.envs_t > 0  # (B,)
    check_mask = has_active & non_boundary  # (B,)
    if check_mask.any():
        max_slot_step, _ = filled.max(dim=1)  # (B,)
        mismatch = check_mask & (max_slot_step + 1 != mb.envs_t)
        if mismatch.any():
            bad_idx = mismatch.nonzero(as_tuple=False).squeeze(-1)
            b0 = bad_idx[0].item()
            raise ValueError(
                f"envs_t mismatch for {mismatch.sum().item()} minibatch elements. "
                f"E.g. element {b0}: max_slot_step={max_slot_step[b0].item()}, "  # type: ignore
                f"max+1={max_slot_step[b0].item() + 1}, mb.envs_t={mb.envs_t[b0].item()}"  # type: ignore
            )  # type: ignore

    # ── Unmasked memory frames must be non-zero in every layer ────────────
    if masks.any():
        unmasked_frames = frames[masks]  # (K, L, D)
        per_layer_ok = (unmasked_frames != 0).any(dim=-1)  # (K, L)
        if not per_layer_ok.all():
            bad_count = int((~per_layer_ok).any(dim=-1).sum().item())
            raise ValueError(
                f"{bad_count} unmasked memory slots have all-zero values in at least one layer."
            )

    # ── RF binary monotonicity: layer l subset layer l+1 ─────────────────────
    if masks.any():
        unmasked_rf = rf_binary[masks]  # (K, L+1, W, H)
        num_rf_layers = unmasked_rf.shape[1]
        for layer in range(num_rf_layers - 1):
            violation = unmasked_rf[:, layer] & ~unmasked_rf[:, layer + 1]
            if violation.any():
                n_bad = int(violation.any(dim=(-1, -2)).sum().item())
                raise ValueError(
                    f"RF monotonicity violated: {n_bad} slots have cells True "
                    f"in layer {layer} but False in layer {layer + 1}."
                )

    # ── No NaN / Inf in critical tensors ─────────────────────────────────
    if torch.isnan(frames).any() or torch.isinf(frames).any():
        raise ValueError("NaN or Inf in memory_window.pure_memory_frames")
    if torch.isnan(mb.values).any() or torch.isinf(mb.values).any():
        raise ValueError("NaN or Inf in mb.values")
