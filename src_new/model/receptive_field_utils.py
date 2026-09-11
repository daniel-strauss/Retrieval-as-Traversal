"""
Utility for computing fields


Agent will call this utility to compute receptive field

Two main uscases:
    - receptive fields ov envneed to be plotted.
        -> This class can return the RF just for env 0.
    - receptive fields are needed for spatial memory writing.
        -> This class can compute a tanseor that contains all receptive fields for a given timestep


to compute and store RFs for env 0, and to cache the most recent output RF for visualization in the renderer.


There are two types of receptive fields:
scaled receptive field: the raw attention map, scaled to the grid size. Values are continuous and can be >1 if the
attention is stronger than any previously seen token.
binary receptive field: a binary mask of the same shape, where cells are 1 if they are currently the strongest
attended cell for that layer, and 0 otherwise.


"""

import torch

from src_new.memory.types import MemoryWindow

'''
def _compute_rf_all_envs(
        self,
        fovs: list,
        all_attention_weights: list[torch.Tensor],
        memory_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RF maps for all envs/layers for current tokens.

        Returns:
            rf_maps: (num_envs, num_layers, grid_w, grid_h)
        """
        #raise NotImplementedError("TODO.")
        num_envs = memory_indices.shape[0]
        num_layers = self.args.trxl_num_layers
        device = all_attention_weights[0].device

        fov_tensor = torch.stack(
            [torch.as_tensor(fov, dtype=torch.float32, device=device) for fov in fovs],
            dim=0,
        )
        rf_maps = torch.zeros(
            (num_envs, num_layers, self._internal_grid_w, self._internal_grid_h),
            dtype=torch.float32,
            device=device,
        )
        rf_maps[:, 0] = fov_tensor

        if num_layers == 1:
            return rf_maps

        for layer_idx in range(num_layers - 1):
            attn = all_attention_weights[layer_idx].mean(dim=1).squeeze(1)  # (N, M)
            for env_id in range(num_envs):
                window_prev_rf = self._internal_rf_all[
                    env_id, memory_indices[env_id], layer_idx
                ]  # (M, W, H)
                attended = torch.einsum("m,mwh->wh", attn[env_id], window_prev_rf)
                own_rf = rf_maps[env_id, layer_idx]
                rf_maps[env_id, layer_idx + 1] = 0.5 * own_rf + 0.5 * attended

        return rf_maps
'''


# TODO move to agent

# ========== Receptive field tracking (optional) ==========


# old version does not purely rely on attention and does automatic shit.
def compute_receptive_fields(
    memory_window: MemoryWindow,
    fov: torch.Tensor,
    attention_weights: list[torch.Tensor],
    envs_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute and store RF^(0..L) for the current timestep of env 0.

    Records the raw FOV as layer-0 RF, then propagates through transformer
    layers.

    Returns a binary and a attetion scaled receptive field for each layer.
    The scaled RF is the attention-weighted sum of the RFs of the tokens in the memory window, incorporating
    skip connection with the own RF.
    The binary RF is the union of the own RF and the RFs of all tokens in the memory window.

    IMPORTANT:
    input_rf_scaled/binary[0][i] shows the scaled receptive field of the !INPUT! layer to the i'th layer of env 0
        (and of output if i-1'th layer.).
        We do this because we also need to store the field of view, which is the rf of the input to the 0'th layer.
        The last (the L'th) rf of that tensor is the rf of the last (the (L-1)'th) layer.

    Args:
        memory_window: Provides the receptive fields amd memories in context window of the current
                timestep for all envs.
        fov: Boolean tensor (N, grid_w, grid_h) from ``get_visible_cells``.
        attention_weights: List of L tensors, each (N, H, 1, M) = (num_envs, num_attn_heads, 1, memory_length).
        envs_t: (N) tensor of current episode steps for each environment in the batch.
                Needed to determine the memory window corresponding to the attention weights of env 0.
    Return:
        input_rf_scaled: float tensor of shape (N, L+1, grid_w, grid_h).
        input_rf_binary: binary tensor of shape (N, L+1, grid_w, grid_h)

    """

    _validate_receptive_field_inputs(
        memory_window=memory_window, fov=fov, attention_weights=attention_weights, envs_t=envs_t
    )

    N = fov.shape[0]
    L = len(attention_weights)

    device = fov.device

    # --- Layer 0: raw FOV ---
    input_rf_scaled = torch.zeros(
        size=(N, L + 1, *fov[0].size()), dtype=torch.float32, device=device
    )
    input_rf_binary = torch.zeros(size=(N, L + 1, *fov[0].size()), dtype=torch.bool, device=device)
    input_rf_scaled[:, 0] = torch.as_tensor(fov, dtype=torch.float32, device=device)
    input_rf_binary[:, 0] = torch.as_tensor(fov, dtype=torch.bool, device=device)

    # RFs of the tokens currently in the memory window
    # tensors of shape (N, M, L, grid_w, grid_h)
    window_rfs = memory_window.receptive_fields_scaled
    window_rfs_bin = memory_window.receptive_fields_binary

    for layer_idx in range(L):
        # --- scaled RF: attention-weighted sum ---
        attn = attention_weights[layer_idx].mean(dim=1).squeeze(1)  # (N, M)
        attended_rf = torch.einsum("nm,nmwh->nwh", attn, window_rfs[:, :, layer_idx])
        # incorporate skip connection
        input_rf_scaled[:, layer_idx + 1] = 0.5 * input_rf_scaled[:, layer_idx] + 0.5 * attended_rf

        # --- binary RF: union of own RF with ALL memory tokens' RFs ---
        attended_token = memory_window.masks
        attended_bin = (window_rfs_bin[:, :, layer_idx] & attended_token[:, :, None, None]).any(
            dim=1
        )  # (N, W, H)

        input_rf_binary[:, layer_idx + 1] = input_rf_binary[:, layer_idx] | attended_bin

    return input_rf_scaled, input_rf_binary


def _validate_receptive_field_inputs(
    memory_window: MemoryWindow,
    fov: torch.Tensor,
    attention_weights: list[torch.Tensor],
    envs_t: torch.Tensor,
    require_attention_respects_mask: bool = False,
    atol: float = 1e-7,
) -> None:
    # TODO centralize this check

    # Core tensors
    memory_masks = memory_window.masks  # (N, M)
    memory_indices = memory_window.indices_env  # (N, M)
    window_rfs = memory_window.receptive_fields_scaled  # (N, M, L, W, H)
    window_rfs_bin = memory_window.receptive_fields_binary

    if memory_masks.ndim != 2 or memory_indices.ndim != 2:
        raise ValueError(
            f"memory_masks/memory_indices must be 2D, got {memory_masks.shape} and {memory_indices.shape}."
        )
    if memory_masks.shape != memory_indices.shape:
        raise ValueError(
            f"memory_masks/memory_indices shape mismatch: {memory_masks.shape} vs {memory_indices.shape}."
        )
    if memory_masks.dtype != torch.bool:
        raise ValueError(f"memory_masks must be bool, got {memory_masks.dtype}.")
    if memory_indices.dtype != torch.long:
        raise ValueError(f"memory_indices_env must be long, got {memory_indices.dtype}.")

    N, M = memory_masks.shape
    L = len(attention_weights)

    if envs_t.shape != (N,):
        raise ValueError(f"envs_t must have shape {(N,)}, got {tuple(envs_t.shape)}.")
    if envs_t.dtype != torch.long:
        raise ValueError(f"envs_t must be torch.long, got {envs_t.dtype}.")
    if (envs_t < 0).any():
        raise ValueError("envs_t must be non-negative.")

    if fov.ndim != 3 or fov.shape[0] != N:
        raise ValueError(f"fov must have shape (N, W, H), got {tuple(fov.shape)}.")

    # Window RF shape checks
    if window_rfs.ndim != 5:
        raise ValueError(
            f"receptive_fields_scaled must be 5D (N, M, L, W, H), got {tuple(window_rfs.shape)}."
        )
    if window_rfs.shape[0] != N or window_rfs.shape[1] != M:
        raise ValueError(
            f"receptive_fields_scaled incompatible with masks: {tuple(window_rfs.shape)} vs (N={N}, M={M})."
        )
    if window_rfs.shape[2] != L + 1:
        raise ValueError(
            f"Number of attention layers ({L}) + 1 must match RF layer axis ({window_rfs.shape[2]})."
        )

    if window_rfs_bin.shape != window_rfs.shape:
        raise ValueError(
            f"receptive_fields_binary shape mismatch: {tuple(window_rfs_bin.shape)} vs {tuple(window_rfs.shape)}."
        )
    if window_rfs_bin.dtype != torch.bool:
        raise ValueError(f"receptive_fields_binary must be bool, got {window_rfs_bin.dtype}.")

    # Temporal mask rule:
    # If mask is True for token k in row n, then index must be < envs_t[n].
    # Never allow index == t or > t in masked-in positions.
    valid_past = memory_indices < envs_t[:, None]
    illegal_mask_true = memory_masks & (~valid_past)
    if illegal_mask_true.any():
        bad_n, bad_k = illegal_mask_true.nonzero(as_tuple=True)
        n = int(bad_n[0].item())
        k = int(bad_k[0].item())
        idx = int(memory_indices[n, k].item())
        t = int(envs_t[n].item())
        raise ValueError(
            f"Invalid mask/index temporal relation at row={n}, token={k}: "
            f"mask=True but index={idx} is not strictly before t={t}."
        )

    # Attention shape/value checks
    for layer_idx, attn in enumerate(attention_weights):
        if attn.ndim != 4:
            raise ValueError(
                f"attention[{layer_idx}] must be 4D (N, H, 1, M), got {tuple(attn.shape)}."
            )
        if attn.shape[0] != N or attn.shape[2] != 1 or attn.shape[3] != M:
            raise ValueError(
                f"attention[{layer_idx}] shape {tuple(attn.shape)} incompatible with N={N}, M={M}."
            )
        if not torch.isfinite(attn).all():
            raise ValueError(f"attention[{layer_idx}] contains non-finite values.")

        # Optional strict coupling check (disable if you want to visualize mismatches)
        if require_attention_respects_mask:
            invalid_mass = (attn * (~memory_masks).unsqueeze(1).unsqueeze(1).to(attn.dtype)).sum(
                dim=3
            )
            if (invalid_mass > atol).any():
                raise ValueError(f"attention[{layer_idx}] places non-zero mass on masked tokens.")


'''
def compute_receptive_fields_env0(
        memory_handler: MemoryHandler,
        memory_mask_env0: torch.Tensor,
        fov_0: torch.Tensor,
        all_attention_weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and store RF^(0..L) for the current timestep of env 0.

        Records the raw FOV as layer-0 RF, then propagates through transformer
        layers.

        Returns a binary and a attetion scaled receptive field for each layer. 
        The scaled RF is the attention-weighted sum of the RFs of the tokens in the memory window, incorporating 
        skip connection with the own RF. 
        The binary RF is the union of the own RF and the RFs of all tokens in the memory window.

        IMPORTANT:
        input_rf_scaled/binary_env0[i] shows the scaled receptive field of the !INPUT! layer to the i'th layer 
            (and of output if i-1'th layer.). 
            We do this because we also need to store the field of view, which is the rf of the input to the 0'th layer.
            The last (the L'th) rf of that tensor is the rf of the last (the (L-1)'th) layer.

        Args:
            memory_handler: provides the receptive fields of memories in context window of this memory 
            memory_mask_env0: Tensor of shape (M). memory_mask_env0 specifies the memory window corresponding to the 
                attention weights of env 0. The values are the episode step indices of the tokens in the memory window.
            fov_0: Boolean array (grid_w, grid_h) from ``get_visible_cells``. shows the binary of the layer in the current 
            all_attention_weights: List of L tensors, each (N, H, 1, M) = (num_envs, num_attn_heads, 1, memory_length).

        Return:
            input_rf_scaled_env0: float tensor of shape (L+1, grid_w, grid_h).  
            input_rf_binary_env0: binary tensor of shape (L+1, grid_w, grid_h)
        
        """
        raise NotImplementedError("function depricated")
        assert memory_handler.trxl_num_layers == len(all_attention_weights), "I understood something seriously wrong"

        if memory_mask_env0.size() != torch.Size([memory_handler.max_episode_steps]):
            raise ValueError(f"Expected tensor size {torch.Size([memory_handler.max_episode_steps])} for memory_mask_env0,"
                f"got shape {memory_mask_env0.size()}.")
        
        L = memory_handler.trxl_num_layers

        # --- Layer 0: raw FOV ---
        input_rf_scaled_env0 = torch.zeros(size = (L+1, *fov_0.size()), dtype=torch.float32)
        input_rf_binary_env0 = torch.zeros(size = (L+1, *fov_0.size()), dtype=torch.bool)
        input_rf_scaled_env0[0] = torch.as_tensor(fov_0, dtype=torch.float32)
        input_rf_binary_env0[0] = torch.as_tensor(fov_0, dtype=torch.bool)

        # RFs of the tokens currently in the memory window for env 0
        # tensors of shape (M, L, grid_w, grid_h)
        window_rfs_env0, window_rfs_bin_env0 = memory_handler.get_rf_window_env0(memory_mask_env0)  
        
        for layer_idx in range(L):
            # --- scaled RF: attention-weighted sum ---
            attn_env0 = all_attention_weights[layer_idx][0].mean(dim=0).squeeze(0)  # (M,)
            attended_rf_env0 = torch.einsum("m,mwh->wh", attn_env0, window_rfs_env0[:, layer_idx])
            # incorporate skip connection
            input_rf_scaled_env0[layer_idx + 1] = 0.5 * input_rf_binary_env0[layer_idx] + 0.5 * attended_rf_env0 

            # --- binary RF: union of own RF with ALL memory tokens' RFs ---
            attended_bin_env0 = window_rfs_bin_env0[:, layer_idx].any(dim=0)  # (W, H)
            input_rf_binary_env0[layer_idx + 1] = input_rf_binary_env0[layer_idx] | attended_bin_env0

        return input_rf_scaled_env0, input_rf_binary_env0

def get_receptive_field_env0(self) -> tuple[np.ndarray, np.ndarray]:
    """Return the most recent output receptive field for env 0.

    Returns:
        Tuple of (scaled_rf, binary_rf) as numpy arrays of shape
        (grid_w, grid_h).  ``scaled_rf`` is float, ``binary_rf``
        is bool.
    """
    raise NotImplementedError("function depricated")
    if self._last_output_rf_scaled_env0 is None:
        w = self.memory_handler._rf_grid_w
        h = self.memory_handler._rf_grid_h
        return np.zeros((w, h), dtype=np.float32), np.zeros((w, h), dtype=bool)
    return (
        self._last_output_rf_scaled_env0.detach().cpu().numpy(),
        self._last_output_rf_binary_env0.detach().cpu().numpy(),
    )

    


'''
