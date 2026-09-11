import warnings

import numpy as np
import torch
from einops import rearrange
from torch import nn

from src_new.config import TrainConfig
from src_new.memory.types import MemoryWindow
from src_new.trainer.forward_diagnostics import ForwardDiagnostics


class PositionalEncoding(nn.Module):
    pos_emb: torch.Tensor

    def __init__(self, dim, max_seq_len, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        seq = torch.arange(max_seq_len - 1, -1, -1.0)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pos_emb[-seq_len:]


class GridCellEncoding(nn.Module):
    """Multi-scale grid cell positional encoding for 2D positions.

    Implements the "theory" encoder (PE^(t)) from:
        Mai et al., "Multi-Scale Representation Learning for Spatial Feature
        Distributions using Grid Cells" (Space2Vec), ICLR 2020.
        https://arxiv.org/abs/2003.00824

    For a 2D position x, S scales, and 3 projection directions, the encoding is
    (Eq. 3 of the paper):

        PE_{s,j}(x) = [sin(<x, a_j> * freq_s), cos(<x, a_j> * freq_s)]
                      for j = 1, 2, 3  and  s = 0, ..., S-1

    where freq_s are S frequencies log-spaced in [1/max_scale, 1/min_scale], and
    a_1, a_2, a_3 are unit vectors at angles 0, 2π/3, 4π/3 (separated by 120° as
    required by Theorem 1). The full embedding is the concatenation over all
    (j, s) pairs, yielding a vector of dimension S * 6 (3 dirs × S scales × 2).
    """

    freqs: torch.Tensor
    dirs: torch.Tensor

    def __init__(self, dim, min_scale=1.0, max_scale=20.0):
        super().__init__()
        self.actual_dim = dim
        self.used_dim = (dim // 6) * 6  # must be divisible by 6 (3 dirs × 2)
        n_scales = self.used_dim // 6  # S in the paper

        # S frequencies log-spaced from 1/λ_max to 1/λ_min (Eq. 3: freq_s = λ_min * g^(s/(S-1)))
        freqs = torch.logspace(np.log10(1.0 / max_scale), np.log10(1.0 / min_scale), n_scales)
        self.register_buffer("freqs", freqs)

        # Three unit-vector directions separated by 2π/3 (120°) as required by Theorem 1:
        # a_1 = (1, 0),  a_2 = (cos 2π/3, sin 2π/3),  a_3 = (cos 4π/3, sin 4π/3)
        angles = torch.tensor([0.0, 2 * np.pi / 3, 4 * np.pi / 3])  # Theorem 1
        dirs = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (3, 2)
        self.register_buffer("dirs", dirs)

    @property
    def n_scales(self) -> int:
        return self.used_dim // 6

    @property
    def n_angles(self) -> int:
        return 3

    @property
    def angle_labels(self) -> list[str]:
        return ["0°", "120°", "240°"]

    def get_index_for_angle_and_frequency(self, angle_index: int, frequency_index: int) -> int:
        """Return the flat index of the sin component for a given direction and scale.

        The reshape (N, 3, S, 2) → (N, 3·S·2) flattens in C order, so the layout is:
          [dir0_scale0_sin, dir0_scale0_cos, dir0_scale1_sin, ..., dir2_scaleS-1_cos]
        The cos component is always at the returned index + 1.
        """

        n_scales = self.used_dim // 6
        return angle_index * (n_scales * 2) + frequency_index * 2

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (N, 2) tensor of (x, y) positions to encode.

        Returns:
            (N, dim) position embedding.
        """

        N = pos.shape[0]

        pos = pos.float()  # ensure float for dot product and scaling

        # Eq. 3: dot product <x, a_j> for each of the 3 directions
        proj = pos @ self.dirs.T  # (N, 3)

        # Eq. 3: multiply by each frequency freq_s = 1 / (λ_min * g^(s/(S-1)))
        phases = proj.unsqueeze(-1) * self.freqs.view(1, 1, -1)  # (N, 3, S)

        # Eq. 3: PE_{s,j}(x) = [sin(<x, a_j> * freq_s), cos(<x, a_j> * freq_s)]
        emb = torch.stack([phases.sin(), phases.cos()], dim=-1)  # (N, 3, S, 2)
        emb = emb.reshape(N, self.used_dim)  # (N, 3·S·2)

        if self.used_dim < self.actual_dim:
            padding = torch.zeros(N, self.actual_dim - self.used_dim, device=emb.device)
            emb = torch.cat([emb, padding], dim=-1)  # (N, actual_dim)

        return emb


class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""

    def __init__(self, embed_dim, num_heads, uniform_attention_on_fully_masked_rows=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        # this reproduces bug in original code

        self.uniform_attention_on_fully_masked_rows = uniform_attention_on_fully_masked_rows

        assert self.head_size * num_heads == embed_dim, (
            "Embedding dimension needs to be divisible by the number of heads"
        )

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward_OLD(self, values, keys, query, mask):
        # was set to old because the case of mask being 0s not handled
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(
                mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")
            )  # -inf causes NaN

        # Normalize energy values and apply softmax to retrieve the attention scores
        attention = torch.softmax(
            energy / (self.embed_dim ** (1 / 2)), dim=3
        )  # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )

        return self.fc_out(out), attention

    def forward(self, values, keys, query, mask):
        # was set to old because the case of mask being 0s not handled
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # checking it is nan free, because now we fill it a bit with nans.
        assert not torch.isnan(energy).any(), "NaN values in energy before masking."

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(
                mask.unsqueeze(1).unsqueeze(1) == 0, float("-inf")
            )  # -inf causes NaN

        if self.uniform_attention_on_fully_masked_rows:
            # Handle the case where all entries in a row are masked (i.e., energy is -inf for all keys)
            # In this case, we set the energy to 0 for that row, which will result in uniform attention weights after softmax
            all_masked = mask.sum(dim=1) == 0  # shape: (N,)
            if all_masked.any():
                energy[all_masked] = float("-1e20")

        # Normalize energy values and apply softmax to retrieve the attention scores
        attention = torch.softmax(
            energy / (self.embed_dim ** (1 / 2)), dim=3
        )  # attention shape: (N, heads, query_len, key_len)

        attention = torch.nan_to_num(attention, nan=0.0)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )

        return self.fc_out(out), attention


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, uniform_attention_on_fully_masked_rows=True):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim,
            num_heads,
            uniform_attention_on_fully_masked_rows=uniform_attention_on_fully_masked_rows,
        )
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        # Pre-layer normalization (post-layer normalization is usually less effective)
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value  # shared pre-projection representation
        attention, attention_weights = self.attention(value, key, query_, mask)  # MHA
        x = attention + query  # Skip connection
        x_ = self.layer_norm_attn(x)  # Pre-layer normalization
        forward = self.fc_projection(x_)  # Forward projection
        out = forward + x  # Skip connection
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(
        self,
        tc: TrainConfig,
        max_episode_steps,
    ):
        super().__init__()

        self.dim = tc.trxl_dim
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = tc.trxl_positional_encoding
        self.prior_pe = tc.prior_pe
        self.grid_cell_encoding = tc.grid_cell_encoding
        self.grid_cell_encoding_weight = tc.grid_cell_encoding_weight

        if self.grid_cell_encoding:
            # TODO
            warnings.warn(
                "outstanding todo: grid scales should be adapted to the environment size. Currently set to [1.0, 20.0]."
            )
            self.grid_cell_encoder = GridCellEncoding(self.dim, min_scale=1.0, max_scale=20.0)

        if self.positional_encoding == "absolute":
            warnings.warn(
                "Currently the timescale of encoder is not adapted to the max_episode_steps, which may lead to suboptimal performance. Consider using 'learned' positional encoding or implementing adaptive timescales for the sinusoidal encoder."
            )
            self.pos_embedding_a = PositionalEncoding(self.dim, max_episode_steps + 1)
        elif self.positional_encoding == "learned":
            self.pos_embedding_l = nn.Parameter(torch.randn(max_episode_steps, self.dim))
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.dim,
                    tc.trxl_num_heads,
                    uniform_attention_on_fully_masked_rows=tc.uniform_attention_on_fully_masked_rows,
                )
                for _ in range(tc.trxl_num_layers)
            ]
        )

    def encode_position(self, pos: torch.Tensor) -> torch.Tensor:
        """Encode 2D positions using the grid cell encoder.

        Args:
            pos: (N, 2) tensor of (x, y) positions.
        Returns:
            (N, dim) position embedding.
        """
        if not self.grid_cell_encoding:
            raise RuntimeError("grid_cell_encoding is not enabled on this Transformer.")
        return self.grid_cell_encoder(pos)

    def forward(
        self,
        x: torch.Tensor,
        envs_t: torch.Tensor,
        memory_window: MemoryWindow,
        perceived_pos: torch.Tensor,
        forward_diagnostics: ForwardDiagnostics | None = None,
    ):
        """
        Args:
            x: (N, dim) tensor of the current transformer layer input (i.e. current observation encoding).
            memory_window: MemoryWindow object containing the retrieval window for the minibatch items.
        """

        memory_frames = memory_window.frames
        memory_mask = memory_window.masks
        memory_indices = memory_window.indices_env

        # check tensor shapes
        if memory_frames.ndim != 4:
            raise ValueError(
                f"Expected memory_frames to be 4D (N, M, L, D), got {memory_frames.shape}."
            )
        if x.shape != (memory_frames.shape[0], self.dim):
            raise ValueError(
                f"Expected x to have shape {(memory_frames.shape[0], self.dim)}, but got {x.shape}"
            )
        if memory_mask.shape != memory_indices.shape:
            raise ValueError(
                f"memory_mask and memory_indices must have the same shape, got {memory_mask.shape} and {memory_indices.shape}."
            )
        if (
            memory_mask.shape[0] != memory_frames.shape[0]
            or memory_mask.shape[1] != memory_frames.shape[1]
        ):
            raise ValueError(
                f"Expected memory_mask shape {(memory_frames.shape[0], memory_frames.shape[1])}, got {memory_mask.shape}."
            )

        if memory_indices.max().item() >= self.max_episode_steps:
            raise ValueError(
                f"Memory index exceeds max episode steps."
                f"Got max memory index {memory_indices.max().item()}, "
                f"but max episode steps is {self.max_episode_steps}"
            )

        # Add positional encoding to memory frames
        pe_current_token = torch.zeros_like(x)
        pe_memories = torch.zeros_like(memory_frames)

        grid_cell_pe = None
        if self.grid_cell_encoding:
            if perceived_pos is None:
                raise ValueError(
                    "perceived_pos must be provided when grid_cell_encoding is enabled."
                )
            grid_cell_pe = self.encode_position(perceived_pos)

        if self.positional_encoding == "absolute":
            if self.prior_pe:
                # Only add PE to the to be generated token, not to the memory, frames as in this scenario
                # they already have PE added when they were generated and stored in memory.
                # +1 because the current token can be the last one
                pe_current_token = self.pos_embedding_a(self.max_episode_steps + 1)[envs_t]
            else:
                pe_memories = self.pos_embedding_a(self.max_episode_steps)[
                    memory_indices
                ].unsqueeze(2)

        elif self.positional_encoding == "learned":
            if self.prior_pe:
                pe_current_token = self.pos_embedding_l[envs_t]
            else:
                pe_memories = self.pos_embedding_l[memory_indices].unsqueeze(2)
        elif self.positional_encoding == "none":
            # pe_current_token and pe_memories are already zero
            pass
        else:
            raise ValueError(
                f"Invalid positional encoding type: {self.positional_encoding}. "
                f"Expected 'absolute', 'learned', or 'none'."
            )

        if grid_cell_pe is not None:
            t_wheight = 1 - self.grid_cell_encoding_weight
            pe_current_token = (
                t_wheight * pe_current_token + self.grid_cell_encoding_weight * grid_cell_pe
            )

        x = x + pe_current_token
        memory_frames = memory_frames + pe_memories

        assert not torch.isnan(memory_frames).any(), (
            "NaN values in memory_frames after positional encoding."
        )
        assert not torch.isnan(x).any(), (
            f"NaN values in x after positional encoding. Envs t: {envs_t}. "
            f"Has nan in pos embedding: {torch.isnan(pe_current_token).any()}"
        )

        # Forward transformer layers and return new memories (i.e. hidden states)
        out_memories = []
        all_attention_weights = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, attention_weights = layer(
                memory_frames[:, :, i], memory_frames[:, :, i], x.unsqueeze(1), memory_mask
            )  # args: value, key, query, mask
            all_attention_weights.append(attention_weights.detach())  # (N, H, 1, M)
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        out_memories_st = torch.stack(out_memories, dim=1)

        ##
        # pack forward diagnostics
        ##
        if forward_diagnostics is not None:
            # L+1, D tensor
            forward_diagnostics.layer_activations = torch.cat(
                [out_memories_st[0], x[0].detach().unsqueeze(0)], dim=0
            )

            forward_diagnostics.pe_current_token = pe_current_token[0].detach()  # (D,)

        return x, out_memories_st, all_attention_weights
