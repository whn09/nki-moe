"""
NKI Kernels for Attention Operations

This module contains NKI kernels for attention-related computations:
- Rotary Position Embedding (RoPE)
- QK Normalization (RMSNorm on Q and K)
- Fused QKV Projection
- Attention Score Computation
"""

import math
import torch
import torch.nn as nn
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


# ============================================================================
# Rotary Position Embedding (RoPE) Kernel
# ============================================================================

@nki.jit
def nki_rope_kernel(
    q_tensor,      # [batch, seq_len, num_heads, head_dim]
    k_tensor,      # [batch, seq_len, num_kv_heads, head_dim]
    cos_cache,     # [max_seq_len, head_dim]
    sin_cache,     # [max_seq_len, head_dim]
    seq_offset=0   # Starting position for KV cache scenarios
):
    """
    NKI kernel for Rotary Position Embedding (RoPE).

    Applies rotary embeddings to query and key tensors:
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    where rotate_half([x1, x2]) = [-x2, x1]

    Args:
        q_tensor: Query tensor [batch, seq_len, num_heads, head_dim]
        k_tensor: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        cos_cache: Precomputed cosines [max_seq_len, head_dim]
        sin_cache: Precomputed sines [max_seq_len, head_dim]
        seq_offset: Position offset for incremental decoding

    Returns:
        q_rotated: Rotated query tensor
        k_rotated: Rotated key tensor
    """
    batch = q_tensor.shape[0]
    seq_len = q_tensor.shape[1]
    num_heads = q_tensor.shape[2]
    head_dim = q_tensor.shape[3]
    num_kv_heads = k_tensor.shape[2]

    q_out = nl.ndarray(q_tensor.shape, dtype=q_tensor.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray(k_tensor.shape, dtype=k_tensor.dtype, buffer=nl.shared_hbm)

    half_dim = head_dim // 2

    # Fixed tile size (must be compile-time constant)
    TILE_S = 128

    # Index templates
    is_idx = nl.arange(TILE_S)[:, None]
    i_half_first = nl.arange(half_dim)[None, :]
    i_half_second = half_dim + nl.arange(half_dim)[None, :]

    for b in nl.affine_range(batch):
        for s_tile in nl.affine_range(math.ceil(seq_len / TILE_S)):
            s_start = s_tile * TILE_S

            # Position indices for cos/sin lookup
            pos_indices = seq_offset + s_start + is_idx

            # Load cos and sin values for this tile [TILE_S, half_dim]
            cos_first = nl.load(cos_cache[pos_indices, i_half_first],
                               mask=(s_start + is_idx < seq_len))
            cos_second = nl.load(cos_cache[pos_indices, i_half_second],
                                mask=(s_start + is_idx < seq_len))
            sin_first = nl.load(sin_cache[pos_indices, i_half_first],
                               mask=(s_start + is_idx < seq_len))
            sin_second = nl.load(sin_cache[pos_indices, i_half_second],
                                mask=(s_start + is_idx < seq_len))

            # Process query heads
            for h in nl.affine_range(num_heads):
                # Load Q values [TILE_S, half_dim]
                q_first = nl.load(q_tensor[b, s_start + is_idx, h, i_half_first],
                                 mask=(s_start + is_idx < seq_len))
                q_second = nl.load(q_tensor[b, s_start + is_idx, h, i_half_second],
                                  mask=(s_start + is_idx < seq_len))

                # Apply rotation:
                # new_first = q_first * cos_first - q_second * sin_first
                # new_second = q_second * cos_second + q_first * sin_second
                q_rot_first = nl.subtract(
                    nl.multiply(q_first, cos_first),
                    nl.multiply(q_second, sin_first)
                )
                q_rot_second = nl.add(
                    nl.multiply(q_second, cos_second),
                    nl.multiply(q_first, sin_second)
                )

                # Store results
                nl.store(q_out[b, s_start + is_idx, h, i_half_first],
                        value=q_rot_first, mask=(s_start + is_idx < seq_len))
                nl.store(q_out[b, s_start + is_idx, h, i_half_second],
                        value=q_rot_second, mask=(s_start + is_idx < seq_len))

            # Process key heads (same pattern)
            for h in nl.affine_range(num_kv_heads):
                k_first = nl.load(k_tensor[b, s_start + is_idx, h, i_half_first],
                                 mask=(s_start + is_idx < seq_len))
                k_second = nl.load(k_tensor[b, s_start + is_idx, h, i_half_second],
                                  mask=(s_start + is_idx < seq_len))

                k_rot_first = nl.subtract(
                    nl.multiply(k_first, cos_first),
                    nl.multiply(k_second, sin_first)
                )
                k_rot_second = nl.add(
                    nl.multiply(k_second, cos_second),
                    nl.multiply(k_first, sin_second)
                )

                nl.store(k_out[b, s_start + is_idx, h, i_half_first],
                        value=k_rot_first, mask=(s_start + is_idx < seq_len))
                nl.store(k_out[b, s_start + is_idx, h, i_half_second],
                        value=k_rot_second, mask=(s_start + is_idx < seq_len))

    return q_out, k_out


# ============================================================================
# Fused QKV Projection Kernel
# ============================================================================

@nki.jit
def nki_fused_qkv_proj_kernel(
    hidden_states,  # [batch, seq_len, hidden_size]
    qkv_weight,     # [hidden_size, q_size + k_size + v_size]
    q_size,
    k_size,
    v_size
):
    """
    Fused QKV projection kernel.

    Computes Q, K, V projections in a single fused operation.

    Args:
        hidden_states: Input [batch, seq_len, hidden_size]
        qkv_weight: Fused QKV weights [hidden_size, q+k+v]
        q_size: Query projection output size
        k_size: Key projection output size
        v_size: Value projection output size

    Returns:
        q: Query projection
        k: Key projection
        v: Value projection
    """
    batch = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    total_size = q_size + k_size + v_size

    q_out = nl.ndarray((batch, seq_len, q_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((batch, seq_len, k_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    v_out = nl.ndarray((batch, seq_len, v_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    TILE_M = 128
    TILE_K = 128
    TILE_N = 128

    for b in nl.affine_range(batch):
        for s_tile in nl.affine_range(math.ceil(seq_len / TILE_M)):
            s_start = s_tile * TILE_M
            s_len = min(TILE_M, seq_len - s_start)

            ix = nl.arange(s_len)[:, None]

            # Initialize accumulators
            q_acc = nl.zeros((s_len, q_size), dtype=nl.float32)
            k_acc = nl.zeros((s_len, k_size), dtype=nl.float32)
            v_acc = nl.zeros((s_len, v_size), dtype=nl.float32)

            # Tiled matrix multiplication over hidden dimension
            for k_tile in nl.affine_range(math.ceil(hidden_size / TILE_K)):
                k_start = k_tile * TILE_K
                k_len = min(TILE_K, hidden_size - k_start)

                ik = nl.arange(k_len)[None, :]
                ik_col = nl.arange(k_len)[:, None]

                # Load hidden states tile
                h_tile = nl.load(
                    hidden_states[b, s_start + ix, k_start + ik],
                    mask=(s_start + ix < seq_len) & (k_start + ik < hidden_size)
                )

                # Load and compute Q projection
                for n_tile in nl.affine_range(math.ceil(q_size / TILE_N)):
                    n_start = n_tile * TILE_N
                    n_len = min(TILE_N, q_size - n_start)
                    in_idx = nl.arange(n_len)[None, :]

                    w_tile = nl.load(
                        qkv_weight[k_start + ik_col, n_start + in_idx],
                        mask=(k_start + ik_col < hidden_size) & (n_start + in_idx < q_size)
                    )
                    q_partial = nl.matmul(h_tile, w_tile)
                    q_acc[:, n_start:n_start+n_len] = nl.add(
                        q_acc[:, n_start:n_start+n_len],
                        q_partial[:, :n_len]
                    )

                # Load and compute K projection
                for n_tile in nl.affine_range(math.ceil(k_size / TILE_N)):
                    n_start = n_tile * TILE_N
                    n_len = min(TILE_N, k_size - n_start)
                    in_idx = nl.arange(n_len)[None, :]

                    w_tile = nl.load(
                        qkv_weight[k_start + ik_col, q_size + n_start + in_idx],
                        mask=(k_start + ik_col < hidden_size) & (n_start + in_idx < k_size)
                    )
                    k_partial = nl.matmul(h_tile, w_tile)
                    k_acc[:, n_start:n_start+n_len] = nl.add(
                        k_acc[:, n_start:n_start+n_len],
                        k_partial[:, :n_len]
                    )

                # Load and compute V projection
                for n_tile in nl.affine_range(math.ceil(v_size / TILE_N)):
                    n_start = n_tile * TILE_N
                    n_len = min(TILE_N, v_size - n_start)
                    in_idx = nl.arange(n_len)[None, :]

                    w_tile = nl.load(
                        qkv_weight[k_start + ik_col, q_size + k_size + n_start + in_idx],
                        mask=(k_start + ik_col < hidden_size) & (n_start + in_idx < v_size)
                    )
                    v_partial = nl.matmul(h_tile, w_tile)
                    v_acc[:, n_start:n_start+n_len] = nl.add(
                        v_acc[:, n_start:n_start+n_len],
                        v_partial[:, :n_len]
                    )

            # Cast and store results
            iq = nl.arange(q_size)[None, :]
            ik_out = nl.arange(k_size)[None, :]
            iv = nl.arange(v_size)[None, :]

            nl.store(q_out[b, s_start + ix, iq], value=q_acc.astype(hidden_states.dtype))
            nl.store(k_out[b, s_start + ix, ik_out], value=k_acc.astype(hidden_states.dtype))
            nl.store(v_out[b, s_start + ix, iv], value=v_acc.astype(hidden_states.dtype))

    return q_out, k_out, v_out


# ============================================================================
# QK LayerNorm (RMSNorm for Q and K)
# ============================================================================

@nki.jit
def nki_qk_rmsnorm_kernel(q_tensor, k_tensor, q_weight, k_weight, eps):
    """
    NKI kernel for RMSNorm on Q and K tensors.

    Used in Qwen3 attention for query/key normalization.

    Args:
        q_tensor: Query tensor [batch, seq_len, num_heads, head_dim]
        k_tensor: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        q_weight: Q norm weight [head_dim]
        k_weight: K norm weight [head_dim]
        eps: Epsilon for numerical stability

    Returns:
        q_normed: Normalized query tensor
        k_normed: Normalized key tensor
    """
    batch = q_tensor.shape[0]
    seq_len = q_tensor.shape[1]
    num_heads = q_tensor.shape[2]
    head_dim = q_tensor.shape[3]
    num_kv_heads = k_tensor.shape[2]

    q_out = nl.ndarray(q_tensor.shape, dtype=q_tensor.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray(k_tensor.shape, dtype=k_tensor.dtype, buffer=nl.shared_hbm)

    # Load weights once
    iw = nl.arange(1)[:, None]
    id_idx = nl.arange(head_dim)[None, :]
    q_w = nl.load(q_weight.reshape((1, head_dim))[iw, id_idx])
    k_w = nl.load(k_weight.reshape((1, head_dim))[iw, id_idx])

    TILE_M = 128

    for b in nl.affine_range(batch):
        for s_tile in nl.affine_range(math.ceil(seq_len / TILE_M)):
            s_start = s_tile * TILE_M
            s_end = min(s_start + TILE_M, seq_len)
            tile_size = s_end - s_start

            is_idx = nl.arange(tile_size)[:, None]

            # Process Q heads
            for h in nl.affine_range(num_heads):
                q_tile = nl.load(q_tensor[b, s_start:s_end, h, :head_dim])

                # RMSNorm
                q_sq = nl.square(q_tile)
                q_mean = nl.sum(q_sq, axis=[1]) / head_dim
                q_rms = nl.rsqrt(q_mean + eps)
                q_norm = nl.multiply(q_tile, q_rms)

                # Apply weight
                q_w_bcast = q_w.broadcast_to((tile_size, head_dim))
                q_final = nl.multiply(q_norm, q_w_bcast)

                nl.store(q_out[b, s_start:s_end, h, :head_dim], value=q_final)

            # Process K heads
            for h in nl.affine_range(num_kv_heads):
                k_tile = nl.load(k_tensor[b, s_start:s_end, h, :head_dim])

                # RMSNorm
                k_sq = nl.square(k_tile)
                k_mean = nl.sum(k_sq, axis=[1]) / head_dim
                k_rms = nl.rsqrt(k_mean + eps)
                k_norm = nl.multiply(k_tile, k_rms)

                # Apply weight
                k_w_bcast = k_w.broadcast_to((tile_size, head_dim))
                k_final = nl.multiply(k_norm, k_w_bcast)

                nl.store(k_out[b, s_start:s_end, h, :head_dim], value=k_final)

    return q_out, k_out


# ============================================================================
# Fused Attention Score Kernel (Q @ K^T)
# ============================================================================

@nki.jit
def nki_attention_scores_kernel(
    query,   # [batch, num_heads, seq_len_q, head_dim]
    key,     # [batch, num_kv_heads, seq_len_k, head_dim]
    scale    # Attention scaling factor
):
    """
    NKI kernel for computing attention scores.

    Computes: scores = (Q @ K^T) * scale

    Args:
        query: Query tensor [batch, num_heads, seq_len_q, head_dim]
        key: Key tensor [batch, num_kv_heads, seq_len_k, head_dim]
        scale: Scaling factor (typically 1/sqrt(head_dim))

    Returns:
        scores: Attention scores [batch, num_heads, seq_len_q, seq_len_k]
    """
    batch = query.shape[0]
    num_heads = query.shape[1]
    seq_len_q = query.shape[2]
    head_dim = query.shape[3]
    num_kv_heads = key.shape[1]
    seq_len_k = key.shape[2]

    # GQA: num_heads may be multiple of num_kv_heads
    kv_group_size = num_heads // num_kv_heads

    scores = nl.ndarray(
        (batch, num_heads, seq_len_q, seq_len_k),
        dtype=query.dtype,
        buffer=nl.shared_hbm
    )

    TILE_Q = 128
    TILE_K = 128
    TILE_D = 64

    for b in nl.affine_range(batch):
        for h in nl.affine_range(num_heads):
            kv_head = h // kv_group_size

            for q_tile in nl.affine_range(math.ceil(seq_len_q / TILE_Q)):
                q_start = q_tile * TILE_Q
                q_len = min(TILE_Q, seq_len_q - q_start)

                for k_tile in nl.affine_range(math.ceil(seq_len_k / TILE_K)):
                    k_start = k_tile * TILE_K
                    k_len = min(TILE_K, seq_len_k - k_start)

                    # Accumulate over head_dim
                    score_acc = nl.zeros((q_len, k_len), dtype=nl.float32)

                    for d_tile in nl.affine_range(math.ceil(head_dim / TILE_D)):
                        d_start = d_tile * TILE_D
                        d_len = min(TILE_D, head_dim - d_start)

                        iq = nl.arange(q_len)[:, None]
                        id_q = nl.arange(d_len)[None, :]

                        # Load Q tile [q_len, d_len]
                        q_tile_data = nl.load(
                            query[b, h, q_start + iq, d_start + id_q],
                            mask=(q_start + iq < seq_len_q) & (d_start + id_q < head_dim)
                        )

                        # Load K tile and transpose [d_len, k_len]
                        ik = nl.arange(k_len)[None, :]
                        id_k = nl.arange(d_len)[:, None]
                        k_tile_data = nl.load(
                            key[b, kv_head, k_start + ik, d_start + id_k.T],
                            mask=(k_start + ik < seq_len_k) & (d_start + id_k.T < head_dim)
                        )
                        k_tile_t = k_tile_data.T  # [d_len, k_len]

                        # Accumulate Q @ K^T
                        score_partial = nl.matmul(q_tile_data, k_tile_t)
                        score_acc = nl.add(score_acc, score_partial)

                    # Apply scale
                    score_scaled = nl.multiply(score_acc, scale)

                    # Store
                    iq_out = nl.arange(q_len)[:, None]
                    ik_out = nl.arange(k_len)[None, :]
                    nl.store(
                        scores[b, h, q_start + iq_out, k_start + ik_out],
                        value=score_scaled.astype(query.dtype),
                        mask=(q_start + iq_out < seq_len_q) & (k_start + ik_out < seq_len_k)
                    )

    return scores


# ============================================================================
# PyTorch Module Wrappers
# ============================================================================

class NKIRoPE(nn.Module):
    """NKI-accelerated Rotary Position Embedding."""

    def __init__(self, head_dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute cos/sin caches
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cos/sin cache
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, q, k, position_ids=None, seq_offset=0):
        if position_ids is not None:
            seq_offset = position_ids[0, 0].item()

        q_rot, k_rot = nki_rope_kernel(
            q, k,
            self.cos_cache,
            self.sin_cache,
            seq_offset
        )
        return q_rot, k_rot


class NKIQKNorm(nn.Module):
    """NKI-accelerated Q/K RMSNorm for Qwen3 attention."""

    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.q_weight = nn.Parameter(torch.ones(head_dim))
        self.k_weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, q, k):
        return nki_qk_rmsnorm_kernel(q, k, self.q_weight, self.k_weight, self.eps)


class NKIFusedQKVProj(nn.Module):
    """NKI-accelerated fused QKV projection."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_size = num_heads * head_dim
        self.k_size = num_kv_heads * head_dim
        self.v_size = num_kv_heads * head_dim

        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_size + self.k_size + self.v_size,
            bias=False
        )

    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape

        q, k, v = nki_fused_qkv_proj_kernel(
            hidden_states,
            self.qkv_proj.weight.T,
            self.q_size,
            self.k_size,
            self.v_size
        )

        # Reshape for attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        return q, k, v


# ============================================================================
# Export Functions
# ============================================================================

def get_nki_rope(head_dim, max_position_embeddings=2048, base=10000.0):
    """Factory function for NKI RoPE."""
    return NKIRoPE(head_dim, max_position_embeddings, base)


def get_nki_qk_norm(head_dim, eps=1e-6):
    """Factory function for NKI QK Norm."""
    return NKIQKNorm(head_dim, eps)


def get_nki_fused_qkv(hidden_size, num_heads, num_kv_heads, head_dim):
    """Factory function for NKI Fused QKV projection."""
    return NKIFusedQKVProj(hidden_size, num_heads, num_kv_heads, head_dim)
