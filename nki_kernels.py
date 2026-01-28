"""
NKI Kernels for Qwen3-30B-A3B MoE Model Optimization

This module contains custom NKI kernels for optimizing the Qwen3 MoE model
on AWS Trainium2/3 hardware. Key optimizations include:
- SiLU activation kernel
- Softmax kernel for expert routing
- Fused SwiGLU kernel for MoE expert computation
- RoPE (Rotary Position Embedding) kernel
"""

import math
import torch
import torch.nn as nn
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from torch_neuronx.xla_impl.ops import nki_jit


# ============================================================================
# SiLU (Swish) Activation Kernel
# silu(x) = x * sigmoid(x)
# ============================================================================

@nki.jit
def nki_silu_kernel(input_tensor):
    """
    NKI kernel for SiLU (Swish) activation: silu(x) = x * sigmoid(x)

    Args:
        input_tensor: Input tensor [batch*seq_len, hidden_size]

    Returns:
        output: SiLU-activated tensor with same shape as input
    """
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    num_rows = input_tensor.shape[0]
    num_cols = input_tensor.shape[1]

    # Process 128 rows at a time (NeuronCore partition constraint)
    TILE_M = 128
    TILE_N = min(512, num_cols)  # Process columns in tiles

    ix = nl.arange(TILE_M)[:, None]

    for i_row in nl.affine_range(math.ceil(num_rows / TILE_M)):
        for i_col in nl.affine_range(math.ceil(num_cols / TILE_N)):
            iy = i_col * TILE_N + nl.arange(TILE_N)[None, :]

            # Load input tile
            x_tile = nl.load(
                input_tensor[i_row * TILE_M + ix, iy],
                mask=(i_row * TILE_M + ix < num_rows) & (iy < num_cols)
            )

            # Compute sigmoid(x) = 1 / (1 + exp(-x))
            neg_x = nl.negative(x_tile)
            exp_neg_x = nl.exp(neg_x)
            one_plus_exp = nl.add(exp_neg_x, 1.0)
            sigmoid_x = nl.reciprocal(one_plus_exp)

            # SiLU = x * sigmoid(x)
            silu_out = nl.multiply(x_tile, sigmoid_x)

            # Store result
            nl.store(
                output[i_row * TILE_M + ix, iy],
                value=silu_out,
                mask=(i_row * TILE_M + ix < num_rows) & (iy < num_cols)
            )

    return output


# ============================================================================
# Softmax Kernel for Expert Routing
# ============================================================================

@nki.jit
def nki_softmax_kernel(input_tensor):
    """
    NKI kernel for softmax activation used in MoE routing.
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        input_tensor: Input tensor [batch*seq_len, num_experts]

    Returns:
        output: Softmax probabilities with same shape as input
    """
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    num_rows = input_tensor.shape[0]
    num_cols = input_tensor.shape[1]  # num_experts = 128

    TILE_M = 128
    ix = nl.arange(TILE_M)[:, None]
    iy = nl.arange(num_cols)[None, :]

    for i_row in nl.affine_range(math.ceil(num_rows / TILE_M)):
        # Load input tile
        x_tile = nl.load(
            input_tensor[i_row * TILE_M + ix, iy],
            mask=(i_row * TILE_M + ix < num_rows)
        )

        # Compute max for numerical stability
        x_max = nl.max(x_tile, axis=[1])

        # Subtract max and compute exp
        x_centered = nl.subtract(x_tile, x_max)
        exp_x = nl.exp(x_centered)

        # Compute sum of exponentials
        exp_sum = nl.sum(exp_x, axis=[1])

        # Compute softmax = exp(x - max) / sum(exp)
        softmax_out = nl.divide(exp_x, exp_sum)

        # Store result
        nl.store(
            output[i_row * TILE_M + ix, iy],
            value=softmax_out,
            mask=(i_row * TILE_M + ix < num_rows)
        )

    return output


# ============================================================================
# TopK Selection Kernel for Expert Routing
# ============================================================================

@nki.jit
def nki_topk_kernel(input_tensor, k=8):
    """
    NKI kernel for selecting top-k experts.

    Args:
        input_tensor: Router logits [batch*seq_len, num_experts]
        k: Number of experts to select (default: 8)

    Returns:
        topk_values: Top-k values [batch*seq_len, k]
        topk_indices: Top-k indices [batch*seq_len, k]
    """
    num_rows = input_tensor.shape[0]
    num_experts = input_tensor.shape[1]

    topk_values = nl.ndarray((num_rows, k), dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray((num_rows, k), dtype=nl.int32, buffer=nl.shared_hbm)

    TILE_M = 128
    ix = nl.arange(TILE_M)[:, None]
    iy = nl.arange(num_experts)[None, :]
    ik = nl.arange(k)[None, :]

    for i_row in nl.affine_range(math.ceil(num_rows / TILE_M)):
        # Load input tile
        x_tile = nl.load(
            input_tensor[i_row * TILE_M + ix, iy],
            mask=(i_row * TILE_M + ix < num_rows)
        )

        # Initialize output tiles
        out_vals = nl.zeros((TILE_M, k), dtype=input_tensor.dtype)
        out_idx = nl.zeros((TILE_M, k), dtype=nl.int32)

        # Simple iterative top-k selection
        x_work = x_tile
        for j in range(k):
            # Find max value and index
            max_val = nl.max(x_work, axis=[1])

            # Store top value
            out_vals[:, j:j+1] = max_val

            # Mask out the selected value for next iteration
            is_max = nl.equal(x_work, max_val)
            x_work = nl.where(is_max, nl.full_like(x_work, float('-inf')), x_work)

        # Store results
        nl.store(
            topk_values[i_row * TILE_M + ix, ik],
            value=out_vals,
            mask=(i_row * TILE_M + ix < num_rows)
        )

    return topk_values, topk_indices


# ============================================================================
# Fused SwiGLU Kernel (Core MoE Expert Computation)
# SwiGLU(x, W_gate, W_up) = SiLU(x @ W_gate) * (x @ W_up)
# ============================================================================

@nki.jit
def nki_swiglu_kernel(input_tensor, gate_weight, up_weight):
    """
    Fused SwiGLU kernel for MoE expert computation.

    Computes: SwiGLU(x) = SiLU(x @ gate_weight) * (x @ up_weight)

    This fuses:
    1. gate_proj = x @ gate_weight
    2. up_proj = x @ up_weight
    3. silu_gate = silu(gate_proj)
    4. output = silu_gate * up_proj

    Args:
        input_tensor: Input [batch*seq_len, hidden_size]
        gate_weight: Gate projection weight [hidden_size, intermediate_size]
        up_weight: Up projection weight [hidden_size, intermediate_size]

    Returns:
        output: SwiGLU output [batch*seq_len, intermediate_size]
    """
    batch_size = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    intermediate_size = gate_weight.shape[1]

    output = nl.ndarray((batch_size, intermediate_size), dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    # Tile sizes for efficient memory access
    TILE_M = 128  # Batch/sequence tile
    TILE_K = 128  # Hidden dimension tile
    TILE_N = 128  # Intermediate dimension tile

    for i_m in nl.affine_range(math.ceil(batch_size / TILE_M)):
        for i_n in nl.affine_range(math.ceil(intermediate_size / TILE_N)):
            # Initialize accumulators for gate and up projections
            gate_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32)
            up_acc = nl.zeros((TILE_M, TILE_N), dtype=nl.float32)

            ix = nl.arange(TILE_M)[:, None]
            in_idx = i_n * TILE_N + nl.arange(TILE_N)[None, :]

            # Accumulate matrix multiplication over K dimension
            for i_k in nl.affine_range(math.ceil(hidden_size / TILE_K)):
                ik = i_k * TILE_K + nl.arange(TILE_K)[None, :]
                ik_col = i_k * TILE_K + nl.arange(TILE_K)[:, None]

                # Load input tile [TILE_M, TILE_K]
                x_tile = nl.load(
                    input_tensor[i_m * TILE_M + ix, ik],
                    mask=(i_m * TILE_M + ix < batch_size) & (ik < hidden_size)
                )

                # Load gate weight tile [TILE_K, TILE_N]
                gate_tile = nl.load(
                    gate_weight[ik_col, in_idx],
                    mask=(ik_col < hidden_size) & (in_idx < intermediate_size)
                )

                # Load up weight tile [TILE_K, TILE_N]
                up_tile = nl.load(
                    up_weight[ik_col, in_idx],
                    mask=(ik_col < hidden_size) & (in_idx < intermediate_size)
                )

                # Accumulate: gate_proj += x @ gate_weight
                gate_acc = nl.add(gate_acc, nl.matmul(x_tile, gate_tile))

                # Accumulate: up_proj += x @ up_weight
                up_acc = nl.add(up_acc, nl.matmul(x_tile, up_tile))

            # Apply SiLU to gate projection
            neg_gate = nl.negative(gate_acc)
            exp_neg = nl.exp(neg_gate)
            sigmoid_gate = nl.reciprocal(nl.add(exp_neg, 1.0))
            silu_gate = nl.multiply(gate_acc, sigmoid_gate)

            # Final output: silu(gate) * up
            out_tile = nl.multiply(silu_gate, up_acc)

            # Cast back to input dtype if needed
            out_tile = out_tile.astype(input_tensor.dtype)

            # Store result
            nl.store(
                output[i_m * TILE_M + ix, in_idx],
                value=out_tile,
                mask=(i_m * TILE_M + ix < batch_size) & (in_idx < intermediate_size)
            )

    return output


# ============================================================================
# RoPE (Rotary Position Embedding) Kernel
# ============================================================================

@nki.jit
def nki_rope_kernel(q_tensor, k_tensor, cos_cache, sin_cache, position_ids):
    """
    NKI kernel for Rotary Position Embedding (RoPE).

    Applies rotary embeddings to query and key tensors:
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    Args:
        q_tensor: Query tensor [batch, seq_len, num_heads, head_dim]
        k_tensor: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        cos_cache: Cosine cache [max_seq_len, head_dim]
        sin_cache: Sine cache [max_seq_len, head_dim]
        position_ids: Position indices [batch, seq_len]

    Returns:
        q_rotated: Rotated query tensor
        k_rotated: Rotated key tensor
    """
    batch_size = q_tensor.shape[0]
    seq_len = q_tensor.shape[1]
    num_heads = q_tensor.shape[2]
    head_dim = q_tensor.shape[3]
    num_kv_heads = k_tensor.shape[2]

    q_out = nl.ndarray(q_tensor.shape, dtype=q_tensor.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray(k_tensor.shape, dtype=k_tensor.dtype, buffer=nl.shared_hbm)

    half_dim = head_dim // 2

    TILE_M = 128

    for b in nl.affine_range(batch_size):
        for s in nl.affine_range(seq_len):
            # Load position-specific cos/sin
            pos_idx = nl.load(position_ids[b:b+1, s:s+1])

            # Load cos and sin for this position
            cos_tile = nl.load(cos_cache[pos_idx, :])  # [1, head_dim]
            sin_tile = nl.load(sin_cache[pos_idx, :])  # [1, head_dim]

            # Process query heads
            for h in nl.affine_range(num_heads):
                q_tile = nl.load(q_tensor[b, s, h, :])  # [head_dim]

                # Split into first half and second half
                q_first = q_tile[:half_dim]
                q_second = q_tile[half_dim:]

                cos_first = cos_tile[0, :half_dim]
                cos_second = cos_tile[0, half_dim:]
                sin_first = sin_tile[0, :half_dim]
                sin_second = sin_tile[0, half_dim:]

                # rotate_half: [-x2, x1]
                # q_rotated = q * cos + rotate_half(q) * sin
                q_rot_first = nl.add(
                    nl.multiply(q_first, cos_first),
                    nl.multiply(nl.negative(q_second), sin_first)
                )
                q_rot_second = nl.add(
                    nl.multiply(q_second, cos_second),
                    nl.multiply(q_first, sin_second)
                )

                # Concatenate
                nl.store(q_out[b, s, h, :half_dim], value=q_rot_first)
                nl.store(q_out[b, s, h, half_dim:], value=q_rot_second)

            # Process key heads (similar pattern)
            for h in nl.affine_range(num_kv_heads):
                k_tile = nl.load(k_tensor[b, s, h, :])

                k_first = k_tile[:half_dim]
                k_second = k_tile[half_dim:]

                cos_first = cos_tile[0, :half_dim]
                cos_second = cos_tile[0, half_dim:]
                sin_first = sin_tile[0, :half_dim]
                sin_second = sin_tile[0, half_dim:]

                k_rot_first = nl.add(
                    nl.multiply(k_first, cos_first),
                    nl.multiply(nl.negative(k_second), sin_first)
                )
                k_rot_second = nl.add(
                    nl.multiply(k_second, cos_second),
                    nl.multiply(k_first, sin_second)
                )

                nl.store(k_out[b, s, h, :half_dim], value=k_rot_first)
                nl.store(k_out[b, s, h, half_dim:], value=k_rot_second)

    return q_out, k_out


# ============================================================================
# Fused Add + RMSNorm Kernel
# Combines residual addition with RMSNorm for reduced memory traffic
# ============================================================================

@nki.jit
def nki_fused_add_rmsnorm_kernel(input_tensor, residual, weight, eps):
    """
    Fused kernel combining residual addition with RMSNorm.

    output = RMSNorm(input + residual, weight, eps)

    This reduces memory traffic by avoiding intermediate tensor storage.

    Args:
        input_tensor: Input tensor [batch*seq_len, hidden_size]
        residual: Residual tensor [batch*seq_len, hidden_size]
        weight: RMSNorm weight [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        output: Normalized tensor [batch*seq_len, hidden_size]
    """
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)

    num_rows = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]

    TILE_M = 128
    ix = nl.arange(TILE_M)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(hidden_size)[None, :]

    # Load weight once
    w_tile = nl.load(weight.reshape((1, hidden_size))[iw, iy])

    for i in nl.affine_range(math.ceil(num_rows / TILE_M)):
        # Load input and residual
        x_tile = nl.load(
            input_tensor[i * TILE_M + ix, iy],
            mask=(i * TILE_M + ix < num_rows)
        )
        res_tile = nl.load(
            residual[i * TILE_M + ix, iy],
            mask=(i * TILE_M + ix < num_rows)
        )

        # Fused add: x = input + residual
        x_sum = nl.add(x_tile, res_tile)

        # RMSNorm computation
        x_square = nl.square(x_sum)
        mean_square = nl.sum(x_square, axis=[1]) / hidden_size
        rms_inv = nl.rsqrt(mean_square + eps)
        x_norm = nl.multiply(x_sum, rms_inv)

        # Apply weight
        w_bcast = w_tile.broadcast_to((TILE_M, hidden_size))
        out_tile = nl.multiply(x_norm, w_bcast, mask=(i * TILE_M + ix < num_rows))

        # Store result
        nl.store(
            output[i * TILE_M + ix, iy],
            value=out_tile,
            mask=(i * TILE_M + ix < num_rows)
        )

    return output


# ============================================================================
# PyTorch Module Wrappers
# ============================================================================

class NKISiLU(nn.Module):
    """NKI-accelerated SiLU activation."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        output = nki_silu_kernel(x_2d)
        return output.view(original_shape)


class NKISoftmax(nn.Module):
    """NKI-accelerated Softmax for expert routing."""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim != -1:
            x = x.transpose(self.dim, -1)

        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        output = nki_softmax_kernel(x_2d)
        output = output.view(original_shape)

        if self.dim != -1:
            output = output.transpose(self.dim, -1)

        return output


class NKIFusedAddRMSNorm(nn.Module):
    """NKI-accelerated fused Add + RMSNorm."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, residual):
        original_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        res_2d = residual.view(-1, residual.shape[-1])
        output = nki_fused_add_rmsnorm_kernel(x_2d, res_2d, self.weight, self.eps)
        return output.view(original_shape)


# ============================================================================
# Export functions
# ============================================================================

def get_nki_silu():
    """Returns NKI SiLU activation module."""
    return NKISiLU()


def get_nki_softmax(dim=-1):
    """Returns NKI Softmax module."""
    return NKISoftmax(dim=dim)


def get_nki_fused_add_rmsnorm(hidden_size, eps=1e-6):
    """Returns NKI Fused Add + RMSNorm module."""
    return NKIFusedAddRMSNorm(hidden_size, eps)
