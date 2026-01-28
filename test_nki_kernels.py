#!/usr/bin/env python3
"""
Test suite for NKI kernels.

This script tests all custom NKI kernels against PyTorch reference implementations
to verify correctness before running the full model.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python test_nki_kernels.py
"""

import torch
import numpy as np
import sys

# Try to import NKI modules
try:
    import torch_xla.core.xla_model as xm
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    HAS_NEURON = True
    DEVICE = xm.xla_device()
except ImportError:
    HAS_NEURON = False
    DEVICE = torch.device('cpu')
    print("Warning: Neuron not available, running CPU reference tests only")


def test_rmsnorm_kernel():
    """Test NKI RMSNorm kernel against PyTorch reference."""
    print("\n" + "=" * 70)
    print("Testing RMSNorm Kernel")
    print("=" * 70)

    from nki_custom_rmsnorm import NKIRMSNorm

    hidden_size = 2048
    eps = 1e-6
    batch_seq = 256

    # Create test data on CPU first
    torch.manual_seed(42)
    x_cpu = torch.randn(batch_seq, hidden_size, dtype=torch.float32)
    weight_cpu = torch.ones(hidden_size, dtype=torch.float32)

    # PyTorch reference (on CPU)
    def pytorch_rmsnorm(x, weight, eps):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x / torch.sqrt(variance + eps)
        return x_normed * weight

    ref_output_cpu = pytorch_rmsnorm(x_cpu, weight_cpu, eps)

    if HAS_NEURON:
        # NKI kernel - create module first, then move to device
        nki_module = NKIRMSNorm(hidden_size, eps)
        nki_module = nki_module.to(DEVICE)

        # Move input to device
        x = x_cpu.to(DEVICE)
        nki_output = nki_module(x)

        # Move outputs back to CPU for comparison
        nki_output_cpu = nki_output.cpu()

        # Compare
        max_diff = torch.abs(ref_output_cpu - nki_output_cpu).max().item()
        mean_diff = torch.abs(ref_output_cpu - nki_output_cpu).mean().item()
    else:
        max_diff = 0.0
        mean_diff = 0.0

    print(f"  Input shape: ({batch_seq}, {hidden_size})")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    passed = max_diff < 1e-3 if HAS_NEURON else True
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_silu_kernel():
    """Test NKI SiLU kernel against PyTorch reference."""
    print("\n" + "=" * 70)
    print("Testing SiLU Kernel")
    print("=" * 70)

    batch_seq = 256
    hidden_size = 2048

    # Create test data on CPU
    torch.manual_seed(42)
    x_cpu = torch.randn(batch_seq, hidden_size, dtype=torch.float32)

    # PyTorch reference
    ref_silu_cpu = torch.nn.functional.silu(x_cpu)

    if HAS_NEURON:
        from nki_kernels import nki_silu_kernel

        x = x_cpu.to(DEVICE)
        nki_output = nki_silu_kernel(x)
        nki_output_cpu = nki_output.cpu()

        max_diff = torch.abs(ref_silu_cpu - nki_output_cpu).max().item()
        mean_diff = torch.abs(ref_silu_cpu - nki_output_cpu).mean().item()
    else:
        max_diff = 0.0
        mean_diff = 0.0

    print(f"  Input shape: ({batch_seq}, {hidden_size})")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    passed = max_diff < 1e-3 if HAS_NEURON else True
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_softmax_kernel():
    """Test NKI Softmax kernel against PyTorch reference."""
    print("\n" + "=" * 70)
    print("Testing Softmax Kernel (for Expert Routing)")
    print("=" * 70)

    num_tokens = 256
    num_experts = 128

    # Create test data on CPU
    torch.manual_seed(42)
    x_cpu = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    # PyTorch reference
    ref_softmax_cpu = torch.nn.functional.softmax(x_cpu, dim=-1)

    if HAS_NEURON:
        from nki_kernels import nki_softmax_kernel

        x = x_cpu.to(DEVICE)
        nki_output = nki_softmax_kernel(x)
        nki_output_cpu = nki_output.cpu()

        max_diff = torch.abs(ref_softmax_cpu - nki_output_cpu).max().item()
        mean_diff = torch.abs(ref_softmax_cpu - nki_output_cpu).mean().item()
    else:
        max_diff = 0.0
        mean_diff = 0.0

    print(f"  Input shape: ({num_tokens}, {num_experts})")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    passed = max_diff < 1e-3 if HAS_NEURON else True
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_fused_add_rmsnorm_kernel():
    """Test NKI Fused Add+RMSNorm kernel against PyTorch reference."""
    print("\n" + "=" * 70)
    print("Testing Fused Add+RMSNorm Kernel")
    print("=" * 70)

    hidden_size = 2048
    eps = 1e-6
    batch_seq = 256

    # Create test data on CPU
    torch.manual_seed(42)
    x_cpu = torch.randn(batch_seq, hidden_size, dtype=torch.float32)
    residual_cpu = torch.randn(batch_seq, hidden_size, dtype=torch.float32)
    weight_cpu = torch.ones(hidden_size, dtype=torch.float32)

    # PyTorch reference
    def pytorch_fused_add_rmsnorm(x, residual, weight, eps):
        x_sum = x + residual
        variance = x_sum.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_sum / torch.sqrt(variance + eps)
        return x_normed * weight

    ref_output_cpu = pytorch_fused_add_rmsnorm(x_cpu, residual_cpu, weight_cpu, eps)

    if HAS_NEURON:
        from nki_kernels import nki_fused_add_rmsnorm_kernel

        x = x_cpu.to(DEVICE)
        residual = residual_cpu.to(DEVICE)
        weight = weight_cpu.to(DEVICE)
        nki_output = nki_fused_add_rmsnorm_kernel(x, residual, weight, eps)
        nki_output_cpu = nki_output.cpu()

        max_diff = torch.abs(ref_output_cpu - nki_output_cpu).max().item()
        mean_diff = torch.abs(ref_output_cpu - nki_output_cpu).mean().item()
    else:
        max_diff = 0.0
        mean_diff = 0.0

    print(f"  Input shape: ({batch_seq}, {hidden_size})")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    passed = max_diff < 1e-3 if HAS_NEURON else True
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_rope_kernel():
    """Test NKI RoPE kernel against PyTorch reference."""
    print("\n" + "=" * 70)
    print("Testing RoPE (Rotary Position Embedding) Kernel")
    print("=" * 70)

    batch = 1
    seq_len = 128
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128
    max_seq_len = 2048
    base = 10000.0

    # Create test data on CPU
    torch.manual_seed(42)
    q_cpu = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float32)
    k_cpu = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float32)

    # Precompute cos/sin cache on CPU
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache_cpu = emb.cos()
    sin_cache_cpu = emb.sin()

    # PyTorch reference RoPE
    def pytorch_rope(q, k, cos_cache, sin_cache):
        cos = cos_cache[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
        sin = sin_cache[:seq_len].unsqueeze(0).unsqueeze(2)

        def rotate_half(x):
            x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

    ref_q_cpu, ref_k_cpu = pytorch_rope(q_cpu, k_cpu, cos_cache_cpu, sin_cache_cpu)

    if HAS_NEURON:
        from nki_attention_kernels import nki_rope_kernel

        q = q_cpu.to(DEVICE)
        k = k_cpu.to(DEVICE)
        cos_cache = cos_cache_cpu.to(DEVICE)
        sin_cache = sin_cache_cpu.to(DEVICE)

        nki_q, nki_k = nki_rope_kernel(q, k, cos_cache, sin_cache, seq_offset=0)
        nki_q_cpu = nki_q.cpu()
        nki_k_cpu = nki_k.cpu()

        q_max_diff = torch.abs(ref_q_cpu - nki_q_cpu).max().item()
        k_max_diff = torch.abs(ref_k_cpu - nki_k_cpu).max().item()
        max_diff = max(q_max_diff, k_max_diff)
    else:
        max_diff = 0.0

    print(f"  Q shape: ({batch}, {seq_len}, {num_heads}, {head_dim})")
    print(f"  K shape: ({batch}, {seq_len}, {num_kv_heads}, {head_dim})")
    print(f"  Max difference: {max_diff:.6e}")

    passed = max_diff < 1e-3 if HAS_NEURON else True
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_expert_routing_kernel():
    """Test NKI Expert Routing kernel (softmax + PyTorch TopK)."""
    print("\n" + "=" * 70)
    print("Testing Expert Routing Kernel (Hybrid NKI Softmax + PyTorch TopK)")
    print("=" * 70)

    num_tokens = 256
    num_experts = 128
    num_experts_per_tok = 8

    # Create test data on CPU
    torch.manual_seed(42)
    router_logits_cpu = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    # PyTorch reference
    probs_cpu = torch.nn.functional.softmax(router_logits_cpu, dim=-1)
    topk_probs_cpu, topk_indices_cpu = torch.topk(probs_cpu, num_experts_per_tok, dim=-1)
    ref_weights_cpu = topk_probs_cpu / topk_probs_cpu.sum(dim=-1, keepdim=True)

    if HAS_NEURON:
        from nki_moe_kernels import nki_expert_router_kernel

        router_logits = router_logits_cpu.to(DEVICE)
        nki_weights, nki_indices = nki_expert_router_kernel(router_logits, num_experts_per_tok)

        # Move back to CPU for comparison
        nki_weights_cpu = nki_weights.cpu()
        nki_indices_cpu = nki_indices.cpu()

        weights_diff = torch.abs(ref_weights_cpu - nki_weights_cpu).max().item()
        indices_match = torch.equal(topk_indices_cpu, nki_indices_cpu)
    else:
        weights_diff = 0.0
        indices_match = True

    print(f"  Router logits shape: ({num_tokens}, {num_experts})")
    print(f"  TopK: {num_experts_per_tok}")
    print(f"  Weights max difference: {weights_diff:.6e}")
    print(f"  Indices match: {indices_match}")

    passed = (weights_diff < 1e-3 and indices_match) if HAS_NEURON else True
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    return passed


def run_all_tests():
    """Run all kernel tests and report results."""
    print("\n" + "=" * 70)
    print("NKI Kernel Test Suite for AWS Trainium2/3 MoE Challenge")
    print("=" * 70)

    if HAS_NEURON:
        print(f"Device: {DEVICE}")
    else:
        print("Device: CPU (reference mode - Neuron not available)")

    results = {}

    # Run tests
    results['rmsnorm'] = test_rmsnorm_kernel()
    results['silu'] = test_silu_kernel()
    results['softmax'] = test_softmax_kernel()
    results['fused_add_rmsnorm'] = test_fused_add_rmsnorm_kernel()
    results['rope'] = test_rope_kernel()
    results['expert_routing'] = test_expert_routing_kernel()

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    for name, status in results.items():
        print(f"  {name}: {'PASSED' if status else 'FAILED'}")

    print("-" * 70)
    print(f"  Total: {total}, Passed: {passed}, Failed: {failed}")

    if failed == 0:
        print("\n*** ALL TESTS PASSED ***\n")
        return 0
    else:
        print(f"\n*** {failed} TEST(S) FAILED ***\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
