# NKI Kernel Implementation for Qwen3-30B-A3B MoE

## Overview

This document describes the custom NKI (Neuron Kernel Interface) kernels implemented for the AWS Trainium2/3 MoE Kernel Challenge. These kernels optimize the Qwen3-30B-A3B Mixture of Experts model for inference on AWS Trainium hardware.

## Implemented Kernels

### 1. RMSNorm Kernel (`nki_custom_rmsnorm.py`)

**Function:** `nki_rmsnorm_kernel`

**Purpose:** Layer normalization using Root Mean Square normalization.

**Optimization Strategy:**
- Process 128 rows at a time (NeuronCore partition constraint)
- Load weights once and broadcast for efficiency
- Fused square, mean, rsqrt operations
- Masked stores for boundary handling

**Formula:**
```
output = (x / sqrt(mean(x^2) + eps)) * weight
```

**Performance Impact:** Used throughout the model for normalization operations.

---

### 2. SiLU Activation Kernel (`nki_kernels.py`)

**Function:** `nki_silu_kernel`

**Purpose:** Swish/SiLU activation function used in MoE expert MLPs.

**Optimization Strategy:**
- Tiled processing (128x512 tiles)
- Fused sigmoid computation: `1 / (1 + exp(-x))`
- In-place multiplication for SiLU output
- Efficient memory access patterns

**Formula:**
```
silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
```

**Performance Impact:** Called for every expert MLP gate activation.

---

### 3. Softmax Kernel (`nki_kernels.py`)

**Function:** `nki_softmax_kernel`

**Purpose:** Expert routing probability computation.

**Optimization Strategy:**
- Numerically stable implementation with max subtraction
- Single-pass computation where possible
- Optimized for 128 experts (Qwen3-30B-A3B specification)

**Formula:**
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

**Performance Impact:** Critical for MoE token-to-expert routing.

---

### 4. Fused Add+RMSNorm Kernel (`nki_kernels.py`)

**Function:** `nki_fused_add_rmsnorm_kernel`

**Purpose:** Combines residual connection with layer normalization.

**Optimization Strategy:**
- Single kernel reduces memory traffic
- Avoids intermediate tensor materialization
- Efficient for transformer residual patterns

**Formula:**
```
output = RMSNorm(x + residual)
```

**Performance Impact:** Reduces memory bandwidth by ~50% for these operations.

---

### 5. Fused Expert MLP Kernel (`nki_moe_kernels.py`)

**Function:** `nki_fused_expert_mlp_kernel`

**Purpose:** Complete MoE expert MLP computation in a single kernel.

**Optimization Strategy:**
- Fused gate_proj and up_proj computations
- In-kernel SiLU activation
- Tiled matrix multiplication
- Efficient weight reuse

**Computation Flow:**
```
1. gate_proj = x @ gate_weight
2. up_proj = x @ up_weight
3. hidden = silu(gate_proj) * up_proj
4. output = hidden @ down_weight
```

**Performance Impact:** Major kernel - covers significant portion of model FLOPs.

---

### 6. Expert Router Kernel (`nki_moe_kernels.py`)

**Function:** `nki_expert_router_kernel`

**Purpose:** Token-to-expert routing with TopK selection.

**Optimization Strategy:**
- Combined softmax and TopK in single kernel
- Normalized routing weights output
- Efficient iterative TopK selection

**Output:**
- `routing_weights`: [batch*seq, num_experts_per_tok]
- `selected_experts`: [batch*seq, num_experts_per_tok]

---

### 7. Expert Aggregation Kernel (`nki_moe_kernels.py`)

**Function:** `nki_expert_aggregation_kernel`

**Purpose:** Weighted sum of expert outputs.

**Optimization Strategy:**
- Efficient broadcast of routing weights
- Tiled accumulation for large hidden dimensions
- Mixed precision accumulation for accuracy

---

### 8. RoPE (Rotary Position Embedding) Kernel (`nki_attention_kernels.py`)

**Function:** `nki_rope_kernel`

**Purpose:** Apply rotary position embeddings to Q and K tensors.

**Optimization Strategy:**
- Precomputed cos/sin caches
- Efficient half-dimension rotation
- Support for incremental decoding offsets

**Formula:**
```
q_rotated = q * cos + rotate_half(q) * sin
k_rotated = k * cos + rotate_half(k) * sin
```

---

### 9. QK RMSNorm Kernel (`nki_attention_kernels.py`)

**Function:** `nki_qk_rmsnorm_kernel`

**Purpose:** Normalize Q and K tensors before attention (Qwen3 specific).

**Optimization Strategy:**
- Process Q and K in same kernel
- Per-head normalization
- Efficient weight broadcasting

---

## File Structure

```
nki-moe/
├── nki_custom_rmsnorm.py     # RMSNorm NKI kernel
├── nki_kernels.py            # SiLU, Softmax, Fused Add+RMSNorm
├── nki_moe_kernels.py        # Expert MLP, Router, Aggregation
├── nki_attention_kernels.py  # RoPE, QK Norm, Attention
├── qwen_with_nki.py          # Model with NKI integration
├── test_nki_kernels.py       # Kernel correctness tests
└── run_nki_inference.sh      # Inference runner script
```

## Usage

### Test Kernels
```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python3 test_nki_kernels.py
```

### Run Inference with NKI
```bash
./run_nki_inference.sh --generate
```

### Run Evaluation
```bash
./run_nki_inference.sh --evaluate
```

## NKI FLOPS Estimation

| Kernel | Estimated FLOPS Contribution |
|--------|------------------------------|
| Expert MLPs (gate/up/down) | ~70% |
| RMSNorm (all layers) | ~5% |
| Attention (QKV + scores) | ~15% |
| Routing (softmax + topk) | ~2% |
| Other (embeddings, etc.) | ~8% |

## Optimization Notes

### Memory Access Patterns
- All kernels use 128-row tiles (NeuronCore partition constraint)
- Tile sizes optimized for SBUF capacity
- Masked operations for boundary handling

### Numerical Precision
- FP32 accumulation for matrix multiplications
- Cast to input dtype for outputs
- Numerically stable softmax with max subtraction

### Future Optimizations
1. Flash Attention integration
2. Dynamic expert batching
3. Weight quantization support
4. Expert parallel execution

## Performance Metrics

Target improvements over baseline:
- **Latency (TTFT):** 2-5x reduction
- **Throughput:** 2-3x increase
- **NKI FLOPS Ratio:** >80%

## Contact

For questions about this implementation, refer to the competition guidelines or email: nki-mlsys-2026@amazon.com
