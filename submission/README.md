# AWS Trainium2/3 MoE Kernel Challenge Submission

## Team Information
- **Team Name**: [Your Team Name]
- **Members**: [Member Names]
- **Email**: [Contact Email]

## Submission Contents

### Code Files
| File | Description |
|------|-------------|
| `nki_custom_rmsnorm.py` | NKI RMSNorm kernel implementation |
| `nki_kernels.py` | SiLU, Softmax, Fused Add+RMSNorm kernels |
| `nki_moe_kernels.py` | MoE Expert routing and aggregation kernels |
| `nki_attention_kernels.py` | RoPE and attention-related kernels |
| `qwen_with_nki.py` | Model implementation with NKI integration |
| `test_nki_kernels.py` | Kernel correctness tests |

### Documentation
| File | Description |
|------|-------------|
| `technical_report.tex` | Technical report (LaTeX source) |
| `technical_report.pdf` | Technical report (compiled PDF) |
| `benchmark_results.json` | Detailed benchmark results |

## Performance Summary

| Metric | Baseline | NKI Optimized | Improvement |
|--------|----------|---------------|-------------|
| **Total Score** | 8.24 | **11.84** | **+44%** |
| Avg. Latency | 10,870 ms | 9,095 ms | -16% |
| Avg. Throughput | 72.1 tok/s | 83.5 tok/s | +16% |
| NKI FLOPS Ratio | 99.7% | 99.7% | - |
| Accuracy | 100% | 100% | - |

## NKI Kernels Implemented

1. **RMSNorm** - Layer normalization with 128-row tiling
2. **SiLU** - Fused Swish activation
3. **Softmax** - Numerically stable softmax for routing
4. **RoPE** - Rotary Position Embedding
5. **Fused Add+RMSNorm** - Combined residual add with normalization
6. **Expert Router Softmax** - Optimized routing probability computation

## How to Run

### Prerequisites
```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Download model (if not already done)
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ~/qwen-30b-a3b/hf_model
```

### Run Tests
```bash
python3 test_nki_kernels.py
```

### Run Inference
```bash
python3 main.py \
    --mode generate \
    --enable-nki \
    --model-path ~/qwen-30b-a3b/hf_model \
    --compiled-model-path ~/qwen-30b-a3b/traced_model \
    --prompt "What is the capital of France?"
```

### Run Evaluation
```bash
python3 main.py \
    --mode evaluate_all \
    --enable-nki \
    --model-path ~/qwen-30b-a3b/hf_model \
    --compiled-model-path ~/qwen-30b-a3b/traced_model \
    --skip-compile True
```

## Hardware Requirements
- AWS Trainium2 instance (trn2.xlarge or larger)
- AWS Neuron SDK 2.27
- ~60GB disk space for model weights

## License
Apache 2.0
