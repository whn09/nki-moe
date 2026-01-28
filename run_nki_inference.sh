#!/bin/bash
# =============================================================================
# AWS Trainium2/3 MoE Kernel Challenge - NKI Inference Runner
# =============================================================================
#
# This script runs inference with custom NKI kernels on the Qwen3-30B-A3B model.
#
# Usage:
#   ./run_nki_inference.sh [--test-kernels | --generate | --evaluate]
#
# =============================================================================

set -e

# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Default paths (expand ~ to $HOME)
MODEL_PATH="${MODEL_PATH:-$HOME/qwen-30b-a3b/hf_model}"
COMPILED_MODEL_PATH="${COMPILED_MODEL_PATH:-$HOME/qwen-30b-a3b/traced_model}"
PROMPT="What is the capital of France?"

# Parse arguments
MODE="generate"
if [ "$1" == "--test-kernels" ]; then
    MODE="test"
elif [ "$1" == "--evaluate" ]; then
    MODE="evaluate"
elif [ "$1" == "--generate" ]; then
    MODE="generate"
fi

echo "=============================================="
echo "AWS Trainium2/3 MoE Kernel Challenge"
echo "=============================================="
echo "Mode: $MODE"
echo ""

case $MODE in
    test)
        echo "Running NKI kernel tests..."
        python3 test_nki_kernels.py
        ;;

    generate)
        echo "Running inference with NKI kernels..."
        echo "Model path: $MODEL_PATH"
        echo "Compiled model path: $COMPILED_MODEL_PATH"
        echo ""

        # Check if model exists
        if [ ! -d "$MODEL_PATH" ]; then
            echo "ERROR: Model not found at $MODEL_PATH"
            echo "Please download the model first:"
            echo "  huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir $MODEL_PATH"
            exit 1
        fi

        # Clean compile cache for fresh compilation with NKI
        echo "Clearing compile cache..."
        rm -rf /var/tmp/neuron-compile-cache/*
        rm -rf "$COMPILED_MODEL_PATH"

        echo "Starting NKI-accelerated inference..."
        python3 main.py \
            --mode generate \
            --enable-nki \
            --model-path "$MODEL_PATH" \
            --compiled-model-path "$COMPILED_MODEL_PATH" \
            --prompt "$PROMPT"
        ;;

    evaluate)
        echo "Running evaluation with NKI kernels..."

        # # Clean compile cache
        # rm -rf /var/tmp/neuron-compile-cache/*
        # rm -rf "$COMPILED_MODEL_PATH"

        python3 main.py \
            --mode evaluate_all \
            --enable-nki \
            --model-path "$MODEL_PATH" \
            --compiled-model-path "$COMPILED_MODEL_PATH" \
            --skip-compile True
        ;;

    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

echo ""
echo "Done!"