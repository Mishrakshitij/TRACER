#!/bin/bash
# Wrapper script for TRACER evaluation.
# Uses HF_HOME (default ~/.cache/huggingface) for all HuggingFace caches.

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"

# Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== R3AF Evaluation ==="
echo "Args: $@"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================"

python scripts/evaluate.py "$@"
