#!/bin/bash
# Evaluate all tool-output trained checkpoints
set -e

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"

TRAIN_DIR="outputs/r3af/phase1_tool_grpo_atrc"
EVAL_DIR="outputs/r3af/eval_tool_checkpoints"
CONFIG="configs/experiments/eval_gsm8k_tool.yaml"
mkdir -p "$EVAL_DIR"

for ckpt in $(ls -d "$TRAIN_DIR"/checkpoint-* | sort -t- -k3 -n); do
    name=$(basename "$ckpt")
    outfile="$EVAL_DIR/${name}.json"

    if [ -f "$outfile" ]; then
        echo "SKIP: $name (already evaluated)"
        continue
    fi

    echo "============================================"
    echo "Evaluating: $name"
    echo "============================================"

    python scripts/evaluate_vllm.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt" \
        --dataset gsm8k --split test --max_samples 200 \
        --output "$outfile"

    echo "Saved: $outfile"
    echo ""
done

echo "All checkpoints evaluated!"
echo ""
echo "Summary:"
for f in "$EVAL_DIR"/*.json; do
    name=$(basename "$f" .json)
    clean=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gsm8k']['clean_accuracy']*100:.1f}%\")")
    robust=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gsm8k']['robust_accuracy']*100:.1f}%\")")
    asr=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['gsm8k']['ASR']*100:.1f}%\")")
    echo "$name: Clean=$clean Robust=$robust ASR=$asr"
done
