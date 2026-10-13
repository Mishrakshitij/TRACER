#!/bin/bash
# Evaluate v2 tool-output checkpoints under tool framing (GPU 5)
set -e

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"

export CUDA_VISIBLE_DEVICES=5

V2_DIR="outputs/r3af/phase1_tool_grpo_atrc_v2"
OUT_DIR="outputs/r3af/eval_v2_checkpoints"
mkdir -p "$OUT_DIR"

cd "$(dirname "$0")/.."

for STEP in 2000 2250 2500 2750; do
    CKPT="$V2_DIR/checkpoint-step-$STEP"
    OUT="$OUT_DIR/checkpoint-step-$STEP.json"

    if [ -f "$OUT" ]; then
        echo "Already done: step-$STEP"
        continue
    fi

    if [ ! -d "$CKPT" ]; then
        echo "Checkpoint not found: $CKPT, skipping"
        continue
    fi

    echo "============================================"
    echo "Eval: v2 step-$STEP under tool-output framing"
    echo "============================================"
    python scripts/evaluate_vllm.py \
        --config configs/experiments/eval_gsm8k_tool.yaml \
        --checkpoint "$CKPT" \
        --dataset gsm8k --split test --max_samples 200 \
        --output "$OUT"
    echo "Saved: step-$STEP.json"
done

echo ""
echo "============================================"
echo "V2 checkpoint results (tool framing):"
echo "============================================"
for STEP in 2000 2250 2500 2750; do
    f="$OUT_DIR/checkpoint-step-$STEP.json"
    [ -f "$f" ] && python3 -c "import json; d=json.load(open('$f')); g=d['gsm8k']; print(f'step-$STEP: Clean={g[\"clean_accuracy\"]*100:.1f}% Robust={g[\"robust_accuracy\"]*100:.1f}% ASR={g[\"ASR\"]*100:.1f}%')" 2>/dev/null
done

echo "Done!"
