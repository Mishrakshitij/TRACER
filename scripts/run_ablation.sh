#!/bin/bash
# Ablation: isolate wrapper vs system prompt contribution (GPU 7)
set -e

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=7

CKPT="outputs/r3af/phase1_rag_grpo_atrc/checkpoint-step-4000"
OUT_DIR="outputs/r3af/analysis_ablation"
mkdir -p "$OUT_DIR"

cd "$(dirname "$0")/.."

for MODE in full prompt_only wrapper_only; do
    OUT="$OUT_DIR/rag_step4000_${MODE}.json"
    if [ -f "$OUT" ]; then
        echo "Already done: $MODE"
        continue
    fi
    echo "============================================"
    echo "Ablation mode: $MODE"
    echo "============================================"
    python scripts/eval_ablation.py \
        --config configs/experiments/eval_gsm8k_rag_base.yaml \
        --ablation_mode "$MODE" \
        --checkpoint "$CKPT" \
        --dataset gsm8k --split test --max_samples 200 \
        --output "$OUT"
    echo "Saved: $OUT"
done

echo ""
echo "============================================"
echo "Ablation results:"
echo "============================================"
for MODE in full prompt_only wrapper_only; do
    f="$OUT_DIR/rag_step4000_${MODE}.json"
    [ -f "$f" ] && python3 -c "
import json
d = json.load(open('$f'))
g = d['gsm8k']
print(f'$MODE: Clean={g[\"clean_accuracy\"]*100:.1f}%  Robust={g[\"robust_accuracy\"]*100:.1f}%  ASR={g[\"ASR\"]*100:.1f}%')
" 2>/dev/null
done
echo "Done!"
