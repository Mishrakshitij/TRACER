#!/bin/bash
# Evaluate multi-framing-trained final checkpoint under all 4 framings (GPU 6)
set -e

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=6

CKPT="outputs/r3af/phase1_multi_framing_grpo_atrc/checkpoint-final"
OUT_DIR="outputs/r3af/transfer_matrix"
mkdir -p "$OUT_DIR"

cd "$(dirname "$0")/.."

declare -A CONFIGS=(
    ["mf_final_followup"]="configs/experiments/eval_gsm8k_followup.yaml"
    ["mf_final_rag"]="configs/experiments/eval_gsm8k_rag_base.yaml"
    ["mf_final_tool"]="configs/experiments/eval_gsm8k_tool.yaml"
    ["mf_final_agent"]="configs/experiments/eval_gsm8k_agent.yaml"
)

for NAME in mf_final_followup mf_final_rag mf_final_tool mf_final_agent; do
    OUT="$OUT_DIR/${NAME}.json"
    CFG="${CONFIGS[$NAME]}"
    if [ -f "$OUT" ]; then
        echo "Already done: $NAME"
        continue
    fi
    echo "============================================"
    echo "Eval: $NAME"
    echo "============================================"
    python scripts/evaluate_vllm.py \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --dataset gsm8k --split test --max_samples 200 \
        --output "$OUT"
    echo "Saved: ${NAME}.json"
done

echo ""
echo "============================================"
echo "Multi-framing cross-framing results:"
echo "============================================"
for FRAMING in followup rag tool agent; do
    f="$OUT_DIR/mf_final_${FRAMING}.json"
    [ -f "$f" ] && python3 -c "
import json
d = json.load(open('$f'))
g = d['gsm8k']
print(f'$FRAMING: Clean={g[\"clean_accuracy\"]*100:.1f}%  Robust={g[\"robust_accuracy\"]*100:.1f}%  ASR={g[\"ASR\"]*100:.1f}%')
" 2>/dev/null
done
echo "Done!"
