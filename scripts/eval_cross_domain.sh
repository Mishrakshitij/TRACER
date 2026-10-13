#!/bin/bash
# Cross-domain eval: tool-trained final checkpoint on NQ and TriviaQA (GPU 1)
set -e

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=1

CKPT="outputs/r3af/phase1_tool_grpo_atrc_resume/checkpoint-final"
NQ_OUT="outputs/r3af/eval_nq_rag/tool_final.json"
TQA_OUT="outputs/r3af/eval_triviaqa_rag/tool_final.json"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Eval: Tool-final on Natural Questions (RAG framing)"
echo "============================================"
python scripts/evaluate_vllm.py \
    --config configs/experiments/eval_nq_rag.yaml \
    --checkpoint "$CKPT" \
    --dataset nq --split validation --max_samples 200 \
    --output "$NQ_OUT"
echo "Saved: $NQ_OUT"

echo "============================================"
echo "Eval: Tool-final on TriviaQA (RAG framing)"
echo "============================================"
python scripts/evaluate_vllm.py \
    --config configs/experiments/eval_triviaqa_rag.yaml \
    --checkpoint "$CKPT" \
    --dataset triviaqa --split validation --max_samples 200 \
    --output "$TQA_OUT"
echo "Saved: $TQA_OUT"

echo ""
echo "============================================"
echo "Cross-domain results (tool-trained final):"
echo "============================================"
for TASK in nq triviaqa; do
    f="outputs/r3af/eval_${TASK}_rag/tool_final.json"
    [ -f "$f" ] && python3 -c "
import json
d = json.load(open('$f'))
key = list(d.keys())[0]
g = d[key]
print(f'$TASK: Clean={g[\"clean_accuracy\"]*100:.1f}%  Robust={g[\"robust_accuracy\"]*100:.1f}%  ASR={g[\"ASR\"]*100:.1f}%')
" 2>/dev/null
done
echo "Done!"
