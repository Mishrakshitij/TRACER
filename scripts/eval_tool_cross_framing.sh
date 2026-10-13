#!/bin/bash
# Cross-framing evaluation of tool-output trained checkpoint (v1 step-1750)
# Runs on GPU 3 — evaluates under RAG, followup, and agent framings
set -e

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"

export CUDA_VISIBLE_DEVICES=3

CHECKPOINT="outputs/r3af/phase1_tool_grpo_atrc/checkpoint-step-1750"
OUT_DIR="outputs/r3af/transfer_matrix"
mkdir -p "$OUT_DIR"

cd "$(dirname "$0")/.."

# 1. Tool-trained under RAG framing
if [ ! -f "$OUT_DIR/tool_trained_rag.json" ]; then
    echo "============================================"
    echo "Eval: tool_trained × RAG framing"
    echo "============================================"
    python scripts/evaluate_vllm.py \
        --config configs/experiments/eval_gsm8k_rag_base.yaml \
        --checkpoint "$CHECKPOINT" \
        --dataset gsm8k --split test --max_samples 200 \
        --output "$OUT_DIR/tool_trained_rag.json"
    echo "Saved: tool_trained_rag.json"
fi

# 2. Tool-trained under followup framing
if [ ! -f "$OUT_DIR/tool_trained_followup.json" ]; then
    echo "============================================"
    echo "Eval: tool_trained × followup framing"
    echo "============================================"
    python scripts/evaluate_vllm.py \
        --config configs/experiments/eval_gsm8k_followup.yaml \
        --checkpoint "$CHECKPOINT" \
        --dataset gsm8k --split test --max_samples 200 \
        --output "$OUT_DIR/tool_trained_followup.json"
    echo "Saved: tool_trained_followup.json"
fi

# 3. Tool-trained under agent-msg framing
if [ ! -f "$OUT_DIR/tool_trained_agent.json" ]; then
    echo "============================================"
    echo "Eval: tool_trained × agent-msg framing"
    echo "============================================"
    python scripts/evaluate_vllm.py \
        --config configs/experiments/eval_gsm8k_agent.yaml \
        --checkpoint "$CHECKPOINT" \
        --dataset gsm8k --split test --max_samples 200 \
        --output "$OUT_DIR/tool_trained_agent.json"
    echo "Saved: tool_trained_agent.json"
fi

# 4. Tool-trained under tool-output framing (native — copy from v1 evals)
if [ ! -f "$OUT_DIR/tool_trained_tool.json" ]; then
    echo "Copying native framing result from v1 evals..."
    cp outputs/r3af/eval_tool_checkpoints/checkpoint-step-1750.json \
       "$OUT_DIR/tool_trained_tool.json"
fi

echo ""
echo "============================================"
echo "Cross-framing results for tool-trained model:"
echo "============================================"
for f in "$OUT_DIR"/tool_trained_*.json; do
    name=$(basename "$f" .json)
    python3 -c "import json; d=json.load(open('$f')); g=d['gsm8k']; print(f'$name: Clean={g[\"clean_accuracy\"]*100:.1f}% Robust={g[\"robust_accuracy\"]*100:.1f}% ASR={g[\"ASR\"]*100:.1f}%')" 2>/dev/null
done

echo ""
echo "All cross-framing evals complete!"
