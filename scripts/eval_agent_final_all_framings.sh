#!/bin/bash
# Evaluate agent-trained final checkpoint across all 4 framings.
# Waits for checkpoint-final to appear, then runs eval on GPU 2.

set -e

OUT_DIR="outputs/r3af/transfer_matrix"
AGENT_CKPT="outputs/r3af/phase1_agent_grpo_atrc/checkpoint-final"
SCRIPTS="scripts"
CONFIGS="configs/experiments"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets

mkdir -p "$OUT_DIR"

# Wait for final checkpoint
echo "Waiting for checkpoint-final..."
while [ ! -d "$AGENT_CKPT" ]; do
    sleep 60
done
echo "checkpoint-final found: $AGENT_CKPT"

declare -A FRAMING_CONFIGS=(
    [followup]="${CONFIGS}/eval_gsm8k_followup.yaml"
    [rag]="${CONFIGS}/eval_gsm8k_rag_base.yaml"
    [tool]="${CONFIGS}/eval_gsm8k_tool.yaml"
    [agent]="${CONFIGS}/eval_gsm8k_agent.yaml"
)

for framing in followup rag tool agent; do
    outfile="${OUT_DIR}/agent_trained_${framing}.json"
    if [ -f "$outfile" ]; then
        echo "SKIP: agent_trained x $framing already exists"
        continue
    fi
    echo "Evaluating: agent_trained x $framing"
    CUDA_VISIBLE_DEVICES=2 python "${SCRIPTS}/evaluate_vllm.py" \
        --config "${FRAMING_CONFIGS[$framing]}" \
        --dataset gsm8k --split test --max_samples 200 \
        --checkpoint "${AGENT_CKPT}" \
        --output "$outfile"
    echo "Done: $outfile"
done

echo "All agent evals complete."
python3 -c "
import json, os
OUT_DIR = 'outputs/r3af/transfer_matrix'
for framing in ['followup', 'rag', 'tool', 'agent']:
    f = os.path.join(OUT_DIR, f'agent_trained_{framing}.json')
    if os.path.exists(f):
        d = json.load(open(f)); g = d.get('gsm8k', d)
        print(f'agent_trained x {framing}: ASR={g[\"ASR\"]*100:.1f}%, clean={g[\"clean_accuracy\"]*100:.1f}%')
"
