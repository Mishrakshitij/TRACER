#!/bin/bash
# Evaluate trained models across all framings to build a complete transfer matrix.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_transfer_matrix.sh
#
# Produces a grid of results:
#   rows = models (base, rag-trained, tool-trained, followup-trained)
#   cols = framings (tool, rag, followup)
#
# Results are saved as JSON files under:
#   outputs/r3af/transfer_matrix/
set -e

# ---------------------------------------------------------------------------
# HF cache env vars
# ---------------------------------------------------------------------------
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR="outputs/r3af/transfer_matrix"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Model definitions  (name  checkpoint_path)
# For the base model, checkpoint_path is set to "NONE" (no --checkpoint flag).
# ---------------------------------------------------------------------------
MODEL_NAMES=(
    "base"
    "rag_trained"
    "tool_trained"
    "followup_trained"
)

declare -A MODEL_CKPTS
MODEL_CKPTS[base]="NONE"
MODEL_CKPTS[rag_trained]="outputs/r3af/phase1_rag_grpo_atrc/checkpoint-step-4000"
MODEL_CKPTS[tool_trained]="outputs/r3af/phase1_tool_grpo_atrc/checkpoint-step-BEST"
MODEL_CKPTS[followup_trained]="outputs/r3af/phase1_atrc_pinj/checkpoint-step-1750"

# ---------------------------------------------------------------------------
# Framing definitions  (name  config_path)
# ---------------------------------------------------------------------------
FRAMING_NAMES=(
    "tool"
    "rag"
    "followup"
)

declare -A FRAMING_CONFIGS
FRAMING_CONFIGS[tool]="configs/experiments/eval_gsm8k_tool.yaml"
FRAMING_CONFIGS[rag]="configs/experiments/eval_gsm8k_rag_base.yaml"
FRAMING_CONFIGS[followup]="configs/experiments/eval_gsm8k_followup.yaml"

# ---------------------------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------------------------
DATASET="gsm8k"
SPLIT="test"
MAX_SAMPLES=200

# ---------------------------------------------------------------------------
# Run evaluations
# ---------------------------------------------------------------------------
TOTAL=0
SKIPPED=0
FAILED=0

for model in "${MODEL_NAMES[@]}"; do
    ckpt="${MODEL_CKPTS[$model]}"

    for framing in "${FRAMING_NAMES[@]}"; do
        config="${FRAMING_CONFIGS[$framing]}"
        outfile="${OUT_DIR}/${model}_${framing}.json"
        TOTAL=$((TOTAL + 1))

        # Skip if already evaluated
        if [ -f "$outfile" ]; then
            echo "SKIP: ${model} x ${framing} (already exists: $outfile)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "============================================"
        echo "Model:   $model"
        echo "Framing: $framing"
        echo "Config:  $config"
        echo "Output:  $outfile"
        echo "============================================"

        # Build the command
        CMD=(
            python scripts/evaluate_vllm.py
            --config "$config"
            --dataset "$DATASET"
            --split "$SPLIT"
            --max_samples "$MAX_SAMPLES"
            --output "$outfile"
        )

        # Add --checkpoint only for trained models (not base)
        if [ "$ckpt" != "NONE" ]; then
            CMD+=(--checkpoint "$ckpt")
        fi

        # Run evaluation
        if "${CMD[@]}"; then
            echo "Saved: $outfile"
        else
            echo "FAILED: ${model} x ${framing}"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
done

echo "============================================"
echo "Evaluation complete."
echo "  Total:   $TOTAL"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Run:     $((TOTAL - SKIPPED - FAILED))"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Print summary transfer matrix
# ---------------------------------------------------------------------------
echo "============================= TRANSFER MATRIX ============================="
printf "%-20s" "Model"
for framing in "${FRAMING_NAMES[@]}"; do
    printf "| %-30s" "$framing"
done
echo ""
printf "%-20s" "--------------------"
for framing in "${FRAMING_NAMES[@]}"; do
    printf "| %-30s" "------------------------------"
done
echo ""

for model in "${MODEL_NAMES[@]}"; do
    printf "%-20s" "$model"
    for framing in "${FRAMING_NAMES[@]}"; do
        outfile="${OUT_DIR}/${model}_${framing}.json"
        if [ -f "$outfile" ]; then
            cell=$(python3 -c "
import json, sys
try:
    d = json.load(open('$outfile'))
    ds = d.get('gsm8k', d.get(list(d.keys())[0], {}))
    clean = ds.get('clean_accuracy', 0) * 100
    robust = ds.get('robust_accuracy', 0) * 100
    asr = ds.get('ASR', 0) * 100
    print(f'C={clean:.1f} R={robust:.1f} A={asr:.1f}')
except Exception as e:
    print(f'ERR: {e}')
" 2>/dev/null || echo "PARSE_ERR")
            printf "| %-30s" "$cell"
        else
            printf "| %-30s" "-- missing --"
        fi
    done
    echo ""
done

echo ""
echo "Legend: C=Clean% R=Robust% A=ASR%"
echo "Results directory: $OUT_DIR"
