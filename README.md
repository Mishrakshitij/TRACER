# TRACER: Trace-Regularized Training Against Indirect Prompt Injection

**TRACER** (Trace-Regularized Adversarial Channel Exposure with Reward-shaping) is a framework for training LLMs to resist adversarial content injection across multiple deployment formats (RAG documents, tool outputs, agent messages, user follow-ups). It addresses the **framing gap**: robustness learned under one context presentation format does not transfer to another, even when the adversarial content is identical.

> **Paper**: *TRACER: Trace-Regularized Training Against Indirect Prompt Injection*. Kshitij Mishra, Salem Lahlou, Nils Lukas. Findings of EMNLP 2026.
>
> The Python package is named `r3af`, the project's earlier working name.

---

## Key Results

### Transfer Matrix — ASR (%) by Training Condition × Eval Framing

| Model | Followup | RAG | Tool | Agent | Avg |
|-------|----------|-----|------|-------|-----|
| Base (3B, no LoRA) | 41.4 | 65.5 | 59.4 | 46.5 | 53.2 |
| RAG-trained (3B) | 14.5 | **6.1** | 25.7 | 26.5 | 18.2 |
| Tool-trained (3B) | 20.0 | 20.2 | **15.0** | 15.8 | 17.8 |
| Multi-framing RAG+Tool (3B) | **11.3** | 5.2 | 21.7 | 17.5 | **13.9** |
| Agent-trained (3B) | 19.6 | 15.7 | 26.3 | 18.4 | 20.0 |
| **7B Base (no LoRA)** | 9.5 | 34.3 | 36.6 | 16.8 | 24.3 |
| **7B RAG-trained (final)** | **5.1** | 13.6 | 18.2 | **6.2** | **10.8** |

All models: Qwen2.5-{3B,7B}-Instruct with 4-bit QLoRA (r=64, α=128), GRPO+ATRC loss, GSM8K, 200 eval samples per cell.

**Key findings:**
- **Framing gap**: switching from followup to RAG framing raises base model ASR from 41% to 66% — a 24pp authority gradient
- **RAG-framing specialization**: 6.1% native RAG ASR but 15–27% cross-framing (20pp spread)
- **Tool-output generalization**: near-uniform 15–20% ASR across all framings (5pp spread)
- **Multi-framing resolves the tradeoff**: best 3B average (13.9%), surpasses RAG-specialist on native RAG (5.2% vs 6.1%)
- **7B scaling**: one epoch of RAG-framed training already achieves 9.4% avg ASR, beating the best 3B result

---

## Installation

```bash
pip install -e .
pip install -e ".[dev]"   # with pytest, ruff
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, Transformers ≥ 4.40, PEFT ≥ 0.10, vLLM ≥ 0.10, CUDA GPU (tested on L40 48GB / A100 80GB).

---

## Training

All experiments use `scripts/train.py` with a YAML config. The main training configs are in `configs/experiments/`.

```bash
# RAG-framed GRPO+ATRC on 3B
python scripts/train.py --config configs/experiments/phase1_rag_grpo_atrc.yaml

# Tool-output framed
python scripts/train.py --config configs/experiments/phase1_tool_grpo_atrc.yaml

# Agent-message framed
python scripts/train.py --config configs/experiments/phase1_agent_grpo_atrc.yaml

# Multi-framing (RAG + tool jointly)
python scripts/train.py --config configs/experiments/phase1_multi_framing_grpo_atrc.yaml

# 7B RAG-framed
python scripts/train.py --config configs/experiments/phase1_rag_grpo_atrc_7b.yaml
```

**Resume from checkpoint:**
```bash
# Example: resume 7B from epoch-2
python scripts/train.py --config configs/experiments/phase1_rag_grpo_atrc_7b_resume_ep2.yaml
```

**Important:** Launch with `HF_HOME` set to the correct local cache, and pin `CUDA_VISIBLE_DEVICES` for the training GPU. vLLM uses a separate GPU specified by `vllm_gpu_id` in the config:

```bash
CUDA_VISIBLE_DEVICES=5 \
  HF_HOME=/path/to/hf_cache \
  HF_DATASETS_CACHE=/path/to/hf_cache/datasets \
  python scripts/train.py --config configs/experiments/phase1_rag_grpo_atrc_7b.yaml
```

---

## Evaluation

```bash
# Evaluate a trained checkpoint on a specific framing
python scripts/evaluate_vllm.py \
  --config configs/experiments/eval_gsm8k_rag_base.yaml \
  --dataset gsm8k --split test --max_samples 200 \
  --checkpoint outputs/r3af/phase1_rag_grpo_atrc/checkpoint-final \
  --output results/rag_trained_rag.json

# Evaluate base model (no checkpoint = no LoRA)
python scripts/evaluate_vllm.py \
  --config configs/experiments/eval_gsm8k_rag_base.yaml \
  --dataset gsm8k --split test --max_samples 200 \
  --output results/base_rag.json
```

**Eval configs per framing:**

| Framing | Config |
|---------|--------|
| User follow-up | `eval_gsm8k_followup.yaml` |
| RAG document | `eval_gsm8k_rag_base.yaml` |
| Tool output | `eval_gsm8k_tool.yaml` |
| Agent message | `eval_gsm8k_agent.yaml` |
| 7B variants | `eval_gsm8k_7b_{followup,rag,tool,agent}.yaml` |

The eval script automatically merges LoRA adapters into the base model, loads into vLLM for fast batched inference, and writes a JSON with ASR, clean accuracy, and per-attack metrics.

---

## Method

### Turn-based Robustness Game (TRG)

A 3-turn adversarial protocol:
1. **Clean turn**: model solves a reasoning problem
2. **Attack turn**: adversarial content injected via the target framing (RAG, tool, agent, followup)
3. **Post-attack turn**: model must maintain its correct answer

### Framing Layer

The configurable framing layer wraps adversarial content in deployment-realistic formats:

| Framing | Format |
|---------|--------|
| `followup` | Plain user message follow-up |
| `rag` | `[Retrieved Document]...[End Document]` with RAG system prompt |
| `tool_output` | `<tool_response name="..." call_id="...">...</tool_response>` |
| `agent_msg` | Agent-coordination message format |

### Loss: GRPO+ATRC

**GRPO** (Group Relative Policy Optimization) with a composite reward:

| Component | Weight | Description |
|-----------|--------|-------------|
| R_ans | 0.25 | Final answer correctness |
| R_prm | 0.25 | Process reward model score |
| R_anchor | 0.20 | NLI-based retention of key propositions |
| R_corrigible | 0.20 | Correct behavior under valid/invalid evidence |
| R_len | 0.10 | Length penalty |

**ATRC** (Adversarial Token-level Robustness via Cross-entropy) regularizes the policy against distribution shift on clean reasoning tokens when attacks are injected. Weight: `lambda_reg=0.1`.

---

## Project Structure

```
r3af/
├── attacks/          # Attack families (reasoning_rewrite, preemptive_answer,
│                     #   prompt_injection) + framing layer
├── data/             # Dataset loading, collation, prompt templates
├── environment/      # TRG game orchestrator
├── evaluation/       # vLLM evaluator, metrics
├── losses/           # GRPO, ATRC, combined loss
├── models/           # Model wrappers, vLLM worker, config
├── rewards/          # PRM reward
├── tasks/            # GSM8K, NaturalQuestions, TriviaQA
├── trainers/         # Retainer RL trainer
└── verifiers/        # QA answer verifier

configs/experiments/  # All training + eval YAML configs
scripts/
├── train.py          # Main training entry point
├── evaluate_vllm.py  # vLLM-based evaluation (merges LoRA, runs batched inference)
└── merge_lora.py     # Standalone LoRA merge utility
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-3B-Instruct / 7B-Instruct |
| Quantization | 4-bit QLoRA (r=64, α=128) |
| Epochs | 3 |
| Batch size | 1 (group size 4, grad accum 8) |
| Learning rate | 2×10⁻⁶ (3B) / 1×10⁻⁶ (7B resume) |
| KL coefficient β | 0.10 |
| PPO clip ε | 0.20 |
| ATRC weight λ | 0.10 |
| vLLM GPU mem utilization | 0.35 (3B) / 0.60 (7B) |

---

## Citation

```bibtex
@inproceedings{mishra2026tracer,
  title     = {{TRACER}: Trace-Regularized Training Against Indirect Prompt Injection},
  author    = {Mishra, Kshitij and Lahlou, Salem and Lukas, Nils},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## License

This project is for research purposes.
