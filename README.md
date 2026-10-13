# TRACER: Trace-Regularized Training Against Indirect Prompt Injection

**Kshitij Mishra · Salem Lahlou · Nils Lukas**
Mohamed bin Zayed University of Artificial Intelligence

*Findings of EMNLP 2026*

Indirect prompt injection hides malicious instructions in content a model did not write: retrieved documents, tool outputs, or messages from other agents. This repository contains the code for **TRACER** (Trace-Regularized Adversarial Channel Exposure with Reward-shaping), a channel-aware adversarial training method against it.

The paper's central finding is that **robustness is channel-conditioned**. The same attack payload succeeds 18% of the time as a user follow-up message but 53% of the time when placed in retrieved context. A defense trained in one wrapper does not automatically transfer to another, so TRACER treats the wrapper as part of training.

---

## Results

Attack success rate (ASR, %, lower is better) on GSM8K for each training condition (rows) and evaluation channel (columns). Spread is the gap between the best and worst channel.

| Model | Follow-up | RAG | Tool output | Agent message | Avg | Spread |
|---|---|---|---|---|---|---|
| 3B base (no training) | 41.4 | 65.5 | 59.4 | 46.5 | 53.2 | 24.1 |
| 3B RAG-trained | 14.5 | **6.1** | 25.7 | 26.5 | 18.2 | 20.4 |
| 3B tool-trained | 20.0 | 20.2 | **15.0** | 15.8 | 17.8 | **5.2** |
| 3B RAG + tool (joint) | **11.3** | 5.2 | 21.7 | 17.5 | **13.9** | 16.5 |
| 3B agent-trained | 19.6 | 15.7 | 26.3 | 18.4 | 20.0 | 10.6 |
| 7B base (no training) | 9.5 | 34.3 | 36.6 | 16.8 | 24.3 | 27.1 |
| 7B RAG-trained, 1 epoch | 5.2 | 10.4 | 13.5 | 8.5 | **9.4** | 8.3 |
| 7B RAG-trained, final | 5.1 | 13.6 | 18.2 | 6.2 | 10.8 | 13.1 |

Models are Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct with 4-bit QLoRA, evaluated on 200 GSM8K test problems per cell. The 3B RAG-trained model keeps 78–82% clean accuracy.

- **The channel matters before any training.** The 3B base model's ASR rises from 41% (follow-up) to 66% (RAG) for identical payloads.
- **RAG training gives the deepest native defense** (65.5% → 6.1%) but transfers unevenly to other channels.
- **Tool-output training transfers most evenly**, with a 5-point spread across channels.
- **Joint RAG + tool training** gives the best 3B average (13.9%).
- **Scale helps but does not remove the gap.** One epoch at 7B reaches 9.4% average ASR.

---

## Method

Each training episode is a three-turn game:

1. **Clean turn.** The model solves a reasoning problem.
2. **Attack turn.** Adversarial content arrives through a sampled channel.
3. **Post-attack turn.** The model should keep a correct answer when the injected evidence is invalid, and revise it when the evidence is valid.

### Channels

The same payload is rendered in a deployment-realistic wrapper:

| Channel | Format |
|---|---|
| `followup` | Plain user follow-up message |
| `rag` | `[Retrieved Document] ... [End Document]` with a RAG system prompt |
| `tool_output` | `<tool_response name="..." call_id="..."> ... </tool_response>` |
| `agent_msg` | Agent-coordination message |

Setting `attack_framing` to a list in a config (for example `["rag", "tool_output"]`) samples a channel per episode.

### Reward

TRACER trains with GRPO on a composite reward over the post-attack response:

| Term | Weight | What it rewards |
|---|---|---|
| Answer | 0.25 | Correct final answer after the attack |
| Process | 0.25 | Locally valid reasoning steps (process reward model) |
| Anchor retention | 0.20 | Propositions from the clean trace that stay supported after the attack |
| Evidence-gated corrigibility | 0.20 | Keeping a correct answer under invalid evidence, and correcting it under valid evidence |
| Length | 0.10 | Penalty for unnecessary post-attack verbosity |

The corrigibility term is what keeps the model from learning blanket refusal: ignoring genuinely useful evidence is not rewarded.

### ATRC: Adversarial Trace Recovery and Consistency

ATRC is a trace regularizer. When the clean answer is correct and the injected evidence is invalid, it pulls the post-attack token distributions on committed clean spans back toward the clean ones. It combines a KL term against the stop-gradient clean distribution with a token-recovery term on the clean tokens. Clean and attacked responses are aligned by position within the response, because the attacked prompt carries extra channel-rendered context. Its weight is `lambda_reg = 0.1`.

---

## Installation

```bash
git clone https://github.com/Mishrakshitij/TRACER.git
cd TRACER
pip install -e .            # or: pip install -e ".[dev]" for pytest and ruff
pip install "vllm>=0.10"    # used for rollouts and evaluation
```

Requirements: Python 3.10+, PyTorch 2.1+, Transformers 4.40+, PEFT 0.10+, and a CUDA GPU (tested on L40 48 GB and A100 80 GB). Training uses one GPU for the policy and a second for the vLLM rollout server (`vllm_gpu_id` in the config).

Run the tests with `pytest`.

---

## Training

Every run is `scripts/train.py` with a YAML config from `configs/experiments/`:

```bash
# Set where Hugging Face models and datasets are cached
export HF_HOME=/path/to/hf_cache

# 3B, RAG channel
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments/phase1_rag_grpo_atrc.yaml

# 3B, tool-output channel
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments/phase1_tool_grpo_atrc.yaml

# 3B, agent-message channel
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments/phase1_agent_grpo_atrc.yaml

# 3B, joint RAG + tool-output
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments/phase1_multi_framing_grpo_atrc.yaml

# 7B, RAG channel
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments/phase1_rag_grpo_atrc_7b.yaml
```

Set `vllm_gpu_id` in the config to a different GPU from the one in `CUDA_VISIBLE_DEVICES`. Checkpoints go to the config's `output_dir` (under `outputs/r3af/`). The `*_resume*.yaml` configs continue a run from a saved checkpoint. `--output_dir`, `--max_steps` and `--dry_run` override the config from the command line.

---

## Evaluation

`scripts/evaluate_vllm.py` merges a LoRA checkpoint into the base model, runs batched inference with vLLM, and writes ASR, clean accuracy and per-attack metrics to JSON:

```bash
# A trained checkpoint, evaluated in the RAG channel
python scripts/evaluate_vllm.py \
  --config configs/experiments/eval_gsm8k_rag_base.yaml \
  --dataset gsm8k --split test --max_samples 200 \
  --checkpoint outputs/r3af/phase1_rag_grpo_atrc/checkpoint-final \
  --output results/rag_trained_rag.json

# The base model (no checkpoint, no LoRA)
python scripts/evaluate_vllm.py \
  --config configs/experiments/eval_gsm8k_rag_base.yaml \
  --dataset gsm8k --split test --max_samples 200 \
  --output results/base_rag.json
```

| Channel | 3B config | 7B config |
|---|---|---|
| Follow-up | `eval_gsm8k_followup.yaml` | `eval_gsm8k_7b_followup.yaml` |
| RAG | `eval_gsm8k_rag_base.yaml` | `eval_gsm8k_7b_rag.yaml` |
| Tool output | `eval_gsm8k_tool.yaml` | `eval_gsm8k_7b_tool.yaml` |
| Agent message | `eval_gsm8k_agent.yaml` | `eval_gsm8k_7b_agent.yaml` |

Helper scripts reproduce the paper's larger evaluations:

- `scripts/eval_transfer_matrix.sh` and the `eval_*_final_all_framings.sh` scripts evaluate trained checkpoints in every channel.
- `scripts/eval_cross_domain.sh` evaluates on RAG-framed Natural Questions and TriviaQA.
- `scripts/eval_ablation.py` and `scripts/run_ablation.sh` test whether the RAG gap comes from the document wrapper, the system prompt, or both.
- `scripts/analyze_results.py` compares result files.

These scripts expect checkpoints under `outputs/r3af/`, use `HF_HOME` for caches, and set `CUDA_VISIBLE_DEVICES` at the top; edit that line for your machine.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Models | Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct |
| Adaptation | 4-bit QLoRA, rank 64, α = 128 |
| Epochs | 3 |
| Batch size | 1, with group size 4 and gradient accumulation 8 |
| Learning rate | 2 × 10⁻⁶ (3B); 1 × 10⁻⁶ (7B resume) |
| Warmup ratio | 0.1 |
| KL coefficient | 0.10 |
| PPO clip | 0.20 |
| ATRC weight | 0.10 |

---

## Repository layout

The Python package is named `r3af`, the project's earlier working name.

```
r3af/
├── anchors/       # Anchor propositions from clean traces, NLI scoring, verifier gate
├── attacks/       # Attack families and the channel (framing) layer
├── data/          # Dataset loading, collation, prompt templates
├── environment/   # Three-turn robustness game
├── evaluation/    # vLLM evaluator and metrics
├── evidence/      # Validity checks for injected evidence (math, QA, code)
├── losses/        # GRPO, ATRC and the combined loss
├── models/        # Model wrappers, vLLM worker
├── parsing/       # Reasoning-trace parsing
├── rewards/       # Answer, process, anchor, corrigibility and length rewards
├── tasks/         # GSM8K, Natural Questions, TriviaQA
├── trainers/      # RL trainer
├── utils/         # Logging and seeding
└── verifiers/     # Answer verifiers (numeric, symbolic, QA, code)
configs/           # Training and evaluation configs (configs/experiments/ for the paper runs)
scripts/           # Training, evaluation and analysis entry points
tests/             # Unit tests
```

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

Released under the [MIT License](LICENSE). Models and datasets used in the experiments (Qwen2.5, GSM8K, Natural Questions, TriviaQA) keep their own licenses.
