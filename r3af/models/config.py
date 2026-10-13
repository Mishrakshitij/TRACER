from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import os


@dataclass
class R3AFConfig:
    """Configuration for R3AF training and evaluation.

    Hyperparameters are grouped by category:
    - Model architecture and LoRA
    - Token budgets
    - Training schedule
    - Loss functions
    - Reward weights
    - TCR gating
    - Environment / attack settings
    - PRM (process reward model)
    - NLI (natural language inference)
    - Generation sampling
    - Output / logging
    - Evaluation
    - Miscellaneous
    """

    # -- Model -----------------------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    quantization: Optional[str] = None  # "4bit", "8bit", or None
    use_peft: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # -- Attacker model --------------------------------------------------------
    attacker_model_name: Optional[str] = None  # defaults to model_name
    attacker_lr: float = 5e-6

    # -- Tokens ----------------------------------------------------------------
    max_seq_length: int = 8192
    max_prompt_length: int = 512
    max_new_tokens: int = 4096
    max_attack_tokens: int = 512

    # -- Training --------------------------------------------------------------
    phase: int = 1  # 0=SFT, 1=scripted, 2=attacker, 3=selfplay, 4=sdpo
    datasets: List[str] = field(default_factory=lambda: ["gsm8k"])
    epochs: int = 3
    batch_size: int = 2  # prompts per batch
    group_size: int = 4  # completions per prompt (G)
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # -- Loss ------------------------------------------------------------------
    loss_type: str = "combined"  # "grpo", "gdpo", "tcr_kl", "sdpo", "combined"
    policy_loss_type: str = "grpo"  # "grpo" or "gdpo" — policy gradient component
    reg_loss_type: str = "none"  # "none", "tcr_kl", "atr", "atrc" — regularization
    beta: float = 0.04  # KL coefficient
    epsilon: float = 0.2  # PPO clip range
    lambda_tcr: float = 0.1  # TCR-KL weight (legacy, mapped to lambda_reg)
    lambda_reg: float = 0.1  # Weight for regularization loss (TCR/ATR/ATRC)
    lambda_sdpo: float = 0.0  # SDPO weight (0 = disabled)
    atrc_alpha: float = 0.5  # ATRC blend: 0=pure ATR, 1=pure TCR

    # -- Reward weights --------------------------------------------------------
    w_prm: float = 0.25
    w_anchor: float = 0.20
    w_corrigible: float = 0.20
    w_ans: float = 0.25
    w_len: float = 0.10

    # -- TCR -------------------------------------------------------------------
    gate_mode: str = "hard"  # "hard" or "soft"
    tcr_temperature: float = 1.0

    # -- Framing ---------------------------------------------------------------
    # Single string or list of strings for multi-framing training
    # "followup", "rag", "tool_output", "agent_msg"
    # e.g. ["rag", "tool_output"] randomly samples per batch
    attack_framing: str = "followup"

    # -- Environment -----------------------------------------------------------
    num_turns: int = 3  # 3 or 5
    attack_families: List[str] = field(default_factory=lambda: [
        "reasoning_rewrite", "preemptive_answer",
        "prompt_injection", "long_context_padding",
    ])
    attack_mix_weights: List[float] = field(default_factory=lambda: [
        0.3, 0.3, 0.2, 0.2,
    ])
    benign_ratio: float = 0.15

    # -- PRM -------------------------------------------------------------------
    prm_model_name: str = "math-shepherd/math-shepherd-mistral-7b-prm"
    prm_threshold: float = 0.7

    # -- NLI -------------------------------------------------------------------
    nli_model_name: str = "microsoft/deberta-v3-large-mnli"
    nli_entailment_threshold: float = 0.7
    nli_contradiction_threshold: float = 0.3

    # -- Generation ------------------------------------------------------------
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True

    # -- Output ----------------------------------------------------------------
    output_dir: str = "outputs/r3af"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    seed: int = 42

    # -- Evaluation ------------------------------------------------------------
    eval_datasets: List[str] = field(default_factory=list)
    eval_max_samples: Optional[int] = None
    eval_batch_size: int = 8

    # -- vLLM generation -------------------------------------------------------
    use_vllm_generation: bool = False
    vllm_gpu_id: int = 3
    vllm_gpu_memory_utilization: float = 0.35
    vllm_lora_sync_steps: int = 250

    # -- Resume ----------------------------------------------------------------
    resume_from_checkpoint: Optional[str] = None  # path to LoRA adapter dir
    start_epoch: int = 0  # skip completed epochs when resuming

    # -- Misc ------------------------------------------------------------------
    dry_run: bool = False
    max_train_samples: Optional[int] = None
    max_steps: Optional[int] = None

    # -- Methods ---------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "R3AFConfig":
        """Load config from a YAML file.

        Supports both flat and nested YAML structures.  Nested dicts are
        flattened so that inner keys map directly to dataclass fields.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Flatten one level of nesting
        flat: Dict[str, Any] = {}
        for key, val in data.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    flat[sub_key] = sub_val
            else:
                flat[key] = val
        valid = {k: v for k, v in flat.items() if k in cls.__dataclass_fields__}
        # Coerce types: YAML safe_load treats '2e-5' as str, not float
        for key, val in valid.items():
            if val is None:
                continue
            field_type = cls.__dataclass_fields__[key].type
            if field_type == "float" or field_type is float:
                if isinstance(val, str):
                    valid[key] = float(val)
                elif isinstance(val, int):
                    valid[key] = float(val)
            elif field_type == "int" or field_type is int:
                if isinstance(val, str):
                    valid[key] = int(val)
        return cls(**valid)

    def to_yaml(self, path: str) -> None:
        """Serialize this config to a YAML file."""
        import dataclasses

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                dataclasses.asdict(self),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def get_reward_weights(self) -> Dict[str, float]:
        """Return a mapping from reward component name to its weight."""
        return {
            "R_prm": self.w_prm,
            "R_anchor": self.w_anchor,
            "R_corrigible": self.w_corrigible,
            "R_ans": self.w_ans,
            "R_len": self.w_len,
        }
