import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class R3AFModel(nn.Module):
    """R3AF model wrapper.

    Handles:
    - Loading HF model with optional quantization (4bit/8bit via BitsAndBytes)
    - Applying LoRA via PEFT
    - Reference model via adapter disable/enable
    - Generation, log probability computation, and full logit extraction
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }

        if config.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif config.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            model_kwargs["device_map"] = "auto"

        logger.info(f"Loading model: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **model_kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare quantized models for training (enables input gradients)
        if config.quantization in ("4bit", "8bit"):
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True
            )

        if config.use_peft:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            self._has_peft = True
        else:
            self._has_peft = False

        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Generate completions. Returns full sequences [batch, prompt_len + gen_len]."""
        defaults = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        defaults.update(gen_kwargs)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **defaults,
        )

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for response tokens.

        Returns:
            log_probs: [batch, seq_len] masked to response positions.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab]

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = response_mask[:, 1:]

        # Memory-efficient: use cross_entropy (fuses log_softmax + gather)
        # instead of materializing full [batch, seq_len, vocab] log_softmax
        log_probs = -torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_logits.size(0), -1)

        log_probs = log_probs * shift_mask

        log_probs = torch.cat(
            [
                torch.zeros(
                    log_probs.shape[0],
                    1,
                    device=log_probs.device,
                    dtype=log_probs.dtype,
                ),
                log_probs,
            ],
            dim=1,
        )
        return log_probs

    def compute_ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reference model log probs by disabling LoRA adapters."""
        if self._has_peft:
            self.model.disable_adapter_layers()
            try:
                with torch.no_grad():
                    ref_log_probs = self.compute_log_probs(
                        input_ids, attention_mask, response_mask
                    )
            finally:
                self.model.enable_adapter_layers()
            return ref_log_probs
        else:
            with torch.no_grad():
                return self.compute_log_probs(
                    input_ids, attention_mask, response_mask
                )

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get full logits for TCR-KL loss computation.

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import os

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if self._has_peft:
            self.model.load_adapter(path, adapter_name="default")
        else:
            import os

            state_dict = torch.load(
                os.path.join(path, "pytorch_model.bin"), map_location="cpu"
            )
            self.model.load_state_dict(state_dict)
        logger.info(f"Checkpoint loaded from {path}")
