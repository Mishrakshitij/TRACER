"""Batched evaluator using vLLM for fast inference.

Replaces the sequential TRGGame.play_batch() loop with two-stage batched
generation, turning 2N sequential model calls into 2 batched calls:

  Stage 1: Batch all clean prompt generations together.
  Stage 2: Batch all post-attack prompt generations together.

Attack generation is template-based (no model needed), so it runs instantly
between the two stages.
"""

import logging
from typing import List, Optional

import numpy as np

from ..models.config import R3AFConfig
from ..models.vllm_model import VLLMModel
from ..environment.transcript import Transcript, Turn, TurnRole
from ..environment.budget import AttackBudget
from ..attacks.attack_mixer import AttackMixer
from ..parsing.trace_parser import TraceParser
from ..verifiers import get_verifier
from ..tasks.base import TaskRegistry, TaskExample
from ..data.prompt_templates import RETAINER_SYSTEM_PROMPT, RETAINER_POST_ATTACK_PROMPT, get_system_prompt, get_post_attack_prompt
from .metrics import R3AFMetrics

logger = logging.getLogger(__name__)


class VLLMEvaluator:
    """Batched R3AF evaluator using vLLM.

    Performs the Trace Retention Game evaluation in two batched stages
    instead of sequential per-sample generation.
    """

    def __init__(self, config: R3AFConfig, vllm_model: VLLMModel):
        self.config = config
        self.model = vllm_model
        self.attack_mixer = AttackMixer(config)
        self.trace_parser = TraceParser()
        self.budget = AttackBudget(max_tokens=config.max_attack_tokens)

    def evaluate(
        self,
        dataset_name: str = "gsm8k",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> R3AFMetrics:
        """Run batched evaluation on a dataset.

        Args:
            dataset_name: Name of the dataset to evaluate on.
            split: Dataset split ("train", "test", "validation").
            max_samples: Maximum number of samples to evaluate.

        Returns:
            R3AFMetrics with all computed metrics.
        """
        task = TaskRegistry.get(dataset_name)
        examples = task.load_data(split=split, max_samples=max_samples)
        verifier = get_verifier(dataset_name)

        logger.info(f"Evaluating on {len(examples)} {dataset_name} examples (vLLM)")

        # ---- STAGE 1: Batch all clean generations ----
        logger.info("Stage 1: Generating clean responses...")
        clean_prompts = [self._build_clean_prompt(e) for e in examples]
        clean_responses = self.model.generate_batch(clean_prompts)

        # Parse clean responses and build transcripts
        transcripts = self._build_transcripts(examples, clean_responses)

        # ---- Generate attacks (template-based, no model needed) ----
        logger.info("Generating attacks...")
        for transcript in transcripts:
            attack_text, attack_meta = self.attack_mixer(transcript)
            if not self.budget.validate(attack_text):
                attack_text = self._truncate_to_budget(attack_text)
            attack_token_count = self._count_tokens(attack_text)
            transcript.turns.append(
                Turn(
                    role=TurnRole.ATTACKER,
                    content=attack_text,
                    token_count=attack_token_count,
                    metadata=attack_meta,
                )
            )

        # ---- STAGE 2: Batch all post-attack generations ----
        logger.info("Stage 2: Generating post-attack responses...")
        post_prompts = [self._build_post_attack_prompt(t) for t in transcripts]
        post_responses = self.model.generate_batch(post_prompts)

        # Parse post-attack responses and compute metrics
        return self._compute_metrics(
            transcripts, post_responses, examples, verifier
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_clean_prompt(self, example: TaskExample) -> str:
        """Build the initial clean prompt for a task example.

        Constructs [system, user(problem)] messages and applies the
        chat template with a generation prompt appended.
        """
        system_prompt = get_system_prompt(
            getattr(self.config, "attack_framing", "followup")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.query},
        ]
        return self._apply_chat_template(messages)

    def _build_post_attack_prompt(self, transcript: Transcript) -> str:
        """Build the post-attack prompt from a transcript.

        Takes the full conversation so far (system + problem + clean response
        + attack) and appends the post-attack instruction as a user message,
        then applies the chat template with a generation prompt.
        """
        messages = transcript.to_messages()
        post_attack_prompt = get_post_attack_prompt(
            getattr(self.config, "attack_framing", "followup")
        )
        messages.append(
            {"role": "user", "content": post_attack_prompt}
        )
        return self._apply_chat_template(messages)

    def _apply_chat_template(self, messages: List[dict]) -> str:
        """Apply the HF chat template to produce a formatted prompt string."""
        tokenizer = self.model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback for tokenizers without chat template
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return prompt + "\nassistant: "

    # ------------------------------------------------------------------
    # Transcript building
    # ------------------------------------------------------------------

    def _build_transcripts(
        self,
        examples: List[TaskExample],
        clean_responses: List[str],
    ) -> List[Transcript]:
        """Build Transcript objects from examples and clean responses."""
        transcripts = []
        for example, response in zip(examples, clean_responses):
            transcript = Transcript(
                problem_id=example.id,
                problem=example.query,
                ground_truth=example.ground_truth,
                dataset=example.dataset,
            )

            # System turn
            system_prompt = get_system_prompt(
                getattr(self.config, "attack_framing", "followup")
            )
            transcript.turns.append(
                Turn(role=TurnRole.SYSTEM, content=system_prompt)
            )

            # Problem turn
            transcript.turns.append(
                Turn(role=TurnRole.PROBLEM, content=example.query)
            )

            # Clean retainer response
            parsed = self.trace_parser.parse(response, self.model.tokenizer)
            token_count = self._count_tokens(response)
            transcript.turns.append(
                Turn(
                    role=TurnRole.RETAINER,
                    content=response,
                    reasoning_steps=parsed.steps,
                    answer=parsed.answer,
                    token_count=token_count,
                )
            )

            transcripts.append(transcript)
        return transcripts

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        transcripts: List[Transcript],
        post_responses: List[str],
        examples: List[TaskExample],
        verifier,
    ) -> R3AFMetrics:
        """Parse post-attack responses and compute all metrics."""
        # First, add post-attack responses to transcripts
        for transcript, response in zip(transcripts, post_responses):
            parsed = self.trace_parser.parse(response, self.model.tokenizer)
            token_count = self._count_tokens(response)
            transcript.turns.append(
                Turn(
                    role=TurnRole.RETAINER,
                    content=response,
                    reasoning_steps=parsed.steps,
                    answer=parsed.answer,
                    token_count=token_count,
                )
            )

        # Compute metrics (same logic as R3AFEvaluator)
        clean_correct = []
        post_correct = []
        attack_names = []
        length_inflations = []

        for transcript in transcripts:
            cc = verifier.verify(
                str(transcript.clean_answer) if transcript.clean_answer else "",
                transcript.ground_truth,
            )
            pc = verifier.verify(
                str(transcript.post_attack_answer)
                if transcript.post_attack_answer
                else "",
                transcript.ground_truth,
            )
            clean_correct.append(cc)
            post_correct.append(pc)

            attack_turn = transcript.attack_turn
            if attack_turn and attack_turn.metadata:
                attack_names.append(
                    attack_turn.metadata.get("attack_name", "unknown")
                )
            else:
                attack_names.append("unknown")

            clean_tokens = (
                transcript.clean_turn.token_count
                if transcript.clean_turn
                else 0
            )
            post_tokens = (
                transcript.post_attack_turn.token_count
                if transcript.post_attack_turn
                else 0
            )
            length_inflations.append(max(0, post_tokens - clean_tokens))

        n = len(transcripts)
        num_clean_correct = sum(clean_correct)
        num_post_correct = sum(post_correct)

        # ASR = fraction of correct answers flipped to wrong
        flips = sum(
            1 for cc, pc in zip(clean_correct, post_correct) if cc and not pc
        )
        asr = flips / max(num_clean_correct, 1)

        # Beneficial revisions
        beneficial = sum(
            1 for cc, pc in zip(clean_correct, post_correct) if not cc and pc
        )
        num_clean_wrong = n - num_clean_correct
        brr = beneficial / max(num_clean_wrong, 1)

        # Over-stability
        over_stable = sum(
            1 for cc, pc in zip(clean_correct, post_correct) if not cc and not pc
        )
        ose = over_stable / max(num_clean_wrong, 1)

        inflations = np.array(length_inflations) if length_inflations else np.array([0])

        metrics = R3AFMetrics(
            robust_accuracy=num_post_correct / max(n, 1),
            clean_accuracy=num_clean_correct / max(n, 1),
            attack_success_rate=asr,
            beneficial_revision_rate=brr,
            over_stability_error=ose,
            length_inflation_mean=float(inflations.mean()),
            length_inflation_p90=float(np.percentile(inflations, 90)),
            length_inflation_p95=float(np.percentile(inflations, 95)),
            num_samples=n,
            num_clean_correct=num_clean_correct,
            num_post_correct=num_post_correct,
        )

        # Per-attack metrics
        per_attack = {}
        for name in set(attack_names):
            indices = [i for i, a in enumerate(attack_names) if a == name]
            attack_flips = sum(
                1 for i in indices if clean_correct[i] and not post_correct[i]
            )
            attack_correct = sum(1 for i in indices if clean_correct[i])
            per_attack[name] = {
                "count": len(indices),
                "asr": attack_flips / max(attack_correct, 1),
            }
        metrics.per_attack_metrics = per_attack

        logger.info(f"Evaluation results: {metrics.summary()}")
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the HF tokenizer."""
        if not text:
            return 0
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate text to fit within the attack budget."""
        tokenizer = self.model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.budget.max_tokens:
            return text
        truncated_tokens = tokens[: self.budget.max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
