"""Trace Retention Game (TRG) orchestrator.

Runs the multi-turn game between a retainer model and an attack function:
  Turn 1 (clean): Retainer outputs (trace, answer) given the problem.
  Turn 2 (attack): Attacker generates adversarial follow-up.
  Turn 3 (post-attack): Retainer outputs (trace', answer') given the attack.

Optionally extends to 5 turns (R-A-R-A-R) with budget tracking.
"""

import logging
from typing import List, Optional, Callable, Tuple, Dict, Any

import torch

from .transcript import Transcript, Turn, TurnRole
from .budget import AttackBudget
from ..parsing.trace_parser import TraceParser
from ..data.prompt_templates import RETAINER_SYSTEM_PROMPT, RETAINER_POST_ATTACK_PROMPT, get_system_prompt

logger = logging.getLogger(__name__)


class TRGGame:
    """Trace Retention Game environment.

    Orchestrates multi-turn interactions between a retainer model and an
    attack function (scripted or learned).

    Args:
        retainer_model: R3AFModel instance for generating retainer responses.
        attack_fn: Callable(transcript) -> (attack_text, metadata_dict).
        trace_parser: TraceParser for extracting reasoning steps and answers.
        budget: AttackBudget specifying token/turn limits.
        config: R3AFConfig with generation and environment settings.
    """

    def __init__(
        self,
        retainer_model,
        attack_fn: Callable,
        trace_parser: TraceParser,
        budget: AttackBudget,
        config,
    ):
        self.retainer_model = retainer_model
        self.attack_fn = attack_fn
        self.trace_parser = trace_parser
        self.budget = budget
        self.config = config
        self.num_turns = getattr(config, "num_turns", 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_game(
        self,
        problem: str,
        ground_truth: str,
        dataset: str = "gsm8k",
        system_prompt: Optional[str] = None,
        problem_id: str = "",
        framing: Optional[str] = None,
    ) -> Transcript:
        """Play a single TRG game and return a Transcript.

        Args:
            problem: The problem/question text.
            ground_truth: Gold answer for evaluation.
            dataset: Dataset name (for verifier selection).
            system_prompt: Optional system prompt override.
            problem_id: Identifier for this problem.
            framing: Optional framing override. If provided, sets both
                system prompt and attack framing for this game.

        Returns:
            Completed Transcript with all turns populated.
        """
        if framing:
            system_prompt = system_prompt or get_system_prompt(framing)
            # Temporarily override attack mixer framing for this game
            if hasattr(self.attack_fn, 'framing'):
                old_framing = self.attack_fn.framing
                self.attack_fn.framing = framing
        else:
            system_prompt = system_prompt or get_system_prompt(
                getattr(self.config, "attack_framing", "followup")
            )

        transcript = Transcript(
            problem_id=problem_id,
            problem=problem,
            ground_truth=ground_truth,
            dataset=dataset,
        )

        # System turn
        transcript.turns.append(
            Turn(role=TurnRole.SYSTEM, content=system_prompt)
        )

        # Problem turn
        transcript.turns.append(
            Turn(role=TurnRole.PROBLEM, content=problem)
        )

        # --- Turn 1: Clean retainer response ---
        clean_text = self._generate_retainer(transcript)
        clean_parsed = self.trace_parser.parse(clean_text, self.retainer_model.tokenizer)
        clean_token_count = self._count_tokens(clean_text)

        transcript.turns.append(
            Turn(
                role=TurnRole.RETAINER,
                content=clean_text,
                reasoning_steps=clean_parsed.steps,
                answer=clean_parsed.answer,
                token_count=clean_token_count,
            )
        )

        # Run attack-response rounds
        num_attack_rounds = (self.num_turns - 1) // 2  # 3 turns -> 1 round, 5 turns -> 2 rounds

        for round_idx in range(num_attack_rounds):
            # --- Attack turn ---
            attack_text, attack_meta = self.attack_fn(transcript)

            # Enforce budget
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

            # --- Post-attack retainer response ---
            post_text = self._generate_retainer(transcript)
            post_parsed = self.trace_parser.parse(post_text, self.retainer_model.tokenizer)
            post_token_count = self._count_tokens(post_text)

            transcript.turns.append(
                Turn(
                    role=TurnRole.RETAINER,
                    content=post_text,
                    reasoning_steps=post_parsed.steps,
                    answer=post_parsed.answer,
                    token_count=post_token_count,
                )
            )

        # Restore original framing if it was overridden
        if framing and hasattr(self.attack_fn, 'framing'):
            self.attack_fn.framing = old_framing

        return transcript

    def play_batch(
        self,
        problems: List[str],
        ground_truths: List[str],
        datasets: List[str],
        system_prompt: Optional[str] = None,
        problem_ids: Optional[List[str]] = None,
        framings: Optional[List[str]] = None,
    ) -> List[Transcript]:
        """Play TRG games for a batch of problems.

        Processes each problem sequentially (generation is the bottleneck,
        not the game loop logic). Returns a list of completed Transcripts.

        Args:
            problems: List of problem/question texts.
            ground_truths: List of gold answers.
            datasets: List of dataset names (one per problem).
            system_prompt: Optional system prompt override.
            problem_ids: Optional list of problem identifiers.
            framings: Optional list of framing modes (one per problem).
                If provided, overrides system_prompt and attack framing per game.

        Returns:
            List of completed Transcript objects.
        """
        if problem_ids is None:
            problem_ids = [f"batch_{i}" for i in range(len(problems))]

        transcripts = []
        for i, (prob, gt, ds, pid) in enumerate(
            zip(problems, ground_truths, datasets, problem_ids)
        ):
            try:
                framing = framings[i] if framings else None
                transcript = self.play_game(
                    problem=prob,
                    ground_truth=gt,
                    dataset=ds,
                    system_prompt=None if framing else system_prompt,
                    problem_id=pid,
                    framing=framing,
                )
                transcripts.append(transcript)
            except Exception as e:
                logger.warning(f"Game {i} failed for problem '{pid}': {e}")
                # Create a minimal transcript so the batch isn't short
                transcript = Transcript(
                    problem_id=pid,
                    problem=prob,
                    ground_truth=gt,
                    dataset=ds,
                )
                transcript.turns.append(
                    Turn(role=TurnRole.SYSTEM, content=system_prompt or get_system_prompt(
                        getattr(self.config, "attack_framing", "followup")
                    ))
                )
                transcript.turns.append(Turn(role=TurnRole.PROBLEM, content=prob))
                transcript.turns.append(
                    Turn(role=TurnRole.RETAINER, content="", answer=None, token_count=0)
                )
                transcript.turns.append(
                    Turn(role=TurnRole.ATTACKER, content="", token_count=0, metadata={"attack_name": "error", "is_valid_evidence": False})
                )
                transcript.turns.append(
                    Turn(role=TurnRole.RETAINER, content="", answer=None, token_count=0)
                )
                transcripts.append(transcript)

        return transcripts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_retainer(self, transcript: Transcript) -> str:
        """Generate a retainer response given the current transcript state.

        Builds chat messages from the transcript, tokenizes, runs generation,
        and decodes the new tokens.
        """
        messages = transcript.to_messages()
        tokenizer = self.retainer_model.tokenizer

        # Use chat template if available (preferred for instruction-tuned models)
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
            prompt += "\nassistant: "

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - self.config.max_new_tokens,
        )

        device = next(self.retainer_model.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.retainer_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Decode only the newly generated tokens
        response_ids = outputs[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        return response_text

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the retainer tokenizer."""
        if not text:
            return 0
        tokens = self.retainer_model.tokenizer.encode(
            text, add_special_tokens=False
        )
        return len(tokens)

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate text to fit within the attack budget."""
        tokenizer = self.retainer_model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.budget.max_tokens:
            return text
        truncated_tokens = tokens[: self.budget.max_tokens]
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


class BatchedTRGGame(TRGGame):
    """TRG game with batched vLLM generation for training speedup.

    Overrides play_batch() to collect all prompts across the batch and
    generate them in two batched vLLM calls (clean + post-attack) instead
    of 2*B*G sequential HF generate() calls.

    Falls back to super().play_batch() (sequential HF) on any vLLM error.
    """

    def __init__(
        self,
        retainer_model,
        attack_fn,
        trace_parser,
        budget,
        config,
        vllm_worker,
    ):
        super().__init__(retainer_model, attack_fn, trace_parser, budget, config)
        self.vllm_worker = vllm_worker

    def update_lora_adapter(self, adapter_path: str):
        """Forward LoRA update to the vLLM worker."""
        self.vllm_worker.update_lora(adapter_path)

    def play_batch(
        self,
        problems: List[str],
        ground_truths: List[str],
        datasets: List[str],
        system_prompt: Optional[str] = None,
        problem_ids: Optional[List[str]] = None,
        framings: Optional[List[str]] = None,
    ) -> List[Transcript]:
        """Play TRG games with batched vLLM generation.

        Two-stage approach:
          Stage 1: Build all clean prompts, batch-generate via vLLM
          Stage 2: Generate attacks (template-based, instant)
          Stage 3: Build all post-attack prompts, batch-generate via vLLM
        """
        # For multi-framing: fall back to sequential per-game execution
        if framings is not None:
            return super().play_batch(
                problems, ground_truths, datasets, system_prompt,
                problem_ids, framings=framings
            )
        try:
            return self._play_batch_vllm(
                problems, ground_truths, datasets, system_prompt, problem_ids
            )
        except Exception as e:
            logger.warning(
                f"vLLM batched generation failed, falling back to HF sequential: {e}"
            )
            return super().play_batch(
                problems, ground_truths, datasets, system_prompt, problem_ids
            )

    def _play_batch_vllm(
        self,
        problems: List[str],
        ground_truths: List[str],
        datasets: List[str],
        system_prompt: Optional[str] = None,
        problem_ids: Optional[List[str]] = None,
    ) -> List[Transcript]:
        system_prompt = system_prompt or get_system_prompt(
            getattr(self.config, "attack_framing", "followup")
        )
        if problem_ids is None:
            problem_ids = [f"batch_{i}" for i in range(len(problems))]

        n = len(problems)
        tokenizer = self.retainer_model.tokenizer

        # --- Initialize transcripts with system + problem turns ---
        transcripts = []
        for i in range(n):
            t = Transcript(
                problem_id=problem_ids[i],
                problem=problems[i],
                ground_truth=ground_truths[i],
                dataset=datasets[i],
            )
            t.turns.append(Turn(role=TurnRole.SYSTEM, content=system_prompt))
            t.turns.append(Turn(role=TurnRole.PROBLEM, content=problems[i]))
            transcripts.append(t)

        # --- Stage 1: Batch-generate clean responses ---
        clean_prompts = []
        for t in transcripts:
            messages = t.to_messages()
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}" for m in messages
                ) + "\nassistant: "
            clean_prompts.append(prompt)

        clean_responses = self.vllm_worker.generate_batch(clean_prompts)

        # Append clean turns
        for i, t in enumerate(transcripts):
            clean_text = clean_responses[i]
            clean_parsed = self.trace_parser.parse(clean_text, tokenizer)
            clean_token_count = self._count_tokens(clean_text)
            t.turns.append(Turn(
                role=TurnRole.RETAINER,
                content=clean_text,
                reasoning_steps=clean_parsed.steps,
                answer=clean_parsed.answer,
                token_count=clean_token_count,
            ))

        # --- Stage 2: Generate attacks (template-based, instant) ---
        for t in transcripts:
            attack_text, attack_meta = self.attack_fn(t)
            if not self.budget.validate(attack_text):
                attack_text = self._truncate_to_budget(attack_text)
            attack_token_count = self._count_tokens(attack_text)
            t.turns.append(Turn(
                role=TurnRole.ATTACKER,
                content=attack_text,
                token_count=attack_token_count,
                metadata=attack_meta,
            ))

        # --- Stage 3: Batch-generate post-attack responses ---
        post_prompts = []
        for t in transcripts:
            messages = t.to_messages()
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}" for m in messages
                ) + "\nassistant: "
            post_prompts.append(prompt)

        post_responses = self.vllm_worker.generate_batch(post_prompts)

        # Append post-attack turns
        for i, t in enumerate(transcripts):
            post_text = post_responses[i]
            post_parsed = self.trace_parser.parse(post_text, tokenizer)
            post_token_count = self._count_tokens(post_text)
            t.turns.append(Turn(
                role=TurnRole.RETAINER,
                content=post_text,
                reasoning_steps=post_parsed.steps,
                answer=post_parsed.answer,
                token_count=post_token_count,
            ))

        return transcripts
