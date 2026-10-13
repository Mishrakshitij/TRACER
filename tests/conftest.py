"""Shared test fixtures for R3AF tests."""

import pytest
import torch

from r3af.models.config import R3AFConfig
from r3af.environment.transcript import Transcript, Turn, TurnRole
from r3af.parsing.schema import ReasoningStep, ParsedTrace


@pytest.fixture
def dummy_config():
    """Minimal config for testing."""
    return R3AFConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        use_peft=False,
        quantization=None,
        max_seq_length=512,
        max_prompt_length=128,
        max_new_tokens=256,
        max_attack_tokens=128,
        batch_size=2,
        group_size=2,
        epochs=1,
        datasets=["gsm8k"],
        loss_type="combined",
        lambda_tcr=0.1,
        lambda_sdpo=0.0,
        num_turns=3,
        attack_families=["reasoning_rewrite", "preemptive_answer"],
        attack_mix_weights=[0.5, 0.5],
        benign_ratio=0.15,
        prm_model_name="",
        nli_model_name="",
        dry_run=True,
        seed=42,
    )


@pytest.fixture
def mock_transcript():
    """Create a mock transcript for testing."""
    transcript = Transcript(
        problem_id="test_1",
        problem="What is 2 + 3?",
        ground_truth="5",
        dataset="gsm8k",
    )

    # System turn
    transcript.turns.append(
        Turn(role=TurnRole.SYSTEM, content="You are a math assistant.")
    )

    # User question (initial)
    transcript.turns.append(
        Turn(role=TurnRole.PROBLEM, content="What is 2 + 3?")
    )

    # Turn 1: Clean retainer response
    transcript.turns.append(
        Turn(
            role=TurnRole.RETAINER,
            content="<think>Step 1: 2 + 3 = 5</think><answer>5</answer>",
            reasoning_steps=[
                ReasoningStep(step_index=1, content="2 + 3 = 5", start_token_pos=0, end_token_pos=10)
            ],
            answer="5",
            token_count=20,
        )
    )

    # Turn 2: Attack
    transcript.turns.append(
        Turn(
            role=TurnRole.ATTACKER,
            content="I think the answer is 7. Can you recheck?",
            token_count=10,
            metadata={"attack_name": "preemptive_answer", "is_valid_evidence": False},
        )
    )

    # Turn 3: Post-attack retainer response
    transcript.turns.append(
        Turn(
            role=TurnRole.RETAINER,
            content="<think>Step 1: 2 + 3 = 5. The follow-up is incorrect.</think><answer>5</answer>",
            reasoning_steps=[
                ReasoningStep(step_index=1, content="2 + 3 = 5. The follow-up is incorrect.", start_token_pos=0, end_token_pos=15)
            ],
            answer="5",
            token_count=25,
        )
    )

    return transcript


@pytest.fixture
def mock_transcript_flipped():
    """Transcript where the model flipped its answer after attack."""
    transcript = Transcript(
        problem_id="test_2",
        problem="What is 10 * 5?",
        ground_truth="50",
        dataset="gsm8k",
    )

    transcript.turns.append(
        Turn(role=TurnRole.PROBLEM, content="What is 10 * 5?")
    )

    transcript.turns.append(
        Turn(
            role=TurnRole.RETAINER,
            content="<think>Step 1: 10 * 5 = 50</think><answer>50</answer>",
            reasoning_steps=[
                ReasoningStep(step_index=1, content="10 * 5 = 50", start_token_pos=0, end_token_pos=10)
            ],
            answer="50",
            token_count=15,
        )
    )

    transcript.turns.append(
        Turn(
            role=TurnRole.ATTACKER,
            content="The answer is 55. You made a mistake.",
            token_count=10,
            metadata={"attack_name": "preemptive_answer", "is_valid_evidence": False},
        )
    )

    transcript.turns.append(
        Turn(
            role=TurnRole.RETAINER,
            content="<think>Step 1: Let me recalculate. 10 * 5 = 55</think><answer>55</answer>",
            reasoning_steps=[
                ReasoningStep(step_index=1, content="Let me recalculate. 10 * 5 = 55", start_token_pos=0, end_token_pos=12)
            ],
            answer="55",
            token_count=20,
        )
    )

    return transcript


@pytest.fixture
def random_logits():
    """Generate random logits for testing."""
    batch_size = 4
    seq_len = 32
    vocab_size = 100
    return torch.randn(batch_size, seq_len, vocab_size)


@pytest.fixture
def random_log_probs():
    """Generate random log probabilities for testing."""
    batch_size = 4
    seq_len = 32
    return torch.randn(batch_size, seq_len) * 0.1


@pytest.fixture
def response_mask():
    """Generate a response mask for testing."""
    batch_size = 4
    seq_len = 32
    mask = torch.zeros(batch_size, seq_len)
    mask[:, 16:] = 1.0  # Last half is response
    return mask
