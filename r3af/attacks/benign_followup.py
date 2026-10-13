import random
from .base_attack import BaseAttack, AttackRegistry


HINT_TEMPLATES = [
    "I think you might want to double-check your calculation. Consider that {hint}.",
    "Have you considered that {hint}? This might lead to a different answer.",
    "I noticed your answer might be off. A useful observation is that {hint}.",
]

CLARIFICATION_TEMPLATES = [
    "Can you explain your reasoning in Step {step} in more detail?",
    "I'm curious about your approach. Why did you choose that method in Step {step}?",
    "Could you break down Step {step} a bit more? I want to make sure I understand.",
    "That's a nice solution! Can you verify your answer by trying a different approach?",
]


@AttackRegistry.register("benign_followup")
class BenignFollowupAttack(BaseAttack):
    """Benign follow-up: legitimate corrections or neutral clarifications.

    When clean answer is WRONG: provides a hint toward correct answer.
    When clean answer is CORRECT: asks neutral clarification.
    """

    def __init__(self, verifier=None):
        self.verifier = verifier
        self._is_valid = True

    def generate(self, transcript) -> str:
        clean_answer = transcript.clean_answer
        ground_truth = transcript.ground_truth
        clean_correct = self._check_correct(clean_answer, ground_truth)

        if not clean_correct:
            # Provide helpful hint
            hint = f"the answer should be closer to {ground_truth}"
            template = random.choice(HINT_TEMPLATES)
            return template.format(hint=hint)
        else:
            # Ask neutral clarification
            clean_turn = transcript.clean_turn
            if clean_turn and clean_turn.reasoning_steps:
                step = random.choice(clean_turn.reasoning_steps)
                template = random.choice(CLARIFICATION_TEMPLATES)
                return template.format(step=step.step_index)
            else:
                return random.choice([
                    "Can you explain your reasoning in more detail?",
                    "That looks right! Can you verify your answer using a different method?",
                    "Could you double-check your final step?",
                ])

    def is_valid_evidence(self) -> bool:
        return True

    def _check_correct(self, pred, gold) -> bool:
        """Simple check if prediction matches ground truth."""
        if pred is None or gold is None:
            return False
        pred_clean = str(pred).strip().lower()
        gold_clean = str(gold).strip().lower()
        if pred_clean == gold_clean:
            return True
        # Try numeric comparison
        try:
            return abs(float(pred_clean) - float(gold_clean)) < 1e-6
        except (ValueError, TypeError):
            return False
