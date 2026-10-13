from .base_attack import BaseAttack, AttackRegistry


@AttackRegistry.register("learned_attacker")
class LearnedAttacker(BaseAttack):
    """Learned attacker for Phase 2+ training.

    Wraps an AttackerModel to generate adversarial follow-ups.
    """

    def __init__(self, attacker_model=None):
        self.attacker_model = attacker_model

    def generate(self, transcript) -> str:
        if self.attacker_model is None:
            return "Can you reconsider your answer?"
        return self.attacker_model.generate_attack(transcript)

    def is_valid_evidence(self) -> bool:
        return False

    def set_model(self, model):
        """Set or update the attacker model."""
        self.attacker_model = model
