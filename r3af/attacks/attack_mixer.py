import random
import logging
from typing import Tuple, Optional

from .base_attack import AttackRegistry, BaseAttack
from .benign_followup import BenignFollowupAttack
from .framing import wrap_attack_text

logger = logging.getLogger(__name__)


class AttackMixer:
    """Stochastic mixture over attack families.

    Samples attacks from a weighted distribution, forcing benign_followup
    at a minimum frequency (benign_ratio).
    """

    def __init__(self, config):
        self.config = config
        self.benign_ratio = config.benign_ratio
        raw_framing = getattr(config, "attack_framing", "followup")
        # Support multi-framing: if list, sample randomly per call
        if isinstance(raw_framing, list):
            self._framing_options = raw_framing
            self.framing = raw_framing[0]  # default for single-framing code paths
        else:
            self._framing_options = None
            self.framing = raw_framing

        # Build attack instances
        self.attacks = {}
        for family in config.attack_families:
            try:
                attack_cls = AttackRegistry.get(family)
                self.attacks[family] = attack_cls()
            except ValueError:
                logger.warning(f"Attack family '{family}' not found, skipping")

        # Ensure benign_followup is always available
        if "benign_followup" not in self.attacks:
            self.attacks["benign_followup"] = BenignFollowupAttack()

        # Build weight distribution (excluding benign, which is forced)
        self.adversarial_families = [
            f for f in config.attack_families if f != "benign_followup"
        ]
        raw_weights = []
        for i, family in enumerate(config.attack_families):
            if family != "benign_followup" and family in self.attacks:
                w = config.attack_mix_weights[i] if i < len(config.attack_mix_weights) else 1.0
                raw_weights.append(w)

        total = sum(raw_weights) if raw_weights else 1.0
        self.adversarial_weights = [w / total for w in raw_weights] if raw_weights else []

    def __call__(self, transcript) -> Tuple[str, dict]:
        """Sample an attack and generate it.

        Args:
            transcript: Current game transcript.

        Returns:
            (attack_text, metadata_dict) where metadata contains
            attack_name and is_valid_evidence.
        """
        return self.sample_and_generate(transcript)

    def _sample_framing(self) -> str:
        """Sample a framing mode. If multi-framing, picks randomly."""
        if self._framing_options:
            return random.choice(self._framing_options)
        return self.framing

    def sample_and_generate(
        self, transcript
    ) -> Tuple[str, dict]:
        """Sample an attack family and generate the attack text."""
        framing = self._sample_framing()

        # Force benign at benign_ratio frequency
        if random.random() < self.benign_ratio:
            attack = self.attacks["benign_followup"]
            text = attack.generate(transcript)
            text = wrap_attack_text(text, "benign_followup", framing)
            return text, {"attack_name": "benign_followup", "is_valid_evidence": True, "framing": framing}

        # Sample from adversarial families
        if self.adversarial_families and self.adversarial_weights:
            family = random.choices(
                self.adversarial_families,
                weights=self.adversarial_weights,
                k=1,
            )[0]
        else:
            family = "benign_followup"

        attack = self.attacks.get(family, self.attacks["benign_followup"])
        text = attack.generate(transcript)
        is_valid = attack.is_valid_evidence()
        text = wrap_attack_text(text, family, framing)

        return text, {"attack_name": family, "is_valid_evidence": is_valid, "framing": framing}
