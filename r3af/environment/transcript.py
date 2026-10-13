from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from ..parsing.schema import ReasoningStep, ParsedTrace


class TurnRole(Enum):
    """Role of a participant in the Trace Retention Game."""
    SYSTEM = "system"
    PROBLEM = "problem"
    RETAINER = "retainer"
    ATTACKER = "attacker"


@dataclass
class Turn:
    """A single turn in a TRG game transcript."""

    role: TurnRole
    content: str
    reasoning_steps: Optional[List[ReasoningStep]] = None
    answer: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transcript:
    """Full transcript of a Trace Retention Game.

    Provides convenient properties for accessing clean (pre-attack) and
    post-attack turns, answers, and parsed traces.
    """

    problem_id: str = ""
    problem: str = ""
    ground_truth: str = ""
    dataset: str = ""
    turns: List[Turn] = field(default_factory=list)

    # -- Turn accessors --------------------------------------------------------

    @property
    def clean_turn(self) -> Optional[Turn]:
        """First RETAINER turn (clean response before any attack)."""
        retainer_turns = [t for t in self.turns if t.role == TurnRole.RETAINER]
        return retainer_turns[0] if retainer_turns else None

    @property
    def attack_turn(self) -> Optional[Turn]:
        """First ATTACKER turn."""
        attacker_turns = [t for t in self.turns if t.role == TurnRole.ATTACKER]
        return attacker_turns[0] if attacker_turns else None

    @property
    def post_attack_turn(self) -> Optional[Turn]:
        """Last RETAINER turn that appears after an ATTACKER turn."""
        found_attack = False
        last_retainer_after_attack = None
        for turn in self.turns:
            if turn.role == TurnRole.ATTACKER:
                found_attack = True
            elif turn.role == TurnRole.RETAINER and found_attack:
                last_retainer_after_attack = turn
        return last_retainer_after_attack

    # -- Answer accessors ------------------------------------------------------

    @property
    def clean_answer(self) -> Optional[str]:
        turn = self.clean_turn
        return turn.answer if turn else None

    @property
    def post_attack_answer(self) -> Optional[str]:
        turn = self.post_attack_turn
        return turn.answer if turn else None

    # -- Trace accessors -------------------------------------------------------

    @property
    def clean_trace(self) -> Optional[ParsedTrace]:
        """ParsedTrace constructed from the clean turn reasoning steps."""
        turn = self.clean_turn
        if turn is None or turn.reasoning_steps is None:
            return None
        return ParsedTrace(
            steps=turn.reasoning_steps,
            answer=turn.answer,
            raw_text=turn.content,
            parse_success=True,
        )

    @property
    def post_attack_trace(self) -> Optional[ParsedTrace]:
        """ParsedTrace constructed from the post-attack turn reasoning steps."""
        turn = self.post_attack_turn
        if turn is None or turn.reasoning_steps is None:
            return None
        return ParsedTrace(
            steps=turn.reasoning_steps,
            answer=turn.answer,
            raw_text=turn.content,
            parse_success=True,
        )

    # -- Serialization ---------------------------------------------------------

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert transcript to chat message format for model input.

        Mapping:
        - PROBLEM  -> role "user"
        - RETAINER -> role "assistant"
        - ATTACKER -> role "user"
        """
        messages = []
        for turn in self.turns:
            if turn.role == TurnRole.SYSTEM:
                messages.append({"role": "system", "content": turn.content})
            elif turn.role == TurnRole.PROBLEM:
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == TurnRole.RETAINER:
                messages.append({"role": "assistant", "content": turn.content})
            elif turn.role == TurnRole.ATTACKER:
                messages.append({"role": "user", "content": turn.content})
        return messages
