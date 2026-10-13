from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ReasoningStep:
    step_index: int
    content: str
    start_token_pos: Optional[int] = None
    end_token_pos: Optional[int] = None

@dataclass
class ParsedTrace:
    steps: List[ReasoningStep] = field(default_factory=list)
    answer: Optional[str] = None
    raw_text: str = ""
    parse_success: bool = False

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def step_contents(self) -> List[str]:
        return [s.content for s in self.steps]
