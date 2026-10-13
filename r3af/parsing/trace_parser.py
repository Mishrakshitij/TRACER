import re
from typing import Optional, List, Tuple

from .schema import ReasoningStep, ParsedTrace


class TraceParser:
    """Extract reasoning steps and answer from model output.

    Supports two formats:
    1. <think>...</think><answer>...</answer>
    2. Step N: ... patterns
    Fallback: treat entire text as a single step.
    """

    # Patterns for <think>...</think> format
    THINK_PATTERN = re.compile(
        r"<think>(.*?)</think>", re.DOTALL
    )
    ANSWER_PATTERN = re.compile(
        r"<answer>(.*?)</answer>", re.DOTALL
    )

    # Pattern for Step N: format
    STEP_PATTERN = re.compile(
        r"(?:Step\s+(\d+)\s*[:.])\s*(.*?)(?=Step\s+\d+\s*[:.:]|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # Fallback: split on double newlines
    PARAGRAPH_SPLIT = re.compile(r"\n\n+")

    # Answer extraction patterns
    BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
    FINAL_ANSWER_PATTERN = re.compile(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)",
        re.IGNORECASE,
    )
    HASH_ANSWER_PATTERN = re.compile(r"####\s*(.+?)(?:\n|$)")

    @classmethod
    def parse(cls, text: str, tokenizer=None) -> ParsedTrace:
        """Parse a model output into a structured trace.

        Args:
            text: Raw model output text.
            tokenizer: Optional tokenizer for computing token positions.

        Returns:
            ParsedTrace with extracted steps and answer.
        """
        if not text or not text.strip():
            return ParsedTrace(raw_text=text or "", parse_success=False)

        # Try <think>...</think> format first
        trace = cls._parse_think_format(text, tokenizer)
        if trace.parse_success:
            return trace

        # Try Step N: format
        trace = cls._parse_step_format(text, tokenizer)
        if trace.parse_success:
            return trace

        # Fallback: treat entire text as single step
        return cls._parse_fallback(text, tokenizer)

    @classmethod
    def _parse_think_format(
        cls, text: str, tokenizer=None
    ) -> ParsedTrace:
        """Parse <think>...</think><answer>...</answer> format."""
        think_match = cls.THINK_PATTERN.search(text)
        if not think_match:
            return ParsedTrace(raw_text=text, parse_success=False)

        think_content = think_match.group(1).strip()
        answer = cls._extract_answer(text)

        # Split thinking into steps
        steps = cls._split_think_into_steps(think_content, tokenizer, text)

        return ParsedTrace(
            steps=steps,
            answer=answer,
            raw_text=text,
            parse_success=len(steps) > 0,
        )

    @classmethod
    def _parse_step_format(
        cls, text: str, tokenizer=None
    ) -> ParsedTrace:
        """Parse Step 1: ... Step 2: ... format."""
        matches = list(cls.STEP_PATTERN.finditer(text))
        if not matches:
            return ParsedTrace(raw_text=text, parse_success=False)

        steps = []
        for i, match in enumerate(matches):
            step_idx = int(match.group(1)) if match.group(1) else i + 1
            content = match.group(2).strip()
            if content:
                start_pos, end_pos = cls._compute_token_positions(
                    match.start(2), match.end(2), text, tokenizer
                )
                steps.append(
                    ReasoningStep(
                        step_index=step_idx,
                        content=content,
                        start_token_pos=start_pos,
                        end_token_pos=end_pos,
                    )
                )

        answer = cls._extract_answer(text)

        return ParsedTrace(
            steps=steps,
            answer=answer,
            raw_text=text,
            parse_success=len(steps) > 0,
        )

    @classmethod
    def _parse_fallback(
        cls, text: str, tokenizer=None
    ) -> ParsedTrace:
        """Fallback: treat entire text as a single step."""
        answer = cls._extract_answer(text)
        start_pos, end_pos = cls._compute_token_positions(
            0, len(text), text, tokenizer
        )
        steps = [
            ReasoningStep(
                step_index=1,
                content=text.strip(),
                start_token_pos=start_pos,
                end_token_pos=end_pos,
            )
        ]
        return ParsedTrace(
            steps=steps,
            answer=answer,
            raw_text=text,
            parse_success=True,
        )

    @classmethod
    def _split_think_into_steps(
        cls,
        think_content: str,
        tokenizer=None,
        full_text: str = "",
    ) -> List[ReasoningStep]:
        """Split thinking content into individual steps."""
        # Try Step N: pattern within the think block
        step_matches = list(cls.STEP_PATTERN.finditer(think_content))
        if step_matches:
            steps = []
            think_start = full_text.find(think_content)
            for i, match in enumerate(step_matches):
                step_idx = (
                    int(match.group(1)) if match.group(1) else i + 1
                )
                content = match.group(2).strip()
                if content:
                    abs_start = think_start + match.start(2) if think_start >= 0 else match.start(2)
                    abs_end = think_start + match.end(2) if think_start >= 0 else match.end(2)
                    start_pos, end_pos = cls._compute_token_positions(
                        abs_start, abs_end, full_text, tokenizer
                    )
                    steps.append(
                        ReasoningStep(
                            step_index=step_idx,
                            content=content,
                            start_token_pos=start_pos,
                            end_token_pos=end_pos,
                        )
                    )
            return steps

        # Try splitting by numbered lines: "1. ...", "2. ...", etc.
        numbered = re.split(r"\n\s*\d+\.\s+", think_content)
        if len(numbered) > 1:
            steps = []
            offset = full_text.find(think_content)
            if offset < 0:
                offset = 0
            pos = 0
            for i, segment in enumerate(numbered):
                segment = segment.strip()
                if not segment:
                    continue
                seg_start = think_content.find(segment, pos)
                seg_end = seg_start + len(segment)
                pos = seg_end
                abs_start = offset + seg_start
                abs_end = offset + seg_end
                start_pos, end_pos = cls._compute_token_positions(
                    abs_start, abs_end, full_text, tokenizer
                )
                steps.append(
                    ReasoningStep(
                        step_index=len(steps) + 1,
                        content=segment,
                        start_token_pos=start_pos,
                        end_token_pos=end_pos,
                    )
                )
            if steps:
                return steps

        # Fallback: split by newlines
        lines = [ln.strip() for ln in think_content.split("\n") if ln.strip()]
        if not lines:
            return []

        steps = []
        think_start = full_text.find(think_content)
        if think_start < 0:
            think_start = 0
        for i, line in enumerate(lines):
            line_start = think_content.find(line)
            abs_start = think_start + line_start
            abs_end = abs_start + len(line)
            start_pos, end_pos = cls._compute_token_positions(
                abs_start, abs_end, full_text, tokenizer
            )
            steps.append(
                ReasoningStep(
                    step_index=i + 1,
                    content=line,
                    start_token_pos=start_pos,
                    end_token_pos=end_pos,
                )
            )
        return steps

    @classmethod
    def _extract_answer(cls, text: str) -> Optional[str]:
        """Extract the final answer from text using multiple patterns."""
        # Try <answer>...</answer>
        answer_match = cls.ANSWER_PATTERN.search(text)
        if answer_match:
            return answer_match.group(1).strip()

        # Try \\boxed{...}
        boxed_match = cls.BOXED_PATTERN.search(text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try #### answer
        hash_match = cls.HASH_ANSWER_PATTERN.search(text)
        if hash_match:
            return hash_match.group(1).strip()

        # Try "the answer is ..."
        final_match = cls.FINAL_ANSWER_PATTERN.search(text)
        if final_match:
            return final_match.group(1).strip()

        return None

    @classmethod
    def _compute_token_positions(
        cls,
        char_start: int,
        char_end: int,
        full_text: str,
        tokenizer=None,
    ) -> Tuple[int, int]:
        """Convert character positions to approximate token positions.

        If no tokenizer is provided, uses character offsets as a proxy.
        With a tokenizer, encodes prefixes to compute exact token boundaries.
        """
        if tokenizer is None:
            # Approximate: use character positions directly
            return (char_start, char_end)

        # Tokenize prefix to get start position
        prefix = full_text[:char_start]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        start_token = len(prefix_tokens)

        # Tokenize up to end to get end position
        prefix_and_span = full_text[:char_end]
        span_tokens = tokenizer.encode(prefix_and_span, add_special_tokens=False)
        end_token = len(span_tokens)

        return (start_token, end_token)
