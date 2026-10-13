from .base_reward import BaseRewardComponent, RewardOutput
from ..verifiers import get_verifier


class LengthReward(BaseRewardComponent):
    """R_len = -lambda * 1[a'=y*] * max(0, |tau'| - |tau| - delta)

    Anti-overthinking penalty: only penalizes token inflation when the
    post-attack answer is correct. This prevents the model from gaining
    robustness by simply producing very long responses.
    """

    def __init__(self, penalty_lambda: float = 0.1, delta: int = 200):
        self.penalty_lambda = penalty_lambda
        self.delta = delta

    def compute(self, transcripts, **kwargs) -> RewardOutput:
        values = []
        inflation_stats = []

        for transcript in transcripts:
            post_answer = transcript.post_attack_answer
            ground_truth = transcript.ground_truth
            dataset = transcript.dataset

            verifier = get_verifier(dataset)
            post_correct = verifier.verify(
                str(post_answer) if post_answer else "", ground_truth
            )

            # Compute token counts
            clean_turn = transcript.clean_turn
            post_turn = transcript.post_attack_turn

            clean_tokens = clean_turn.token_count if clean_turn else 0
            post_tokens = post_turn.token_count if post_turn else 0

            inflation = post_tokens - clean_tokens - self.delta

            if post_correct and inflation > 0:
                reward = -self.penalty_lambda * inflation
            else:
                reward = 0.0

            values.append(reward)
            inflation_stats.append({
                "clean_tokens": clean_tokens,
                "post_tokens": post_tokens,
                "inflation": max(0, inflation),
                "penalized": post_correct and inflation > 0,
            })

        return RewardOutput(
            values=values,
            metadata={"inflation_stats": inflation_stats},
        )
