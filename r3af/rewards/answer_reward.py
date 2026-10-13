from .base_reward import BaseRewardComponent, RewardOutput
from ..verifiers import get_verifier


class AnswerReward(BaseRewardComponent):
    """R_ans = 1[a' = y*] using dataset-specific verifiers.

    Binary reward: 1.0 if post-attack answer matches ground truth, 0.0 otherwise.
    """

    def compute(self, transcripts, **kwargs) -> RewardOutput:
        values = []
        correct_count = 0

        for transcript in transcripts:
            post_answer = transcript.post_attack_answer
            ground_truth = transcript.ground_truth
            dataset = transcript.dataset

            verifier = get_verifier(dataset)
            is_correct = verifier.verify(
                str(post_answer) if post_answer else "", ground_truth
            )

            reward = 1.0 if is_correct else 0.0
            values.append(reward)
            if is_correct:
                correct_count += 1

        return RewardOutput(
            values=values,
            metadata={
                "accuracy": correct_count / max(len(transcripts), 1),
                "correct_count": correct_count,
            },
        )
