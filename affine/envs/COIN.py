import random
import affine as af

class COIN(af.BaseEnv):
    async def generate(self) -> af.Challenge:
        return af.Challenge(
            env=self,
            prompt="I flipped a coin, guess HEADS or TAILS.",
            extra={"answer": random.choice(["HEADS", "TAILS"])}
        )

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        ans = challenge.extra["answer"]
        guess = (response.response or "").strip().upper()
        return af.Evaluation(
            env=self,
            score=float(guess == ans),
            extra={"answer": ans, "guess": guess}
        )