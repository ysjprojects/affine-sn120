import random, re, affine as af

class ParityPuzzle(af.BaseEnv):
    n: int

    def __init__(self, n: int = 16):
        super().__init__(n=n)

    async def generate(self):
        bits = [random.choice([0, 1]) for _ in range(self.n)]
        parity = sum(bits) % 2  # 0 even, 1 odd
        prompt = f"Given bits {''.join(map(str, bits))}, is parity odd or even?"
        return af.Challenge(env=self, prompt=prompt, extra={"bits": bits, "parity": parity})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        ans = (response.response or "").strip().lower()
        got = 1 if "odd" in ans else 0
        ok = got == challenge.extra["parity"]
        return af.Evaluation(env=self, score=float(ok), extra={"expected": challenge.extra["parity"], "got": got})

ENV_CLASS = ParityPuzzle 