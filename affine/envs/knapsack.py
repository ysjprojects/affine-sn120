import random, re, affine as af
from itertools import combinations

class Knapsack(af.BaseEnv):
    n: int
    cap: int

    def __init__(self, n=6, cap=15):
        super().__init__(n=n, cap=cap)

    def _solve(self, items):
        best_val, best_set = -1, []
        for r in range(len(items)+1):
            for combo in combinations(range(len(items)), r):
                w = sum(items[i][0] for i in combo)
                v = sum(items[i][1] for i in combo)
                if w<=self.cap and v>best_val:
                    best_val, best_set = v, combo
        return list(best_set)

    async def generate(self):
        items = [(random.randint(1,10), random.randint(1,10)) for _ in range(self.n)]
        best = self._solve(items)
        prompt = f"Items (w,v): {items}, capacity {self.cap}. Return indices chosen (0-based)."
        return af.Challenge(env=self, prompt=prompt, extra={"items": items, "best": best})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        chosen = list(map(int, re.findall(r"\d+", response.response or "")))
        ok = set(chosen) == set(challenge.extra["best"])
        return af.Evaluation(env=self, score=float(ok), extra={"expected":challenge.extra['best'], "got":chosen})

ENV_CLASS = Knapsack 