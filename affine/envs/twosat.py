import random, re, affine as af

class TwoSAT(af.BaseEnv):
    n: int
    m: int

    def __init__(self, n: int = 5, m: int = 10):
        super().__init__(n=n, m=m)

    async def generate(self):
        sol = {i: random.choice([True, False]) for i in range(1, self.n + 1)}
        clauses = []
        for _ in range(self.m):
            a, b = random.sample(range(1, self.n + 1), 2)
            la = a if sol[a] else -a
            lb = b if sol[b] else -b
            if random.random() < 0.5:
                la *= -1
            if random.random() < 0.5:
                lb *= -1
            clauses.append((la, lb))
        formula = " ∧ ".join(
            f"({('¬' if l<0 else '')}x{abs(l)} ∨ {('¬' if r<0 else '')}x{abs(r)})" for l, r in clauses
        )
        prompt = (
            f"Solve this 2-SAT formula on x1..x{self.n}:\n{formula}\n"
            "Answer with comma-separated assignments `x1=True, ...`, or `UNSAT`."
        )
        return af.Challenge(env=self, prompt=prompt, extra={"sol": sol, "clauses": clauses})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        sol, clauses = challenge.extra["sol"], challenge.extra["clauses"]
        got = {int(v): val.lower() in ("true", "1")
               for v, val in re.findall(r"x(\d+)=(True|False|1|0)", response.response or "")}
        ok = all(any((lit > 0) == got.get(abs(lit), None) for lit in clause) for clause in clauses)
        return af.Evaluation(env=self, score=float(ok), extra={"expected": sol, "got": got})

ENV_CLASS = TwoSAT 