import random, re, affine as af

class SortList(af.BaseEnv):
    length: int
    min_val: int
    max_val: int

    def __init__(self, length=10, min_val=0, max_val=100):
        super().__init__(length=length, min_val=min_val, max_val=max_val)

    async def generate(self):
        lst = [random.randint(self.min_val, self.max_val) for _ in range(self.length)]
        prompt = f"Sort this list in ascending order: {lst}"
        return af.Challenge(env=self, prompt=prompt, extra={"list": lst})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        got = list(map(int, re.findall(r"-?\d+", response.response or "")))
        ok = got == sorted(challenge.extra["list"])
        return af.Evaluation(env=self, score=float(ok), extra={"expected":sorted(challenge.extra['list']), "got":got})

ENV_CLASS = SortList 