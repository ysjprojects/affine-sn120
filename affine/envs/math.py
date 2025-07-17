import random, re, math as _math
import affine as af

class MATH(af.BaseEnv):
    """Elementary arithmetic word-problem environment (GSM8K-style)."""

    templates = [
        (
            "If John has {a} apples and eats {b}, how many apples does he have left?",
            lambda a, b: a - b,
        ),
        (
            "Sarah bought {a} books and her friend gave her {b} more. How many books does Sarah have now?",
            lambda a, b: a + b,
        ),
        (
            "A rectangle has length {a} cm and width {b} cm. What is its area in square cm?",
            lambda a, b: a * b,
        ),
        (
            "Tom walked {a} km on Monday and {b} km on Tuesday. How far did he walk in total?",
            lambda a, b: a + b,
        ),
        (
            "A tank contains {a} liters of water. {b} liters are drained. How much water remains?",
            lambda a, b: a - b,
        ),
    ]

    def __init__(self, tol: float = 1e-2):
        super().__init__(tol=tol)
        self.tol = tol

    async def generate(self):
        tmpl, fn = random.choice(self.templates)
        a, b = random.randint(3, 50), random.randint(1, 20)
        while b > a and "left" in tmpl:
            b = random.randint(1, a)
        answer = fn(a, b)
        prompt = tmpl.format(a=a, b=b)
        return af.Challenge(env=self, prompt=prompt, extra={"ans": answer})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        expected = challenge.extra["ans"]
        # extract first number in reply
        m = re.search(r"-?\d+(?:\.\d+)?", response.response or "")
        got = float(m.group()) if m else _math.nan
        score = float(abs(got - expected) <= self.tol)
        return af.Evaluation(env=self, score=score, extra={"expected": expected, "got": got}) 