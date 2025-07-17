import random
import affine as af

class ABDUCTION(af.BaseEnv):
    """Simple propositional abduction puzzles.

    The generator samples a rule "A → B" plus the observation *B*
    and asks the model to propose the most plausible explanation
    (i.e. produce the hypothesis *A*).  Evaluation succeeds only
    if the returned hypothesis matches exactly the ground-truth
    literal (case-insensitive).
    """

    premise: str | None = None
    consequence: str | None = None

    OBJECTS = [
        ("rain", "wet_ground"),
        ("fire", "smoke"),
        ("holiday", "traffic"),
        ("leak", "wet_floor"),
        ("power_outage", "dark_house"),
    ]

    def __init__(self):
        super().__init__()

    async def generate(self):
        hyp, obs = random.choice(self.OBJECTS)
        self.premise, self.consequence = hyp, obs
        prompt = (
            "Given the rule 'If A then B' and the observation that B is true, "
            "what is the most plausible explanation (value of A)?\n"
            f"Rule: If {hyp.replace('_',' ')} then {obs.replace('_',' ')}.\n"
            f"Observation: {obs.replace('_',' ')} is true.\n"
            "Answer with just the hypothesis word (lowercase)."
        )
        return af.Challenge(env=self, prompt=prompt, extra={"hyp": hyp, "obs": obs})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        expected = challenge.extra["hyp"].lower()
        got = (response.response or "").strip().split()[0].lower()
        score = float(got == expected)
        return af.Evaluation(env=self, score=score, extra={"expected": expected, "got": got}) 