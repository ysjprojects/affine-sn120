import random, re, string
import affine as af

# ---------------------------------------------------------------------------
# Environment "needle-in-a-haystack"
# ---------------------------------------------------------------------------

class MemoryNeedle(af.BaseEnv):
    """Tests the model's ability to recall a specific piece of information
    inserted ~200 words earlier in a context of about 1,500 tokens."""

    tokens: int     # total length (≈ words/tokens)
    insert_at: int  # position of the needle
    secret: str     

    def __init__(self, tokens: int = 1500, insert_at: int = 200, secret: str | None = None):
        secret = secret or "".join(random.choices(string.ascii_uppercase, k=6))
        super().__init__(tokens=tokens, insert_at=insert_at, secret=secret)

    # -----------------------------------------------------------------------
    # Phase 1: challenge generation
    # -----------------------------------------------------------------------
    async def generate(self):
        # 1) Create a pile of random words until reaching self.tokens
        words: list[str] = []
        while len(words) < self.tokens:
            word_len = random.randint(3, 10)
            words.append("".join(random.choices(string.ascii_lowercase, k=word_len)))
        
        # 2) Insert the needle-sentence at position self.insert_at
        phrase = f"IMPORTANT: the secret code is {self.secret}."
        words.insert(min(self.insert_at, len(words)), phrase)

        # 3) Compose the final prompt + question
        context = " ".join(words)
        prompt = (
            f"{context}\n\n"
            "Question: What is the secret code?\n"
            "Answer in a single word."
        )

        return af.Challenge(
            env=self,
            prompt=prompt,
            extra={"secret": self.secret}  # stored for evaluation
        )

    # -----------------------------------------------------------------------
    # Phase 2: evaluation of the model's response
    # -----------------------------------------------------------------------
    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        expected = challenge.extra["secret"].upper()
        # Take the first alphabetic word from the response
        m = re.search(r"\b([A-Za-z]+)\b", response.response or "")
        got = m.group(1).upper() if m else ""
        score = float(got == expected)
        return af.Evaluation(env=self, score=score, extra={"expected": expected, "got": got})

# ---------------------------------------------------------------------------
# Deterministic generator (helper used by the incentives system)
# ---------------------------------------------------------------------------

def generate(seed: int, tokens: int = 1500, insert_at: int = 200, secret: str | None = None) -> dict:
    """Deterministically generates a prompt and the ground truth.

    Parameters
    ----------
    seed : int
        RNG seed – same seed ⇒ same prompt / same secret code.
    tokens : int
        Length ~number of words.
    insert_at : int
        Position (in words) where the needle-sentence is inserted.
    secret : str | None
        Secret code to recall; generated randomly if None.

    Returns
    -------
    dict
        ``{"prompt": str, "secret": str}``
    """
    import random as _r
    import string as _s

    rnd = _r.Random(seed)
    secret = secret or "".join(rnd.choices(_s.ascii_uppercase, k=6))

    words: list[str] = []
    while len(words) < tokens:
        word_len = rnd.randint(3, 10)
        words.append("".join(rnd.choices(_s.ascii_lowercase, k=word_len)))

    phrase = f"IMPORTANT: the secret code is {secret}."
    words.insert(min(insert_at, len(words)), phrase)

    context = " ".join(words)
    prompt = (
        f"{context}\n\n"
        "Question: What is the secret code?\n"
        "Answer in a single word."
    )
    return {"prompt": prompt, "secret": secret}
