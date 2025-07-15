"""affine.envs.abd – *Almost Blind Decompiler* mini-environment.

The task: we give the model a short Python **program** together with its
**expected stdout** and ask for an **input** that triggers exactly that
output.  The solver must supply the input inside `<INPUT>...</INPUT>` tags.

This trimmed version (≈150 LoC) is self-contained – no external datasets – and
relies on a few randomly-generated toy programs so that we can generate an
arbitrary amount of samples on-the-fly without network traffic.
"""

from __future__ import annotations

import random, re, textwrap
from typing import List, Dict, Any
import affine as af
from ..utils import ProgramExecutor
from pydantic import Field

# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = textwrap.dedent("""
    You are a programming expert. Given a Python program and its expected
    output, determine a **stdin** input that reproduces the output.

    Program:
    ```python
    {program}
    ```

    Expected Output:
    ```
    {output}
    ```

    Task: provide the input **exactly** as the program should receive it.
    Insert it between the tags below – nothing else will be captured.

    <INPUT>
    your input here
    </INPUT>
    """).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_string(n: int = 5) -> str:
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))


def _toy_program() -> tuple[str, str]:
    """Return *(program, output)* for a randomly-parameterised snippet.

    We generate three kinds of very small deterministic programs so we can
    compute their ground-truth outputs cheaply.
    """
    kind = random.choice(["reverse", "sum", "upper"])

    if kind == "reverse":
        inp = _random_string()
        program = "s=input()[::-1];print(s)"
        expected = inp[::-1]
    elif kind == "sum":
        a, b = random.randint(0, 99), random.randint(0, 99)
        program = "a,b=map(int,input().split());print(a+b)"
        inp = f"{a} {b}"
        expected = str(a + b)
    else:  # upper
        inp = _random_string()
        program = "print(input().upper())"
        expected = inp.upper()

    return program, inp, expected


class ABD(af.BaseEnv):
    """Simple ABD environment using toy programs generated on-the-fly."""

    executor: ProgramExecutor = Field(default_factory=ProgramExecutor, exclude=True)

    # ------------------------------------------------------------------
    # Challenge generation
    # ------------------------------------------------------------------

    async def generate(self) -> af.Challenge:
        program, hidden_input, expected = _toy_program()

        prompt = PROMPT_TEMPLATE.format(program=program, output=expected)
        return af.Challenge(
            env=self,
            prompt=prompt,
            extra={"program": program, "expected": expected, "hidden_input": hidden_input},
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_input(resp: str) -> str:
        """Return the payload inside the last `<INPUT>...</INPUT>` pair."""
        # Strip auxiliary *thinking* tags (optional convention)
        resp = re.sub(r"<(think|thinking)>.*?</\1>", "", resp, flags=re.DOTALL | re.IGNORECASE)
        matches = re.findall(r"<INPUT>(.*?)</INPUT>", resp, flags=re.DOTALL | re.IGNORECASE)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def _same_out(a: str, b: str) -> bool:
        return a.strip() == b.strip()

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        program = challenge.extra["program"]
        expected = challenge.extra["expected"]
        supplied_inp = self._extract_input(response.response or "")

        if not supplied_inp:
            return af.Evaluation(env=self, score=0.0, extra={"error": "no <INPUT> tag found"})

        out, err = self.executor.execute(program, supplied_inp)
        if err:
            return af.Evaluation(env=self, score=0.0, extra={"error": err})

        score = 1.0 if self._same_out(out, expected) else 0.0
        return af.Evaluation(
            env=self,
            score=score,
            extra={"expected": expected, "got": out.strip(), "input": supplied_inp},
        )

    # Clean-up – not strictly needed but keeps temp dir tidy
    def __del__(self):
        try:
            import os, glob, tempfile
            for f in glob.glob(str(tempfile.gettempdir() + "/tmp*")):
                os.unlink(f)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Deterministic helper – used by the incentive mechanism
# ---------------------------------------------------------------------------

def generate(seed: int) -> dict:
    """Deterministically generate a toy ABD sample.

    The implementation mirrors the async *generate* method but uses a
    deterministic RNG seeded with *seed* so that every validator observes the
    **exact** same challenge.
    """
    import random as _r
    rnd = _r.Random(seed)

    # Re-implement the toy program generation with a private RNG instance.
    kind = rnd.choice(["reverse", "sum", "upper"])
    if kind == "reverse":
        txt = "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        program = "s=input()[::-1];print(s)"
        hidden_input = txt
        expected = txt[::-1]
    elif kind == "sum":
        a, b = rnd.randint(0, 99), rnd.randint(0, 99)
        program = "a,b=map(int,input().split());print(a+b)"
        hidden_input = f"{a} {b}"
        expected = str(a + b)
    else:
        txt = "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        program = "print(input().upper())"
        hidden_input = txt
        expected = txt.upper()

    prompt = PROMPT_TEMPLATE.format(program=program, output=expected)
    return {
        "prompt": prompt,
        "program": program,
        "expected": expected,
        "hidden_input": hidden_input,
    } 