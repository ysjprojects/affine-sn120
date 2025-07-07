"""affine.envs.research – toy Research QA environment

Provides deterministic, citation-aware questions similar to web-search tasks.
Used as a third environment labelled "RES" alongside SAT and ABD.
"""
from __future__ import annotations

import random
from typing import Dict, Set

import affine as af

# ---------------------------------------------------------------------------
# Mock corpus (small, static) – in real life you would query a search engine.
# ---------------------------------------------------------------------------
_CORPUS = [
    {
        "query": "Who wrote the novel 1984?",
        "answer": "George Orwell",
        "citations": {"https://en.wikipedia.org/wiki/Nineteen_Eighty-Four"},
    },
    {
        "query": "What is the tallest mountain on Earth?",
        "answer": "Mount Everest",
        "citations": {"https://en.wikipedia.org/wiki/Mount_Everest"},
    },
    {
        "query": "Which element has the chemical symbol Au?",
        "answer": "Gold",
        "citations": {"https://en.wikipedia.org/wiki/Gold"},
    },
]

# ---------------------------------------------------------------------------
# Deterministic generator helper – required by incentive mechanism
# ---------------------------------------------------------------------------

def generate(seed: int) -> Dict:
    """Return a reproducible Research challenge for *seed*.

    Parameters
    ----------
    seed : int
        Global seed – each validator with the same seed gets the same sample.

    Returns
    -------
    dict
        {"query", "expected_answer", "citations"}
    """
    rnd = random.Random(seed)
    sample = _CORPUS[rnd.randrange(len(_CORPUS))]
    return {
        "query": sample["query"],
        "expected_answer": sample["answer"],
        "citations": sample["citations"],
    }

# ---------------------------------------------------------------------------
# Async environment compatible with BaseEnv
# ---------------------------------------------------------------------------

class RES(af.BaseEnv):
    """Research QA environment (mocked)."""

    async def generate(self) -> af.Challenge:  # noqa: D401
        # use real RNG for diversity in ad-hoc runs
        d = random.choice(_CORPUS)
        prompt = (
            f"Answer the following question in one short sentence and cite your source URLs in parentheses.\n"
            f"Question: {d['query']}"
        )
        return af.Challenge(env=self, prompt=prompt, extra=d)

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        exp_ans = challenge.extra["answer"].lower()
        got_ans = (response.response or "").split("(")[0].strip().lower()
        correct = int(exp_ans in got_ans)
        return af.Evaluation(env=self, score=float(correct), extra={}) 