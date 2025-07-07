"""affine.envs.gaia – thin wrapper around the gaia-benchmark package.

GAIA is an eval suite for conversational agents.  We expose it via the same
BaseEnv interface so that it can be run inside the validator.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Any

import affine as af

try:
    from gaia_benchmark import evaluate  # type: ignore
except ImportError:  # pragma: no cover – optional dep
    evaluate = None


class GAIA(af.BaseEnv):
    """One GAIA conversation sample per *generate* call."""

    async def generate(self) -> af.Challenge:  # noqa: D401
        prompt = "Answer the GAIA benchmark question: What is life?"
        return af.Challenge(env=self, prompt=prompt, extra={})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        if evaluate is None:
            score = 0.0  # cannot evaluate without package
        else:
            # GAIA's evaluate is synchronous; wrap in executor
            loop = asyncio.get_running_loop()
            score = await loop.run_in_executor(None, evaluate, response.response or "")
        return af.Evaluation(env=self, score=float(bool(score)), extra={}) 