
"""affine.envs.gsm8k – math-word-problem environment using the GSM8K dataset.

This environment presents a grade-school math word problem (from the
`openai/gsm8k` HuggingFace dataset) and expects the solver to return **only**
the final numeric answer.  It follows the same design philosophy as
`affine.envs.abd.ABD`: streaming dataset loading, reservoir sampling, stock
replenishment in background, and thorough output normalisation.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Any, ClassVar, Dict, List, TYPE_CHECKING

from datasets import load_dataset

# Some datasets typing stubs are incomplete; use *Any* for runtime flexibility.
if TYPE_CHECKING:  # noqa: F401 – for static checkers only
    from affine import Challenge, Response, Evaluation  # type: ignore[attr-defined]

import affine as af
from .. import val_config

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Prompt template
# ────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Solve the following problem and **reply with only the final numeric "
    "answer** (no explanation, no units, no punctuation).\n\n{question}\n"
)

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────

def _extract_number(s: str) -> str:
    """Return the first integer/decimal number found in *s* as a canonical str.

    This mirrors the normalisation done in many GSM8K baselines: answers are
    typically written as `"#### 42"` or `"42."`.  We strip commas, leading
    pluses or zeros, and trailing decimal ".0".
    """
    # Remove thousands separators and whitespace
    s = s.replace(",", "").replace(" ", "")
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+)", s)
    if not match:
        return s.strip()

    num = match.group(0)
    # Strip leading '+', leading zeros (but keep single zero), and trailing '.0'
    num = num.lstrip("+")
    if re.match(r"0+\d", num):
        num = num.lstrip("0") or "0"
    if num.endswith(".0"):
        num = num[:-2]
    return num

# ────────────────────────────────────────────────────────────────────────────
# Environment class
# ────────────────────────────────────────────────────────────────────────────

# pyright: reportGeneralTypeIssues=false
class GSM8K(af.BaseEnv):  # type: ignore[attr-defined]
    """Grade-School Math (GSM8K) environment."""

    # Dataset config -------------------------------------------------------
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"  # original GSM8K split name
    dataset_split: str = "train"

    # Internal handles -----------------------------------------------------
    _dataset: Any = None  # stream dataset object (typing stub incomplete)
    _iter: Any = None  # iterator over the streaming split

    # Operational parameters ----------------------------------------------
    LLM_RESPONSE_TIMEOUT: ClassVar[float] = val_config.LLM_RESPONSE_TIMEOUT
    MAX_SAMPLES: ClassVar[int] = 1_000  # how many to keep in in-memory buffer

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        if self._dataset is not None:
            return
        try:
            self._dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.dataset_split,
                streaming=True,
            )
            self._iter = iter(self._dataset)
            # Ensure we get a real iterator object (with __next__). Older
            # versions of `datasets` (< 2.0) may return a *method* that must
            # be called to obtain the actual generator.  Keep unwrapping
            # until we finally get an object that implements `__next__`.
            while callable(self._iter) and not hasattr(self._iter, "__next__"):
                self._iter = self._iter()  # type: ignore[func-call] – runtime guard
            logger.info(
                "GSM8K dataset loaded (%s/%s, streaming mode)",
                self.dataset_config,
                self.dataset_split,
            )
        except Exception as e:
            logger.error("Failed to load GSM8K dataset: %s", e)
            raise RuntimeError(f"Dataset load error: {e}")

    # ------------------------------------------------------------------
    # Sampling helpers (reservoir sampling)
    # ------------------------------------------------------------------

    def _random_samples(self, n: int) -> List[Dict[str, str]]:
        """Return *n* random samples using reservoir sampling."""
        if self._dataset is None or self._iter is None:
            self._load_dataset()

        reservoir: List[Dict[str, str]] = []
        total = 0
        while len(reservoir) < self.MAX_SAMPLES:
            try:
                item = next(self._iter)  # type: ignore[arg-type]
            except StopIteration:
                # Restart iterator once we hit the end of the split
                self._iter = iter(self._dataset)  # type: ignore[arg-type]
                while callable(self._iter) and not hasattr(self._iter, "__next__"):
                    self._iter = self._iter()  # type: ignore[func-call] – runtime guard
                continue

            reservoir.append(item)
            total += 1
            if total >= self.MAX_SAMPLES:
                break

        if n >= len(reservoir):
            return reservoir[:n]
        idxs = random.sample(range(len(reservoir)), n)
        return [reservoir[i] for i in idxs]

    # ------------------------------------------------------------------
    # Challenge generation
    # ------------------------------------------------------------------

    from typing import Any as _Any

    async def _make_challenge_from_sample(self, sample: Dict[str, str]) -> _Any:
        question = sample["question"].strip()
        expected = _extract_number(sample["answer"])
        prompt = PROMPT_TEMPLATE.format(question=question)
        return af.Challenge(  # type: ignore[attr-defined]
            env=self,
            prompt=prompt,
            extra={
                "expected": expected,        # réponse numérique attendue
                "sample": sample             # ← nouveau : question + answer brutes
            },
        )

    async def generate(self):  # returns Challenge (runtime), noqa: D401
        sample = self._random_samples(1)[0]
        chal = await self._make_challenge_from_sample(sample)
        print("PROMPT\n-----\n", chal.prompt)

        # Montre le sample HF complet (debug/trace)
        print("\nRAW SAMPLE\n----------\n", chal.extra["sample"])
        return chal

    async def generate_batch(self, n: int):
        samples = self._random_samples(n)
        challenges = [await self._make_challenge_from_sample(s) for s in samples]  # type: ignore[attr-defined]
        # Optionally, spin a background replenisher similar to ABD
        asyncio.create_task(self._replenish_samples_background())
        return challenges

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _answers_match(self, expected: str, predicted: str) -> bool:
        return _extract_number(expected) == _extract_number(predicted)

    async def evaluate(self, challenge, response):  # type: ignore[valid-type]
        expected = challenge.extra["expected"]
        predicted_raw = (response.response or "").strip().split("\n")[0]
        correct = self._answers_match(expected, predicted_raw)
        return af.Evaluation(  # type: ignore[attr-defined]
            env=self,
            score=1.0 if correct else 0.0,
            extra={"expected": expected, "got": predicted_raw},
        )

    # ------------------------------------------------------------------
    # Background sample replenishment (optional)
    # ------------------------------------------------------------------

    async def _replenish_samples_background(self, target_stock: int = val_config.TARGET_STOCK):
        """Keep a well-sized on-disk sample cache via SampleManager (if present)."""
        try:
            from affine.sample_manager import SampleManager  # lazy import
        except Exception:
            return  # sample manager not available in all deployments

        sm = SampleManager()
        stats = sm.get_stats(self.__class__.__name__)
        if stats.total >= target_stock:
            return

        need = target_stock - stats.total
        logger.info("[GSM8K] Background generating %d samples to replenish stock", need)
        new_chals = await self.generate_batch(need)
        sm.add_samples(  # type: ignore[attr-defined]
            self.__class__.__name__,
            [af.Sample.from_challenge(c, self.__class__.__name__) for c in new_chals],  # type: ignore[attr-defined]
        )
        logger.info("[GSM8K] Added %d new samples to stock", need)


# ────────────────────────────────────────────────────────────────────────────
# Public export – required for affine.envs auto-registration
# ────────────────────────────────────────────────────────────────────────────

ENV_CLASS = GSM8K

