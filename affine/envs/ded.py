from __future__ import annotations

import os
import re
import sys
import json
import random
import aiohttp
import asyncio
import tempfile
import subprocess
import affine as af
from threading import Lock
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

# -------------------------------- Helpers -------------------------------- #
def _to_str(x) -> str:
    """
    Canonicalise any JSON‑serialisable test‑case payload to a single
    newline‑delimited string suitable for feeding to `stdin`.
    """
    if isinstance(x, str):
        return x                     # already a single line
    if isinstance(x, (bytes, bytearray)):
        return x.decode()            # rare, but be safe
    if isinstance(x, list):
        # Recursively stringify nested lists and join with newlines
        return "\n".join(_to_str(e) for e in x)
    # Dicts / numbers / other scalars → JSON text
    return json.dumps(x, ensure_ascii=False)

# --------------------------------------------------------------------------- #
#                           Utility functions                                 #
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """Trim trailing blank lines and per‑line trailing spaces."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


# --------------------------------------------------------------------------- #
#                              Affine Env                                     #
# --------------------------------------------------------------------------- #
class DEDUCTION(af.BaseEnv):
    def __init__(self):
        super().__init__()
        self._executor = af.utils.ProgramExecutor()
        self._data = af.utils.BufferedDataset(
            dataset_name="PrimeIntellect/SYNTHETIC-2-Base-Code",
            total_size=57_300,
            buffer_size=5,
            max_batch=5,
        )

    # ----------------------------- Env API -------------------------------- #
    async def generate(self) -> af.Challenge:
        af.logger.trace("Generating new SYNTHETIC‑2 coding challenge")
        sample = await self._data.get()
        if sample is None:
            raise RuntimeError("Failed to fetch dataset row")

        # Mild redundancy to guarantee correct formatting.
        extra_hint = (
            "\n\n---\n"
            "⚠️ **Instructions** ⚠️\n"
            "Write a complete **Python 3** program that\n"
            "• reads *all* input from **STDIN** (using `input()` / `sys.stdin`),\n"
            "• writes *only* the required answer(s) to **STDOUT** using `print`,\n"
            "• contains no additional prompts or debug text, and\n"
            "• is returned as a single ```python … ``` fenced block.\n"
        )
        prompt = sample["prompt"].rstrip() + extra_hint
        return af.Challenge(env=self, prompt=prompt, extra=sample)

    async def evaluate(
        self, challenge: af.Challenge, response: af.Response
    ) -> af.Evaluation:
        af.logger.trace("Starting evaluation of the challenge.")
        raw_reply = response.response
        program = self._executor._strip_fences(raw_reply)
        af.logger.trace(f"Stripped program from response: {program[:50]}...")

        # ---------------- Verification info ---------------------------- #
        sample = challenge.extra or {}
        ver_raw = sample.get("verification_info") or sample.get("test_cases")
        af.logger.trace(f"Verification raw data: {ver_raw[:50]}...")
        try:
            ver_json = json.loads(ver_raw) if isinstance(ver_raw, str) else ver_raw
            af.logger.trace("Parsed verification info JSON successfully.")
        except Exception as err:
            af.logger.trace(f"Failed to parse verification info JSON: {err}")
            return af.Evaluation(
                env=self,
                score=0.0,
                feedback=f"Invalid verification_info JSON: {err}",
            )

        # Some rows nest the actual data under 'test_cases'
        if "test_cases" in ver_json:
            nested = ver_json["test_cases"]
            ver_json = json.loads(nested) if isinstance(nested, str) else nested
            af.logger.trace("Extracted nested test cases from verification info.")

        inputs: List[str] = ver_json.get("inputs") or []
        outputs: List[str] = ver_json.get("outputs") or []
        af.logger.trace(f"Extracted {len(inputs)} inputs and {len(outputs)} outputs.")

        if not inputs:
            af.logger.trace("No public test cases available.")
            return af.Evaluation(
                env=self, score=0.0, feedback="No public test cases available"
            )
        if outputs and len(inputs) != len(outputs):
            af.logger.trace("Mismatch between number of inputs and outputs.")
            return af.Evaluation(
                env=self,
                score=0.0,
                feedback="Mismatch between #inputs and #outputs in verification data",
            )

        # ----------------- Run programme on tests ---------------------- #
        loop = asyncio.get_running_loop()
        passed, total = 0, len(inputs)
        details: List[Dict[str, Any]] = []

        for i, raw_inp in enumerate(inputs):
            raw_exp = outputs[i] if outputs else None

            # 👇 NEW: canonicalise
            inp = _to_str(raw_inp)
            if not inp.endswith("\n"):
                inp += "\n"                  # many user programs rely on a final EOL
            exp = _to_str(raw_exp) if raw_exp is not None else None

            try:
                out, err = await loop.run_in_executor(
                    None, lambda: self._executor.execute(program, inp)
                )
            except subprocess.TimeoutExpired:
                out, err = "", "TIMEOUT"

            ok_run   = err.strip() == ""
            out_norm = _normalize(out)
            exp_norm = _normalize(exp) if exp is not None else None
            correct  = ok_run and (exp_norm is None or out_norm == exp_norm)
            if correct:
                passed += 1
                af.logger.trace(f"Test case {i+1} passed.")
            else:
                af.logger.trace(f"Test case {i+1} failed. Output: {out_norm}, Expected: {exp_norm}")

            details.append(
                {
                    "input": inp,
                    "expected": exp_norm,
                    "got": out_norm,
                    "stderr": err.strip(),
                    "passed": correct,
                }
            )

        score = passed / total
        feedback = json.dumps(
            {"passed": passed, "total": total, "tests": details}, ensure_ascii=False
        )
        af.logger.trace(f"Evaluation completed with score: {score}")
        return af.Evaluation(env=self, score=score, feedback=feedback)
