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

# --------------------------------------------------------------------------- #
#                         Tunables and constants                              #
# --------------------------------------------------------------------------- #
DEFAULT_PROGRAM_EXECUTION_TIMEOUT = 5  # seconds
DATASET_SIZE = 57_300  # rows in the *train* split (July 2025 snapshot) :contentReference[oaicite:1]{index=1}
DATASET_ROWS_ENDPOINT = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=PrimeIntellect%2FSYNTHETIC-2-Base-Code"
    "&config=default&split=train&offset={offset}&length=1"
)

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
#                              Program runner                                 #
# --------------------------------------------------------------------------- #
class ProgramExecutor:
    """
    Executes untrusted Python programs with controlled stdin/stdout.

    Twostage strategy
    ------------------
    1) Run the user's script verbatim.
    2) If that yields **no output** *and* the script defines a   `solve()`
       function but has no mainguard, append a tiny runner:

           if __name__ == "__main__":
               import sys, json
               res = solve()
               if res is not None:
                   if isinstance(res, (list, tuple)):
                       print(*res, sep=" ")
                   else:
                       print(res)

       Then rerun.
    """

    _FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
    _HAS_MAIN_RE = re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]')

    def __init__(self, timeout: int = DEFAULT_PROGRAM_EXECUTION_TIMEOUT) -> None:
        self.timeout = timeout
        self._temp_files: List[str] = []
        self._lock = Lock()

    # -------------------------- Helpers ---------------------------------- #

    @staticmethod
    def _strip_fences(text: str) -> str:
        m = ProgramExecutor._FENCE_RE.search(text)
        return (m.group(1) if m else text).strip()

    @contextmanager
    def _tempfile(self, content: str, suffix: str = ".py"):
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as fh:
                fh.write(content)
                path = fh.name
            with self._lock:
                self._temp_files.append(path)
            yield path
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                finally:
                    with self._lock:
                        self._temp_files.remove(path)

    # ------------------------- Internal run ------------------------------ #

    def _run(self, code: str, stdin: str) -> Tuple[str, str]:
        with self._tempfile(code) as script:
            completed = subprocess.run(
                [sys.executable, script],
                input=stdin,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                encoding="utf-8",
            )
        return completed.stdout, completed.stderr

    # ------------------------- Public API -------------------------------- #

    def execute(self, raw_program: str, stdin: str) -> Tuple[str, str]:
        """Run *raw_program* with *stdin*; possibly add an auto‑runner."""
        program = self._strip_fences(raw_program)
        out, err = self._run(program, stdin)
        if out.strip() or err.strip():
            return out, err  # success or runtime error – but at least we saw output

        # Stage‑2: no output – try auto‑runner if `solve()` exists
        if "def solve" in program and not self._HAS_MAIN_RE.search(program):
            runner = (
                "\n\nif __name__ == \"__main__\":\n"
                "    res = solve()\n"
                "    if res is not None:\n"
                "        import sys\n"
                "        if isinstance(res, (list, tuple)):\n"
                "            print(*res, sep=\" \")\n"
                "        else:\n"
                "            print(res)\n"
            )
            out, err = self._run(program + runner, stdin)
        return out, err


# --------------------------------------------------------------------------- #
#                           Utility functions                                 #
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    """Trim trailing blank lines and per‑line trailing spaces."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


async def _fetch_dataset_row(offset: int) -> Dict[str, Any] | None:
    url = DATASET_ROWS_ENDPOINT.format(offset=offset)
    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.get(url, timeout=30) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
        except Exception:
            return None
    rows = payload.get("rows", [])
    return rows[0]["row"] if rows else None


# --------------------------------------------------------------------------- #
#                              Affine Env                                     #
# --------------------------------------------------------------------------- #


class DEDUCTION(af.BaseEnv):
    def __init__(self):
        super().__init__()
        self._executor = ProgramExecutor()
        # --- new buffer logic ---
        self._buffer: deque[Dict[str, Any]] = deque()
        self._buffer_size = 10
        # ------------------------

    async def _fill_buffer(self) -> None:
        """
        Pull rows until we have `buffer_size` in our queue.
        Skips any failed fetches.
        """
        while len(self._buffer) < self._buffer_size:
            offset = random.randint(0, DATASET_SIZE - 1)
            row = await _fetch_dataset_row(offset)
            if row is not None:
                self._buffer.append(row)

    async def random_sample(self) -> Dict[str, Any] | None:
        """
        Return one sample from the local buffer.
        If empty, top up first.
        """
        if not self._buffer:
            await self._fill_buffer()
        return self._buffer.popleft() if self._buffer else None

    # ----------------------------- Env API -------------------------------- #

    async def generate(self) -> af.Challenge:
        af.logger.trace("Generating new SYNTHETIC‑2 coding challenge")
        sample = await self.random_sample()
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
        program = ProgramExecutor._strip_fences(raw_reply)
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
