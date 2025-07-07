"""affine.invalidation
~~~~~~~~~~~~~~~~~~~~~
Minimal, single-file persistence layer tracking miners' state (commit hash,
hotkey, model verification…) in a local JSON file under ``~/.affine``.

The public coroutine :func:`invalidate` orchestrates everything:

    entry = await invalidate(uid, commit_hash, hotkey, block, model_name)

Typical caller logic::

    entry = await invalidate(uid, commit, hk, block, model)
    if entry.is_model_verified and await utils.is_model_hot(entry.model_name):
        miners_valid.append(entry)

The goal is to keep blocking file-I/O off the event-loop, hence the heavy use
of :pyfunc:`asyncio.to_thread`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional
import os
import functools

from pydantic import BaseModel, Field

logger = logging.getLogger("affine.invalidate")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH: Path = Path.home() / ".affine" / "data.json"
ELO_PATH: Path = Path.home() / ".affine" / "results" / "elo.json"

# Ensure directories always exist so later writes do not race.
for _p in (DATA_PATH.parent, ELO_PATH.parent):
    _p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    """Load JSON *path* safely, returning an empty dict on error."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover – corrupted file
        logger.warning("Unable to parse %s – resetting (%s)", path, exc)
        return {}


def atomic_write(path: Path, data: dict) -> None:
    """Atomically dump *data* as JSON to *path* using a ``.tmp`` swap file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

class DataEntry(BaseModel):
    """Pydantic model persisted per UID."""

    uid: int
    hotkey: str
    hash: Optional[str] = None  # Last revealed commit hash
    block: Optional[int] = None  # Block height of last reveal
    model_name: Optional[str] = Field(None, alias="model")
    finetune_verified: Optional[bool] = None  # Heavy verification result
    timestamp_hash: Optional[int] = None      # Epoch seconds when *hash* changed
    timestamp_finetune: Optional[int] = None  # When finetune check ran

    model_config = {"populate_by_name": True, "extra": "ignore"}

# ──────────────────────────────────────────────────────────────────────────────
# Internal DB cache (lazy-loaded)
# ──────────────────────────────────────────────────────────────────────────────
_DATA: dict[int, DataEntry] | None = None
_DB_LOCK = asyncio.Lock()

# ── new: global concurrency guard for heavy verification  ────────────────────
_VERIFY_SEMA = asyncio.Semaphore(int(os.getenv("AFFINE_FT_CONCURRENCY", "1")))

# ── new: background worker handling pending fine-tune tests sequentially ─────
_FT_WORKER_STARTED = False

async def _finetune_worker_loop() -> None:
    """Continuously process the first UID whose *finetune_verified* is *None*."""
    while True:
        try:
            db = await _db()
            # Find first pending entry (breadth-first).
            pending = next((e for e in db.values() if e.finetune_verified is None), None)
            if pending is None:
                await asyncio.sleep(30)
                continue
            # Run verification (sequential thanks to _VERIFY_SEMA).
            await _verify_finetune(pending.uid, force=True)
        except Exception as exc:  # pragma: no cover
            logger.debug("Background finetune worker error: %s", exc)
        # Slight delay to yield control and avoid tight loop
        await asyncio.sleep(5)


def _ensure_ft_worker() -> None:
    """Spin up the background worker once per process (best-effort)."""
    global _FT_WORKER_STARTED
    if _FT_WORKER_STARTED:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_finetune_worker_loop(), name="affine-finetune-worker")
        _FT_WORKER_STARTED = True
        logger.debug("Started background finetune worker")
    except RuntimeError:
        # No running event-loop (can happen in import time) – worker will start
        # on the first call inside an async context.
        pass


def _load_db_sync() -> dict[int, DataEntry]:
    raw = load_json(DATA_PATH)
    return {int(k): DataEntry.model_validate(v) for k, v in raw.items()}


def _dump_db_sync(db: dict[int, DataEntry]) -> None:
    serial = {str(k): v.model_dump(by_alias=True) for k, v in db.items()}
    atomic_write(DATA_PATH, serial)


async def _db() -> dict[int, DataEntry]:
    global _DATA
    async with _DB_LOCK:
        if _DATA is None:
            _DATA = await asyncio.to_thread(_load_db_sync)
        return _DATA


async def _flush() -> None:
    async with _DB_LOCK:
        assert _DATA is not None
        await asyncio.to_thread(_dump_db_sync, _DATA)

# ──────────────────────────────────────────────────────────────────────────────
# Public low-level helpers (all async)
# ──────────────────────────────────────────────────────────────────────────────

async def ensure_entry(uid: int, hotkey: str) -> DataEntry:
    """Return a *DataEntry*, creating it if it does not exist."""
    db = await _db()
    if uid not in db:
        db[uid] = DataEntry(uid=uid, hotkey=hotkey, model="")
        await _flush()
        logger.debug("Created entry for uid %s", uid)
    return db[uid]


async def update_hotkey(uid: int, new_hotkey: str) -> bool:
    entry = await ensure_entry(uid, new_hotkey)
    if entry.hotkey == new_hotkey:
        return False
    entry.hotkey = new_hotkey
    await _flush()
    logger.debug("Hotkey updated for uid %s -> %s", uid, new_hotkey)
    return True


async def update_commit(
    uid: int,
    new_hash: str,
    block: int,
    model_name: str,
) -> bool:
    entry = await ensure_entry(uid, "")
    changed = False
    for field, val in {
        "hash": new_hash,
        "block": block,
        "model_name": model_name,
    }.items():
        if getattr(entry, field) != val:
            setattr(entry, field, val)
            changed = True
    if changed:
        entry.timestamp_hash = int(time.time())
        entry.finetune_verified = None  # trigger re-check next cycle
        await _flush()
        logger.debug("Commit info updated for uid %s", uid)
    return changed


# ── heavy fine-tune verification ────────────────────────────────────────────

_BASE_MODEL_ENV = "AFFINE_BASE_MODEL"

# ── heavy helper functions (adapted from backupaffine/Affine1/model_detector.py)

# NOTE: the following functions import large ML libraries *inside* their body
# so that importing affine.invalidation remains lightweight.


def _load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cpu")
    return model, AutoTokenizer.from_pretrained(model_name)


def _compare_weights(model1, model2, max_layers: int = 10):
    import torch
    import numpy as np

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    diffs = []
    count = 0

    for name in params1:
        if name in params2 and params1[name].shape == params2[name].shape:
            with torch.no_grad():
                p1 = params1[name].cpu()
                p2 = params2[name].cpu()
                diff = torch.norm(p1 - p2) / (torch.norm(p1) + 1e-8)
                diffs.append(diff.item())
                count += 1
                if count >= max_layers:
                    break

    if not diffs:
        return float("nan"), []
    return float(np.mean(diffs)), diffs


def _compare_logits_kl(model1, tok1, model2, tok2, prompts):
    import torch
    import torch.nn.functional as F
    import numpy as np

    kl_divs = []
    for prompt in prompts:
        inputs1 = tok1(prompt, return_tensors="pt").to(model1.device)
        inputs2 = tok2(prompt, return_tensors="pt").to(model2.device)

        with torch.no_grad():
            logits1 = model1(**inputs1).logits[:, -1, :]
            logits2 = model2(**inputs2).logits[:, -1, :]

        if logits1.shape != logits2.shape:
            continue

        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)

        kl = F.kl_div(p1, p2, reduction="batchmean", log_target=False)
        kl_divs.append(kl.item())

    if not kl_divs:
        return float("nan"), []
    return float(np.mean(kl_divs)), kl_divs


def _is_finetuned_model(base_model_name: str, test_model_name: str) -> Optional[bool]:

    import numpy as np

    try:
        model1, tok1 = _load_model(base_model_name)
        model2, tok2 = _load_model(test_model_name)
    except Exception as e:
        logger.debug("Model loading failed for finetune check %s vs %s: %s", base_model_name, test_model_name, e)
        return None

    try:
        weight_score, _ = _compare_weights(model1, model2)

        prompts = [
            "What is the capital of France?",
            "Explain the law of gravity.",
            "Write a short story about a robot and a dragon.",
        ]

        kl_score, _ = _compare_logits_kl(model1, tok1, model2, tok2, prompts)
    except Exception as e:
        logger.debug("Comparison failed for %s vs %s: %s", base_model_name, test_model_name, e)
        return None

    if np.isnan(weight_score) and np.isnan(kl_score):

        return False
    if weight_score < 0.01 and kl_score < 0.5:
        return True  
    if weight_score < 0.15:
        return True  
    if weight_score >= 0.15 and kl_score > 5.0:
        return False  

    return None  


async def _verify_finetune(uid: int, *, force: bool = False) -> bool:
    # Serialize heavy operations via semaphore to keep memory under control
    async with _VERIFY_SEMA:
        entry = await ensure_entry(uid, "")
        if entry.finetune_verified is not None and not force:
            return False  # already done, skip
        if not entry.model_name:
            return False

        base_model = os.getenv(_BASE_MODEL_ENV, "Qwen/Qwen3-8B")

        loop = asyncio.get_running_loop()
        verdict = await loop.run_in_executor(
            None,
            functools.partial(_is_finetuned_model, base_model, entry.model_name),
        )

        # Only persist True / False; leave None so we can retry later
        if verdict is not None:
            entry.finetune_verified = bool(verdict)
            entry.timestamp_finetune = int(time.time())
            await _flush()
            logger.debug("Finetune verification uid %s → %s", uid, verdict)
            return True
        else:
            logger.debug("Finetune verification uid %s inconclusive; will retry", uid)
            return False

# ──────────────────────────────────────────────────────────────────────────────
# High-level orchestration
# ──────────────────────────────────────────────────────────────────────────────

async def reset_elo(hotkey: str) -> None:
    """Remove *hotkey* from Elo table (disk)."""

    def _sync():
        table = load_json(ELO_PATH)
        if hotkey in table:
            table.pop(hotkey)
            atomic_write(ELO_PATH, table)
            logger.debug("Elo reset for %s", hotkey)

    await asyncio.to_thread(_sync)

async def invalidate_if_needed(
    uid: int,
    commit_hash: str,
    hotkey: str,
    block: int,
    model_name: str,
) -> DataEntry:
    changed_hotkey = await update_hotkey(uid, hotkey)
    changed_commit = await update_commit(uid, commit_hash, block, model_name)

    if changed_commit:
        # reset Elo because behaviour may change & force new finetune test
        await reset_elo(hotkey)

    await _verify_finetune(uid, force=changed_commit)
    return await ensure_entry(uid, hotkey)


async def invalidate(
    uid: int,
    commit_hash: str | None,
    hotkey: str,
    block: int | None,
    model_name: str | None,
) -> DataEntry:
    """Public single-entry point used by the validator loop."""
    # Ensure background worker is running inside current event-loop.
    _ensure_ft_worker()

    commit_hash = commit_hash or ""
    block = block or 0
    model_name = model_name or ""
    entry = await invalidate_if_needed(uid, commit_hash, hotkey, block, model_name)

    # ── update global incentive state -------------------------------------
    try:
        from .state import upsert_miner  # local import to avoid heavy deps during cold start
        upsert_miner(hotkey, uid, block or 0)
    except Exception as _exc:  # pragma: no cover – keep validation resilient
        logger.debug("state.upsert_miner failed: %s", _exc)

    return entry

# ──────────────────────────────────────────────────────────────────────────────
# Doctest
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import asyncio as _a

    async def _demo():
        e1 = await invalidate(1, "deadbeef", "hk", 42, "model/x")
        e2 = await invalidate(1, "deadbeef", "hk", 42, "model/x")  # no-op
        assert e1.finetune_verified is not None and e1.finetune_verified == e2.finetune_verified
        print("✓ demo", e1.model_dump())

    _a.run(_demo()) 