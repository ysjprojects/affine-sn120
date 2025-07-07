"""
Affine validation module - handles model validation including hot status checks
"""
import asyncio, aiohttp
from typing import Dict, Any, Optional, TYPE_CHECKING, Union, List
import datetime as dt
import os

import logging
import bittensor as bt

# Use module-specific logger but inherit config from the main "affine" logger
logger = logging.getLogger("affine.utils")

if TYPE_CHECKING:
    from . import Miner
else:
    Miner = None


async def get_chutes_info(model: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
    """Fetches additional information about a model from the Chutes.ai API."""
    from affine import get_conf
    key = get_conf("CHUTES_API_KEY")
    if not key:
        return None
    url = f"https://api.chutes.ai/chutes/{model.replace('/','_')}"
    try:
        async with session.get(url, headers={"Authorization": key}) as r:
            return await r.json() if r.status == 200 else None
    except Exception:
        return None


async def is_model_hot(model: str, session: aiohttp.ClientSession) -> bool:
    try:
        info = await get_chutes_info(model, session)
        if info is not None:
            return bool(info.get("hot", True))
        return True
    except Exception as exc:
        logger.debug("is_model_hot fallback to True for %s: %s", model, exc)
        return True


async def get_chutes_model_info(model: str) -> dict:
    """Fetches vLLM config for a model from the Chutes.ai API."""
    from affine import get_conf
    api_key = get_conf("CHUTES_API_KEY")
    if not api_key:
        print(f"No Chutes API key available for model info fetch")
        return None
    url = f"https://api.chutes.ai/guess/vllm_config?model={model}"
    headers = {"Authorization": api_key}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ vLLM config fetched for model '{model}'")
                    return data
                else:
                    print(f"Failed to fetch vLLM config: HTTP {response.status}")
                    return None
    except Exception as e:
        print(f"Error fetching vLLM config: {e}")
        return None

async def get_model_size(model: str) -> dict:
    chutes_data = await get_chutes_model_info(model)

    model_size = chutes_data['total_model_size']

    return model_size

async def validate_miners_hot(miners_dict: Dict[int, "Miner"], require_hot: bool = True) -> Dict[int, "Miner"]:
    """Filter *miners_dict* retaining only entries whose model is hot.

    Parameters
    ----------
    miners_dict : dict[int, Miner]
        Mapping of UID → Miner.
    require_hot : bool, default True
        If *True*, only miners whose model is hot (or missing) are kept; if
        *False*, the original dict is returned unchanged.

    Returns
    -------
    dict[int, Miner]
    """
    if not miners_dict:
        return miners_dict

    async with aiohttp.ClientSession() as sess:
        checks = await asyncio.gather(
            *[is_model_hot(m.model, sess) if m.model else True for m in miners_dict.values()],
            return_exceptions=True,
        )

    def _keep(flag, miner):
        return (not require_hot) or (not miner.model) or (isinstance(flag, bool) and flag)

    return {uid: miner for (uid, miner), flag in zip(miners_dict.items(), checks) if _keep(flag, miner)}



# ── timestamp helpers ───────────────────────────────────────────────────────

async def _block_timestamp_ms(block: int, netuid: int = 120) -> Optional[int]:
    """Return the Unix timestamp (milliseconds) for a given block number.

    Parameters
    ----------
    block   : int
        Block height on the Bittensor chain.
    netuid  : int, default 120
        Network UID (subnet) to query. Defaults to 120 – the common text-prompt
        subnet on main-net.

    Notes
    -----
    The function queries the on-chain `Timestamp.Now` pallet storage at
    `block_hash` corresponding to *block*. It returns *None* if any RPC call
    fails.
    """
    archive_endpoint = os.getenv("BT_ARCHIVE_ENDPOINT")

    def _new_subtensor(endpoint: Optional[str] = None):
        """Create AsyncSubtensor with flexible arg names for cross-version support."""
        if endpoint is None:
            return bt.async_subtensor()
        # Try preferred kwarg first (newer bittensor)
        try:
            return bt.async_subtensor(network=endpoint)
        except TypeError:
            pass
        # Older versions may accept `endpoint`, fall back silently
        try:
            return bt.async_subtensor(endpoint=endpoint)  # type: ignore
        except TypeError:
            # Ultimate fallback: create default instance then monkey-patch chain_endpoint
            st = bt.async_subtensor()
            try:
                st.chain_endpoint = endpoint  # type: ignore
            except Exception:
                logger.warning("Unable to set custom endpoint on AsyncSubtensor; using default")
            return st

    async def _timestamp_via_subtensor(st) -> Optional[int]:
        try:
            block_hash = await st.get_block_hash(block)
            # Try storage query first
            try:
                ts_obj = await st.substrate.query("Timestamp", "Now", block_hash=block_hash)
                return int(ts_obj.value)
            except Exception as e_state:
                logger.debug("State query failed for block %s on %s: %s", block, st.chain_endpoint, e_state)

            # Fallback: fetch block raw and inspect extrinsics
            try:
                blk = await st.substrate.get_block(block_hash=block_hash)
            except TypeError:
                # Older async_substrate versions accept no kwargs
                blk = await st.substrate.get_block(block_hash)

            extrinsics = blk.get("extrinsics") or []
            for ext in extrinsics:
                call = ext.get("call") or {}
                if (call.get("call_module") == "Timestamp" or call.get("module_name") == "Timestamp"):
                    params = call.get("args") or ext.get("params") or []
                    if params:
                        first = params[0]
                        if isinstance(first, dict):
                            val = first.get("value") or first.get("value_raw") or first.get("Value")
                        else:
                            val = first
                        return int(val)
            return None
        except Exception as e:
            logger.debug("Subtensor endpoint %s unable to provide timestamp for block %s: %s", st.chain_endpoint, block, e)
            return None

    # 1) Try default endpoint ------------------------------------------------
    st = _new_subtensor()
    await st.initialize()
    ts_ms = await _timestamp_via_subtensor(st)
    if ts_ms is not None:
        return ts_ms

    # 2) Fallback to archive endpoint if provided ---------------------------
    if archive_endpoint:
        logger.debug("Retrying block %s on archive endpoint %s", block, archive_endpoint)
        st_arch = _new_subtensor(archive_endpoint)
        await st_arch.initialize()
        return await _timestamp_via_subtensor(st_arch)

    return None

def _parse_any_timestamp(value: Union[str, int, float, None]) -> Optional[dt.datetime]:
    """Best-effort conversion of various timestamp formats to *aware* datetime.

    Supports:
      • ISO-8601 strings (with or without trailing *Z*)
      • Unix seconds or milliseconds (int|float)
    Returns *None* if parsing fails.
    """
    if value is None:
        return None

    # Numeric (seconds or milliseconds)
    if isinstance(value, (int, float)):
        ts = float(value)
        # Heuristic: >1e12 → milliseconds
        if ts > 1e12:
            ts /= 1_000.0
        try:
            return dt.datetime.utcfromtimestamp(ts).replace(tzinfo=dt.timezone.utc)
        except Exception:
            return None

    # ISO-formatted string
    if isinstance(value, str):
        try:
            # Replace trailing Z with +00:00 so that fromisoformat works
            v = value.rstrip("Zz") + ("+00:00" if value.endswith(("Z", "z")) else "")
            return dt.datetime.fromisoformat(v).astimezone(dt.timezone.utc)
        except Exception:
            # Fallback: treat as float seconds
            try:
                return dt.datetime.utcfromtimestamp(float(value)).replace(tzinfo=dt.timezone.utc)
            except Exception:
                return None

    return None


async def compare_commit_deploy(miner: "Miner", netuid: int = 120) -> Optional[float]:
    """Compute *deploy_time − commit_time* in seconds for a given *miner*.

    The function fetches:
      1. The on-chain block timestamp for the miner's last reveal commit
      2. The deployment timestamp exposed in the miner's chute metadata
    It returns the time delta in **seconds** (positive ⇒ deployment happened
    *after* commit). It may return *None* if either timestamp is unavailable.
    """
    # Safeguards ----------------------------------------------------------------
    if miner.block is None or miner.chute is None:
        logger.debug(
            "Skip delta: miner %s (uid %s) missing %s",
            getattr(miner, "model", "?"),
            getattr(miner, "uid", "?"),
            "block" if miner.block is None else "chute info",
        )
        return None

    commit_ts_ms = await _block_timestamp_ms(miner.block, netuid)
    if commit_ts_ms is None:
        logger.debug("No block timestamp for uid %s (block %s)", miner.uid, miner.block)
        return None

    commit_dt = dt.datetime.utcfromtimestamp(commit_ts_ms / 1_000).replace(tzinfo=dt.timezone.utc)

    # Keys to look for – snake_case and camelCase variants
    _TS_KEYS = (
        "updated", "updated_at", "updatedAt",
        "created", "created_at", "createdAt",
        "timestamp", "deployed_at", "deployedAt",
        "collected_at", "collectedAt",
    )

    deploy_raw = None
    for key in _TS_KEYS:
        if key in miner.chute:                # top-level
            deploy_raw = miner.chute[key]
            break

    # fallback – sometimes under miner.chute["image"]
    if deploy_raw is None and isinstance(miner.chute.get("image"), dict):
        img = miner.chute["image"]
        for key in _TS_KEYS:
            if key in img:
                deploy_raw = img[key]
                break

    if deploy_raw is None:
        logger.debug("No deployment timestamp field found for uid %s (keys=%s)", miner.uid, list(miner.chute.keys()))
        return None

    deploy_dt = _parse_any_timestamp(deploy_raw)
    if deploy_dt is None:
        logger.debug("Failed to parse deployment timestamp '%s' for uid %s", deploy_raw, miner.uid)
        return None

    # Δt in seconds (positive if deploy later than commit)
    return (deploy_dt - commit_dt).total_seconds()

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Miner-level helper (restored)
# ---------------------------------------------------------------------------


async def validate_miners_hot(miners_dict: Dict[int, "Miner"], require_hot: bool = True) -> Dict[int, "Miner"]:
    """Filter *miners_dict* retaining only entries whose model is hot.

    Parameters
    ----------
    miners_dict : dict[int, Miner]
        Mapping of UID → Miner.
    require_hot : bool, default True
        If *True*, only miners whose model is hot (or missing) are kept; if
        *False*, the original dict is returned unchanged.

    Returns
    -------
    dict[int, Miner]
    """
    if not miners_dict:
        return miners_dict

    async with aiohttp.ClientSession() as sess:
        checks = await asyncio.gather(
            *[is_model_hot(m.model, sess) if m.model else True for m in miners_dict.values()],
            return_exceptions=True,
        )

    def _keep(flag, miner):
        return (not require_hot) or (not miner.model) or (isinstance(flag, bool) and flag)

    return {uid: miner for (uid, miner), flag in zip(miners_dict.items(), checks) if _keep(flag, miner)}

# ---------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
# Generic tiny utilities reused across the code-base
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import json, subprocess, tempfile, sys
from contextlib import contextmanager
from typing import List, Dict, Any

# Directory holding per-environment JSONL sample stocks.
SAMPLES_DIR: Path = Path.home() / ".affine" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def _sample_file(env_name: str) -> Path:
    """Return the JSONL file path storing samples for *env_name*."""
    return SAMPLES_DIR / f"{env_name.lower()}.jsonl"

# ---------------------------------------------------------------------------
# Minimal sample persistence helpers (≈20 LoC)
# ---------------------------------------------------------------------------

def load_samples(env_name: str) -> List[Dict[str, Any]]:
    """Load all samples for *env_name* from disk.

    Each line is a JSON object with at least a `prompt` key; other keys are
    stored verbatim and become `Challenge.extra`.
    """
    fp = _sample_file(env_name)
    if not fp.exists():
        return []
    try:
        return [json.loads(line) for line in fp.read_text().splitlines() if line.strip()]
    except Exception:
        # Corrupted file → discard
        fp.unlink(missing_ok=True)
        return []

def save_samples(env_name: str, samples: List[Dict[str, Any]]) -> None:
    """Overwrite sample file for *env_name* with *samples* (JSONL)."""
    fp = _sample_file(env_name)
    with fp.open("w", encoding="utf-8") as f:
        for s in samples:
            json.dump(s, f)
            f.write("\n")

# ---------------------------------------------------------------------------
# Tiny ProgramExecutor (sandboxed `python -` runner) – ~15 LoC
# ---------------------------------------------------------------------------

class ProgramExecutor:
    """Execute a short Python *program* feeding *input_data* to stdin.

    Returns a tuple (stdout, stderr). Execution is killed after *timeout*
    seconds. Temporary files are cleaned up automatically.
    """

    DEFAULT_TIMEOUT = int(os.getenv("AFFINE_EXEC_TIMEOUT", "30"))

    @staticmethod
    def execute(program: str, input_data: str = "", timeout: int | None = None) -> tuple[str, str]:
        timeout = timeout or ProgramExecutor.DEFAULT_TIMEOUT
        # Strip markdown fences if any
        if program.startswith("```"):
            program = program.strip("`\n python")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(program)
            tmp_path = tmp.name
        try:
            completed = subprocess.run([sys.executable, tmp_path],
                                       input=input_data,
                                       text=True,
                                       capture_output=True,
                                       timeout=timeout)
            return completed.stdout, completed.stderr
        except subprocess.TimeoutExpired:
            return "", "Program execution timed out"
        except Exception as e:  # pragma: no cover
            return "", f"Execution error: {e}"
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass