"""
Affine deployment utilities – push model code to HF, deploy via Chutes, and commit on Bittensor.
"""
from __future__ import annotations

import os, random, asyncio, json
from pathlib import Path
from typing import Optional

import aiohttp
import bittensor as bt
from huggingface_hub import HfApi

from . import logger  # shared TRACE logger

__all__ = [
    "DeploymentConfig",
    "deploy_model",
]


class DeploymentConfig:
    """Lightweight config holder – values are taken from ~/.affine/config.env if omitted."""

    def __init__(
        self,
        chutes_api_key: Optional[str] = None,
        hf_user: Optional[str] = None,
        hf_token: Optional[str] = None,
        chute_user: Optional[str] = None,
        wallet_cold: Optional[str] = None,
        wallet_hot: Optional[str] = None,
    ) -> None:
        env = os.getenv
        self.chutes_api_key = chutes_api_key or env("CHUTES_API_KEY")
        self.hf_user = hf_user or env("HF_USER")
        self.hf_token = hf_token or env("HF_TOKEN")
        self.chute_user = chute_user or env("CHUTE_USER", self.hf_user)
        self.wallet_cold = wallet_cold or env("BT_WALLET_COLD", "default")
        self.wallet_hot = wallet_hot or env("BT_WALLET_HOT", "default")


# ── Public orchestration helper ─────────────────────────────────────────
async def deploy_model(
    local_path: str,
    cfg: DeploymentConfig | None = None,
    *,
    existing_repo: str | None = None,
    hardcoded_round_start: int | None = None,
    hide_hf: bool = True,
) -> str:
    """Deploys a model following the "commit → reveal" schedule.

    Steps:
    1. Pushes the code to HF (private repo).
    2. Computes the next *round* (multiple of 10) and the *reveal* block (round_start-5).
    3. Emits the `set_reveal_commitment` extrinsic to reveal at block `reveal_block`.
    4. Waits for the arrival of block `reveal_block`.
    5. Makes the repo public and then deploys it on Chutes.
    6. Warm-up and returns the repo ID.
    """

    cfg = cfg or DeploymentConfig()
    # Ensure that sensitive fields are properly defined (otherwise raise ValueError).
    def _req(val: str | None, name: str) -> str:
        if val is None:
            raise ValueError(f"{name} must be provided via env var or argument")
        return val

    hf_user        = _req(cfg.hf_user, "HF_USER")
    hf_token       = _req(cfg.hf_token, "HF_TOKEN")
    chutes_api_key = _req(cfg.chutes_api_key, "CHUTES_API_KEY")
    chute_user     = _req(cfg.chute_user, "CHUTE_USER")

    repo_id = existing_repo or _generate_repo_name(hf_user)

    # 1) Manage HF storage
    if existing_repo is None:
        # New repo ⇒ push and choose visibility based on hide_hf
        revision = await _push_to_huggingface(local_path, repo_id, hf_token, private=hide_hf)
    else:
        # Repo already exists ⇒ optionally hide it
        if hide_hf:
            api_tmp = HfApi(token=hf_token)
            try:
                api_tmp.update_repo_visibility(repo_id=repo_id, private=True)
            except Exception:
                pass
        revision = _get_hf_revision(repo_id, hf_token)

    # 2) Compute chain timing
    sub = bt.subtensor()
    current_block = sub.get_current_block()
    K = 10
    round_start = hardcoded_round_start or ((current_block // K) + 1) * K
    reveal_block = round_start - 5
    if reveal_block <= current_block:
        # Edge-case : if we've already passed, shift by one round
        round_start += K
        reveal_block = round_start - 5
    blocks_until_reveal = max(reveal_block - current_block, 1)
    logger.info(
        "Timing → current=%d, reveal=%d (Δ=%d), round_start=%d",
        current_block,
        reveal_block,
        blocks_until_reveal,
        round_start,
    )

    # 3) Commit on chain (with delayed reveal) – include revision
    await _commit_on_chain(repo_id, revision, blocks_until_reveal, cfg.wallet_cold, cfg.wallet_hot)

    # 4) Wait until reveal_block is reached (simple polling toutes les 6 s)
    await _wait_until_block(reveal_block, sub)

    # 5) Make repo public (only if it was hidden) and deploy to Chutes
    if hide_hf:
        await _make_repo_public(repo_id, hf_token)
    await _deploy_to_chutes(repo_id, chute_user, chutes_api_key, revision=revision)

    # 6) Warm-up
    await _warmup(repo_id, chutes_api_key)

    logger.info("✅ Deployment pipeline completed: %s", repo_id)
    return repo_id


# ── Individual steps (private) ──────────────────────────────────────────

def _generate_repo_name(hf_user: str) -> str:
    return f"{hf_user}/Affine-{random.randint(1_000_000, 9_999_999)}"


async def _push_to_huggingface(local_path: str, repo_id: str, token: str, *, private: bool = False) -> str:
    logger.info("📤 Pushing %s to HF repo %s (private=%s)", local_path, repo_id, private)
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        if private:
            # ensure that the repo is private until the reveal
            try:
                api.update_repo_visibility(repo_id=repo_id, private=True)
            except Exception:
                pass
        logger.debug("HF repo exists, will upload files")
    except Exception:
        logger.debug("Creating HF repo %s", repo_id)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    files: list[str] = []
    for root, _, fnames in os.walk(local_path):
        if ".cache" in root or any(p.startswith(".") for p in root.split(os.sep)):
            continue
        files.extend(os.path.join(root, f) for f in fnames if not f.startswith(".") and not f.endswith(".lock"))

    async def _upload(path: str):
        rel = os.path.relpath(path, local_path)
        # Use kwargs to satisfy type checker and wrap in lambda.
        await asyncio.to_thread(lambda: api.upload_file(path_or_fileobj=path, path_in_repo=rel, repo_id=repo_id, repo_type="model"))  # type: ignore[arg-type]
        logger.debug("↑ %s", rel)

    await asyncio.gather(*(_upload(p) for p in files))
    logger.info("HF push done (%d files)", len(files))
    # Return the current revision
    return _get_hf_revision(repo_id, token)

# -- HF helpers --------------------------------------------------------

def _get_hf_revision(repo_id: str, token: str) -> str:
    """Return HEAD commit sha for repo."""
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    return getattr(info, "sha", getattr(info, "oid", ""))


async def _deploy_to_chutes(repo_id: str, chute_user: str, api_key: str, *, revision: str | None = None):
    logger.info("🚀 Deploying %s to Chutes", repo_id)
    config = _generate_chute_config(repo_id, chute_user, revision)
    tmp_file = Path("tmp_chute.py")
    tmp_file.write_text(config)
    cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--public"]
    env = {**os.environ, "CHUTES_API_KEY": api_key}
    proc = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    stdout, _ = await proc.communicate()
    logger.debug(stdout.decode())
    if proc.returncode != 0:
        raise RuntimeError(f"Chutes deploy failed ({proc.returncode})")
    tmp_file.unlink(missing_ok=True)


def _generate_chute_config(repo_id: str, chute_user: str, revision: str | None = None) -> str:
    """Generate Python code for a Chutes deployment.

    If *revision* is provided it is injected in `--revision <sha>`; otherwise the
    flag is omitted so the latest HEAD of the model is used.
    """
    rev_flag = f"--revision {revision} " if revision else ""
    return f'''import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_id}",
    model_name="{repo_id}",
    image="chutes/sglang:0.4.6.post5b",
    concurrency=20,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=40,

    ),
    engine_args=(
        "--trust-remote-code "
        "{rev_flag}"
        "--tool-call-parser deepseekv3 "
    ),
)
'''


async def _warmup(repo_id: str, api_key: str):
    logger.info("🔥 Warming up model %s", repo_id)
    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": repo_id,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload, timeout=10) as r:
                await r.text()
    except Exception as e:
        logger.debug("Expected warm-up timeout ignored: %s", e)
    logger.info("Warm-up finished")


async def _commit_on_chain(repo_id: str, revision: str, blocks: int, cold: str, hot: str):
    logger.info("⛓️  Committing repo %s (rev %s) to chain", repo_id, revision)
    sub = bt.subtensor()
    wallet = bt.wallet(name=cold, hotkey=hot)
    payload = json.dumps({"model": repo_id, "revision": revision})
    sub.set_reveal_commitment(wallet=wallet, netuid=120, data=payload, blocks_until_reveal=blocks)
    logger.info("Commit extrinsic sent (reveal in %d blocks)", blocks) 

# ── Visibility helpers ────────────────────────────────────────────────

async def _make_repo_public(repo_id: str, token: str):
    """Switch HF repo visibility to public."""
    api = HfApi(token=token)
    logger.info("🔓 Making HF repo %s public", repo_id)
    try:
        await asyncio.to_thread(lambda: api.update_repo_visibility(repo_id=repo_id, private=False))
    except Exception as e:
        raise RuntimeError(f"Cannot set repo public: {e}")


async def _wait_until_block(target: int, sub):
    """Polls chain every ~6 s until current block ≥ target."""
    logger.info("⏳ Waiting for block %d", target)
    while True:
        cur = sub.get_current_block()
        if cur >= target:
            break
        await asyncio.sleep(6)
    logger.info("Reached block %d", target) 