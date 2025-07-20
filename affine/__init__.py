#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import os
import sys
import json
import time
import click
import random
import aiohttp
import asyncio
import logging
import textwrap
import botocore
import traceback
import bittensor as bt
import botocore.config
from pathlib import Path
from rich.table import Table
from dotenv import load_dotenv
from rich.console import Console
from huggingface_hub import HfApi
from collections import defaultdict
from abc import ABC, abstractmethod
from alive_progress import alive_bar
from pydantic import BaseModel, Field
from aiobotocore.session import get_session
from huggingface_hub import snapshot_download
from bittensor.core.errors import MetadataError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence

NETUID = 120

# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
TRACE = 5
logging.addLevelName(TRACE, "TRACE")
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    if verbosity >= 3: level = TRACE
    elif verbosity == 2: level = logging.DEBUG
    elif verbosity == 1: level = logging.INFO
    else: level = logging.CRITICAL + 1
    for noisy_logger in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.debug("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor()
        await SUBTENSOR.initialize()
        logger.debug("Connected")
    return SUBTENSOR

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)
    
# --------------------------------------------------------------------------- #
#                               Config                                        #
# --------------------------------------------------------------------------- #
LOCAL_BASE = Path.home() / ".affine"         # ~/.affine
LOCAL_BASE.mkdir(parents=True, exist_ok=True)
load_dotenv(os.path.expanduser("~/.affine/config.env"), override=True)
load_dotenv(override=True)
def get_conf(key, default = None) -> Any:
    value = os.getenv(key)
    if not value and default == None:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True)
        sys.exit(1)
    elif not value:
        return default
    return value

@cli.command('set')
@click.argument('key')
@click.argument('value')
def set_config(key: str, value: str):
    """Set a key-value pair in ~/.affine/config.env."""
    path = os.path.expanduser("~/.affine/config.env")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [l for l in open(path).readlines() if not l.startswith(f"{key}=")] if os.path.exists(path) else []
    lines.append(f"{key}={value}\n")
    open(path, "w").writelines(lines)
    logger.info("Set %s in %s", key, path)
    click.echo(f"Set {key} in {path}")
    
# --------------------------------------------------------------------------- #
#                               Model                                         #
# --------------------------------------------------------------------------- #
def _truncate(text: Optional[str], max_length: int = 80) -> str:
    if not text:
        return ""
    return textwrap.shorten(text, width=max_length, placeholder="...")

class Miner(BaseModel):
    uid: int
    hotkey: str
    model: Optional[str] = None
    revision: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]
    success: bool
    def __repr__(self) -> str:
        resp = _truncate(self.response)
        err  = _truncate(self.error)
        return (
            f"<Response model={self.model!r} success={self.success} "
            f"latency={self.latency_seconds:.3f}s attempts={self.attempts} "
            f"response={resp!r} error={err!r}>"
        )
    __str__ = __repr__

class BaseEnv(BaseModel, ABC):
    @property
    def name(self) -> str:return self.__class__.__name__
    class Config: 
        arbitrary_types_allowed = True
    def __hash__(self): return hash(self.name)
    def __repr__(self): return self.name
    async def many(self, n: int) -> List["Challenge"]:
        return await asyncio.gather(*(self.generate() for _ in range(n)))
    @abstractmethod
    async def generate(self) -> "Challenge": ...
    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: Response) -> "Evaluation": ...

class Challenge(BaseModel):
    env: BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    def json(self, **kwargs): # Hack to get name of env.
        dd = self.dict(**kwargs); dd['env'] = self.env.name;
        return json.dumps(dd)
    async def evaluate(self, response: Response) -> "Evaluation":
        return await self.env.evaluate(self, response)
    def __repr__(self) -> str:
        pr = _truncate(self.prompt)
        return f"<Challenge env={self.env.name!r} prompt={pr!r}>"
    __str__ = __repr__

class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)
    def json(self, **kwargs): # Hack to get name of env.
        dd = self.dict(**kwargs); dd['env'] = self.env.name;
        return json.dumps(dd)
    def __repr__(self) -> str:
        ex = {k: _truncate(str(v)) for k, v in self.extra.items()}
        return f"<Evaluation env={self.env.name!r} score={self.score:.4f} extra={ex!r}>"
    __str__ = __repr__

class Result(BaseModel):
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation
    
    

# --------------------------------------------------------------------------- #
#                               API                                           #
# --------------------------------------------------------------------------- #
async def get_chute(model: str) -> Dict[str, Any]:
    url = f"https://api.chutes.ai/chutes/{model}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            text = await r.text(errors="ignore")
            if r.status != 200:
                return None
            info = await r.json()
            for k in ('readme','cords','tagline','instances'):
                info.pop(k, None)
            info.get('image', {}).pop('readme', None)
            return info
        
async def get_chute_code(identifier: str) -> Optional[str]:
    url = f"https://api.chutes.ai/chutes/code/{identifier}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            return await r.text(errors="ignore")

def _extract_revision(code: str) -> Optional[str]:
    """Extract `--revision <sha>` from chute code string."""
    import re
    m = re.search(r"--revision\s+([0-9a-f]{40})", code)
    return m.group(1) if m else None



# ── run challenges ────────────────────────────────────────────────────────────
HTTP_SEM = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
TERMINAL = {400, 404, 410}

async def query(prompt:str, model:str = "unsloth/gemma-3-12b-it", timeout=120, retries=0, backoff=1) -> Response:
    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}",
        "Content-Type": "application/json"
    }
    start = time.monotonic()
    logger.trace("Starting %r on model=%s", prompt[:30], model)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        for attempt in range(1, retries + 2):
            try:
                async with HTTP_SEM, sess.post(
                    url,
                    json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                    headers=headers,
                    timeout=timeout
                ) as resp:
                    text = await resp.text(errors="ignore")
                    if resp.status in TERMINAL:
                        return Response(
                            response=None,
                            latency_seconds=time.monotonic() - start,
                            attempts=attempt,
                            model=model,
                            error=f"{resp.status}:{text}",
                            success=False
                        )
                    if resp.status != 200:
                        raise RuntimeError(f"{resp.status}:{text}")
                    body = await resp.json()
                    content = body["choices"][0]["message"]["content"]
                    return Response(
                        response=content,
                        latency_seconds=time.monotonic() - start,
                        attempts=attempt,
                        model=model,
                        error=None,
                        success=True
                    )

            except Exception as e:
                if attempt > retries:
                    return Response(
                        response=None,
                        latency_seconds=time.monotonic() - start,
                        attempts=attempt,
                        model=model,
                        error=str(e),
                        success=False
                    )
                await asyncio.sleep(backoff * 2**(attempt-1) * (1 + random.uniform(-0.1, 0.1)))


async def run(challenges, miners, timeout=120, retries=0, backoff=1, progress=True) -> List[Result]:
    if not isinstance(challenges, list): challenges = [challenges]
    if isinstance(miners, Miner): miners = [miners]
    if isinstance(miners, dict):  mmap = miners
    elif isinstance(miners, list) and all(hasattr(m, "uid") for m in miners):  mmap = {m.uid: m for m in miners}
    else: mmap = await miners(miners)
    logger.trace("Running challenges: %s on miners: %s", [chal.prompt[:30] for chal in challenges], list(mmap.keys()))
    results = []
    async def proc(miner, chal):
        resp = await query(chal.prompt, miner.model, timeout, retries, backoff)
        try:
            ev = await chal.evaluate(resp)
            logger.trace("Evaluation done for %s: %r", miner.model, ev)
        except Exception as e:
            logger.warning("Evaluation failed for miner %s on %s: %s", miner.uid, chal.env.name, e)
            ev = Evaluation(env=chal.env, score=0.0, extra={"error": str(e), "evaluation_failed": True})
        return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)

    tasks = [
        asyncio.create_task(proc(m, chal))
        for m in mmap.values() if m.model for chal in challenges
    ]
    if progress:
        with alive_bar(len(tasks), title="Running challenges") as bar:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
                bar()
    else:
        for task in asyncio.as_completed(tasks):
            results.append(await task)

    logger.trace("Finished all runs (%d results)", len(results))
    return results



# --------------------------------------------------------------------------- #
#                              Miner                                          #
# --------------------------------------------------------------------------- #
async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    no_null: bool = False,
    netuid: int = NETUID,
    min_block: int = 0,
    metagraph: object = None,
) -> Dict[int, Miner]:
    # Preamble
    subtensor = await get_subtensor()
    metagraph = metagraph or await subtensor.metagraph(netuid)
    hotkeys = metagraph.hotkeys
    revs = await subtensor.get_all_revealed_commitments(netuid)
    if uids is None:uids = list(range(len(hotkeys)))
    elif isinstance(uids, int): uids = [uids]

    # Single async fetch,
    async def fetch(uid: int) -> Optional[Miner]:
        # Check exists.
        if not (0 <= uid < len(hotkeys)): return None

        hk = hotkeys[uid]
        commits = revs.get(hk) or []
        if not commits: return None

        # Check revision.
        block, data = commits[-1]
        rev = None
        if isinstance(data, str) and data.strip().startswith("{"):
            try:
                parsed = json.loads(data)
                data, rev = parsed.get("model"), parsed.get("revision")
            except json.JSONDecodeError:
                pass

        # Filer non block.
        model = data
        if no_null and (block is None or model is None or block < min_block): return None

        # fetch chute info
        chute = await get_chute(str(model))
        if chute:
            rev = chute.get("revision") or rev
            if rev is None and (cid := chute.get("id") or chute.get("uuid")):
                code = await get_chute_code(cid)
                rev = _extract_revision(code or "")
        else:
            code = await get_chute_code(str(model))
            rev = _extract_revision(code or "")

        if no_null and chute is None: return None
        miner = Miner(
            uid=uid,
            hotkey=hk,
            model=str(model),
            revision=rev,
            block=int(block),
            chute=chute,
        )
        return miner
    results = await asyncio.gather(*(fetch(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}
    return output


# --------------------------------------------------------------------------- #
#                              S3 Operations                                  #
# --------------------------------------------------------------------------- #

def _local_path(key: str) -> Path:
    """
    Map every remote key such as
        'affine/snapshots/<addr>/<blk>.json'
    to a local file under ~/.affine/…
    """
    p = LOCAL_BASE / key
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def get_client_ctx():
    ACCOUNT = get_conf("R2_ACCOUNT_ID")
    KEY_ID  = get_conf("R2_WRITE_ACCESS_KEY_ID")
    SECRET  = get_conf("R2_WRITE_SECRET_ACCESS_KEY")
    endpoint = f"https://{ACCOUNT}.r2.cloudflarestorage.com"
    cfg      = botocore.config.Config(max_pool_connections=256)
    session  = get_session()
    return session.create_client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
        config=cfg,
    )


async def _json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, default=lambda o: o.json() if hasattr(o, "json") else o
    ).encode("utf-8")
    

async def get(key: str) -> Any:
    async with get_client_ctx() as client:
        bucket = get_conf("R2_BUCKET_ID")
        resp   = await client.get_object(Bucket=bucket, Key=key)
        body   = await resp["Body"].read()
        ctype  = resp.get("ContentType", "")
        if ctype == "application/json":
            return json.loads(body.decode("utf-8"))
        return body

async def sink(key: str, obj: Any, *, content_type: str = "application/json"):
    """
    Write *once* to R2 (best‑effort) and *always* to ~/.affine/…
    Failure in either path is logged but never aborts the caller.
    """
    # Prepare body only once
    if content_type == "application/json":
        body = await _json_bytes(obj)
    else:
        body = obj if isinstance(obj, (bytes, bytearray)) else bytes(obj)

    # ---- Remote write (best‑effort) ---------------------------------------
    try:
        async with get_client_ctx() as client:
            bucket = get_conf("R2_BUCKET_ID")
            await client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
    except Exception:
        logger.error("Unexpected error writing to R2", exc_info=True)

    # ---- Local write (must succeed unless fatal) --------------------------
    try:
        lp = _local_path(key)
        await asyncio.to_thread(lp.write_bytes, body)
    except Exception:
        logger.error("Unexpected error writing to ~/.affine", exc_info=True)

# --------------------------------------------------------------------------- #
#                           Snapshot load on startup                          #
# --------------------------------------------------------------------------- #
def _load_latest_snapshot(wallet_ss58: str):
    """
    Returns tuple:
        blocks, revisions, models, scores, last_set
    If no snapshot is found everything is initialised empty.
    """
    snap_dir = LOCAL_BASE / "affine" / "snapshots" / wallet_ss58
    if not snap_dir.exists():
        return {}, {}, {}, {}, -1

    latest = max(snap_dir.glob("*.json"), default=None)
    if latest is None:
        return {}, {}, {}, {}, -1

    with latest.open("r") as fh:
        data = json.load(fh)

    blocks     = data.get("blocks", {})
    revisions  = data.get("revisions", {})
    models     = data.get("models", {})
    scores_raw = data.get("scores", {})
    scores     = {
        hk: defaultdict(float, sc) for hk, sc in scores_raw.items()
    }
    last_set   = data.get("blk", -1)
    return blocks, revisions, models, scores, last_set

# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #
from .envs.sat import SAT
from .envs.abd import ABDUCTION
from .envs.ded import DEDUCTION

# Registry of active environments
ENVS = {"SAT": SAT, "ABDUCTION": ABDUCTION, "DEDUCTION": DEDUCTION}
@cli.command("validate")
@click.option("--coldkey", default=None, help="Cold wallet name")
@click.option("--hotkey", default=None, help="Hot wallet key")
def validate(coldkey: str, hotkey: str):
    """Starts an affine validator with local‑state persistence."""
    print( get_conf("CHUTES_API_KEY") )
    console = Console()
    coldkey = coldkey or get_conf("BT_WALLET_COLD", "default")
    hotkey  = hotkey  or get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)

    async def main():
        alpha = 0.90
        subtensor = await get_subtensor()
        meta      = await subtensor.metagraph(NETUID)

        # ---- Load previous state (if any) ---------------------------------
        blocks, revisions, models, scores, last_set = _load_latest_snapshot(
            wallet.hotkey.ss58_address
        )

        # ---- Pull initial list of miners ----------------------------------
        miners_map = await miners(no_null=True, metagraph=meta)

        # Ensure every current miner has entries, even if snapshot empty
        for m in miners_map.values():
            blocks.setdefault(m.hotkey, m.block)
            revisions.setdefault(m.hotkey, m.revision)
            models.setdefault(m.hotkey, m.model)
            scores.setdefault(m.hotkey, defaultdict(float))

        while True:
            try:
                # ---- Refresh chain state & miner list ---------------------
                blk        = await subtensor.get_current_block()
                meta       = await subtensor.metagraph(NETUID)
                miners_map = await miners(no_null=True, metagraph=meta)

                # ---------------- Reset logic -----------------------------
                for m in miners_map.values():
                    if (
                        blocks.get(m.hotkey)   != m.block
                        or revisions.get(m.hotkey) != m.revision
                        or models.get(m.hotkey)    != m.model
                    ):
                        # Model, revision, *or* block changed – reset stats
                        blocks[m.hotkey]    = m.block
                        revisions[m.hotkey] = m.revision
                        models[m.hotkey]    = m.model
                        scores[m.hotkey]    = defaultdict(float)

                hotkeys = [m.hotkey for m in miners_map.values()]
                miners_map = {m.hotkey: m for m in miners_map.values()}

                if len(miners_map) == 0:
                    logger.debug("No valid miners, waiting …")
                    await asyncio.sleep(10)
                    continue

                # ---------------- Evaluate miners -------------------------
                try:
                    challenges = [await env().generate() for env in ENVS.values()]
                    results    = await run(
                        challenges=challenges, miners=miners_map, timeout=90
                    )
                    for r in results:
                        e, hk, raw = r.challenge.env.name, r.miner.hotkey, r.evaluation.score
                        if r.response.success:
                            scores[hk][e] = raw * (1 - alpha) + scores[hk][e] * alpha
                        logger.info(
                            f"Environment: {e}, "
                            f"Miner: {hk[:10]}, "
                            f"Model: {r.miner.model}, "
                            f"Score: {raw:.4f}, "
                            f"Current: {scores[hk][e]:.4f}, "
                            f"Success: {r.response.success}"
                        )
                except Exception as e:
                    traceback.print_exc()
                    logger.info(f"[yellow]Transient error during eval:[/yellow] {e}. Continuing validator loop…")
                    continue

                # ---- Persist raw results ---------------------------------
                await sink(
                    f"affine/results/{wallet.hotkey.ss58_address}/{blk}.json",
                    [r.json() for r in results],
                )

                # ---------------- Rank & dominance ------------------------
                ranks, counts = {}, defaultdict(int)
                for e in ENVS:
                    uniq = sorted({scores[h][e] for h in hotkeys}, reverse=True)
                    rank_of = {score: i + 1 for i, score in enumerate(uniq)}
                    ranks[e] = {h: rank_of[scores[h][e]] for h in hotkeys}

                for a in hotkeys:
                    for b in hotkeys:
                        if a == b:
                            continue
                        better = sum(ranks[e][a] < ranks[e][b] for e in ENVS)
                        not_worse = all(ranks[e][a] <= ranks[e][b] for e in ENVS)
                        if not_worse and better >= 1:
                            counts[a] += 1

                best_key, best = (-1, None), None
                for h in hotkeys:
                    key = (counts.get(h, 0), -blocks.get(h, 0))
                    if key > best_key:
                        best_key, best = key, h

                weights = [1.0 if hk == best else 0.0 for hk in meta.hotkeys]

                # ---- Persist snapshot (remote + local) -------------------
                await sink(
                    f"affine/snapshots/{wallet.hotkey.ss58_address}/{blk:08d}.json",
                    {
                        "blk":        blk,
                        "scores":     {hk: dict(scores[hk]) for hk in hotkeys},
                        "blocks":     blocks,
                        "revisions":  revisions,
                        "models":     models,
                        "weights":    weights,
                        "miners": {
                            hk: {"uid": miner.uid, "model": miner.model, "revision": miner.revision}
                            for hk, miner in miners_map.items()
                        },
                    },
                )

                # ---- Render live table -----------------------------------
                table = Table(title=f"Block {blk}")
                table.add_column("UID", style="bold")
                table.add_column("Block", justify="right")
                table.add_column("Model", justify="right")
                table.add_column("Rev", justify="right")
                for e in ENVS:
                    table.add_column(f"{e} Score", justify="right")
                    table.add_column(f"{e} Rank", justify="right")
                table.add_column("Weight", justify="right")

                for hk in hotkeys:
                    miner = miners_map.get(hk)
                    if not miner:
                        continue
                    row = [
                        str(miner.uid),
                        str(miner.block),
                        miner.model,
                        str(miner.revision),
                    ]
                    for e in ENVS:
                        row.extend([f"{scores[hk][e]:.4f}", str(ranks[e][hk])])
                    row.append(f"{weights[meta.hotkeys.index(hk)]:.2f}")
                    table.add_row(*row)
                console.print(table)

                # ---- Submit weights periodically -------------------------
                if blk - last_set > 100:
                    await subtensor.set_weights(
                        wallet=wallet,
                        netuid=NETUID,
                        uids=meta.uids,
                        weights=weights,
                        wait_for_inclusion=False,
                    )
                    last_set = blk

            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                traceback.print_exc()
                console.log(f"[yellow]Transient error:[/yellow] {e}. Continuing validator loop…")
                continue

    asyncio.run(main())
    
    

# --------------------------------------------------------------------------- #
#                              Pull Model                                     #
# --------------------------------------------------------------------------- #
@cli.command("pull")
@click.argument("uid", type=int)
@click.option("--model_path", "-p", default = './model_path', required=True, type=click.Path(), help="Local directory to save the model")
@click.option('--hf-token', default=None, help="Hugging Face API token (env HF_TOKEN if unset)")
def pull(uid: int, model_path: str, hf_token: str):
    """Pulls a model from a specific miner UID if exists."""

    # 1. Ensure HF token
    hf_token     = hf_token or get_conf("HF_TOKEN")

    # 2. Lookup miner on‑chain
    miner_map = asyncio.run(miners(uids=uid, no_null=True))
    miner = miner_map.get(uid)
    
    if miner is None:
        click.echo(f"No miner found for UID {uid}", err=True)
        sys.exit(1)
    repo_name = miner.model
    logger.info("Pulling model %s for UID %d into %s", repo_name, uid, model_path)

    # 3. Download snapshot
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=miner.revision,
        )
        click.echo(f"Model {repo_name} pulled to {model_path}")
    except Exception as e:
        logger.error("Failed to download %s: %s", repo_name, e)
        click.echo(f"Error pulling model: {e}", err=True)
        sys.exit(1)


# --------------------------------------------------------------------------- #
#                              Push Model                                     #
# --------------------------------------------------------------------------- #
@cli.command("push")
@click.option('--model_path',  default='./model_path', help='Local path to model artifacts.')
@click.option('--existing-repo', default=None, help='Use an existing HF repo instead of uploading (format <user>/<repo>)')
@click.option('--coldkey',     default=None, help='Name of the cold wallet to use.')
@click.option('--hotkey',      default=None, help='Name of the hot wallet to use.')
def push(model_path: str, existing_repo: str, coldkey: str, hotkey: str):
    """Pushes a model to be hosted by your miner."""
    # -----------------------------------------------------------------------------
    # 1. Wallet & config
    # -----------------------------------------------------------------------------
    coldkey = coldkey or get_conf("BT_WALLET_COLD", "default")
    hotkey  = hotkey  or get_conf("BT_WALLET_HOT", "default")
    logger.debug("Using coldkey=%s, hotkey=%s", coldkey, hotkey)
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    # Required API credentials
    hf_user        = get_conf("HF_USER")
    hf_token       = get_conf("HF_TOKEN")
    chutes_api_key = get_conf("CHUTES_API_KEY")
    chute_user     = get_conf("CHUTE_USER")
    # TODO: validate API creds, exit gracefully if missing

    # -----------------------------------------------------------------------------
    # 2. Prepare HF repo name
    #    - If --existing-repo provided, use it directly and skip local upload
    # -----------------------------------------------------------------------------
    repo_name = existing_repo or f"{hf_user}/Affine-{wallet.hotkey.ss58_address}"
    logger.debug("Using existing HF repo: %s" if existing_repo else "Hugging Face repo: %s", repo_name)

    # -----------------------------------------------------------------------------
    # 3. Create & secure HF repo
    # -----------------------------------------------------------------------------
    api = HfApi(token=hf_token)
    if not existing_repo:
        api.create_repo(repo_id=repo_name, repo_type="model", private=True, exist_ok=True)
        try: api.update_repo_visibility(repo_id=repo_name, private=True)
        except Exception: logger.debug("Repo already private or visibility update failed")

    # -----------------------------------------------------------------------------
    # 4. Upload model files to HF (skip if using existing repo)
    # -----------------------------------------------------------------------------
    async def deploy_model_to_hf():
        logger.debug("Starting model upload from %s", model_path)
        # Gather files
        files = []
        for root, _, fnames in os.walk(model_path):
            if ".cache" in root or any(p.startswith(".") for p in root.split(os.sep)):
                continue
            for fname in fnames:
                if not (fname.startswith(".") or fname.endswith(".lock")):
                    files.append(os.path.join(root, fname))

        # Upload files with limited concurrency to avoid HF 429 errors
        SEM = asyncio.Semaphore(int(os.getenv("AFFINE_UPLOAD_CONCURRENCY", "2")))

        async def _upload(path: str):
            rel = os.path.relpath(path, model_path)
            async with SEM:  # limit concurrent commits
                await asyncio.to_thread(
                    lambda: api.upload_file(
                        path_or_fileobj=path,
                        path_in_repo=rel,
                        repo_id=repo_name,
                        repo_type="model"
                    )
                )
                logger.debug("Uploaded %s", rel)

        await asyncio.gather(*(_upload(p) for p in files))
        logger.debug("Model upload complete (%d files)", len(files))

    asyncio.run(deploy_model_to_hf()) if not existing_repo else logger.debug("Skipping model upload because --existing-repo was provided")

    # -----------------------------------------------------------------------------
    # 5. Fetch latest revision hash
    # -----------------------------------------------------------------------------
    info     = api.repo_info(repo_id=repo_name, repo_type="model")
    revision = getattr(info, "sha", getattr(info, "oid", "")) or ""
    logger.debug("Latest revision: %s", revision)

    # -----------------------------------------------------------------------------
    # 6. Commit model revision on-chain
    # -----------------------------------------------------------------------------
    async def commit_to_chain():
        """Submit the model commitment, retrying if the subnet space quota is full."""
        logger.debug("Preparing on-chain commitment")
        sub     = await get_subtensor()
        payload = json.dumps({"model": repo_name, "revision": revision})
        while True:
            try:
                await sub.set_reveal_commitment(wallet=wallet, netuid=NETUID, data=payload, blocks_until_reveal=1)
                logger.debug("On-chain commitment submitted")
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    logger.debug("SpaceLimitExceeded – waiting one block before retrying")
                    await sub.wait_for_block()
                else:
                    raise

    asyncio.run(commit_to_chain())

    # -----------------------------------------------------------------------------
    # 7. Make HF repo public
    # -----------------------------------------------------------------------------
    try:
        api.update_repo_visibility(repo_id=repo_name, private=False)
        logger.debug("Repo made public")
    except Exception:
        logger.trace("Failed to make repo public (already public?)")

    # -----------------------------------------------------------------------------
    # 8. Deploy Chute
    # -----------------------------------------------------------------------------
    async def deploy_to_chutes():
        logger.debug("Building Chute config")
        rev_flag = f"--revision {revision} " if revision else ""
        chutes_config = textwrap.dedent(f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_name}",
    model_name="{repo_name}",
    image="chutes/sglang:0.4.6.post5b",
    concurrency=20,
    node_selector=NodeSelector(
        gpu_count=4,
        min_vram_gb_per_gpu=24,
        include=["4090", "l40s", "a6000_ada"],
        exclude=["h200", "b200", "mi300x"],
    ),
    engine_args=(
        "--trust-remote-code "
        "{rev_flag}"
        "--tool-call-parser deepseekv3 "
    ),
)
""")
        tmp_file = Path("tmp_chute.py")
        tmp_file.write_text(chutes_config)
        logger.debug("Wrote Chute config to %s", tmp_file)
        logger.debug("=== chute file ===\n%s", tmp_file.read_text())

        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--public"]
        env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        # Auto-answer the interactive Y/N prompt
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        stdout, _ = await proc.communicate()
        logger.trace(stdout.decode())
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    # -----------------------------------------------------------------------------
    # 9. Warm up model until it’s marked hot
    # -----------------------------------------------------------------------------
    async def warmup_model():
        logger.debug("Warming up model with SAT challenges")
        sub       = await get_subtensor()
        meta      = await sub.metagraph(NETUID)
        my_uid    = meta.hotkeys.index(wallet.hotkey.ss58_address)
        miner  = (await miners(uids=my_uid))[my_uid]

        while not (miner.chute or {}).get("hot", False):
            challenge = await SAT().generate()
            await run(challenges=challenge, miners=[miner])
            await sub.wait_for_block()
            miner = (await miners(uids=my_uid))[my_uid]
            logger.trace("Checked hot status: %s", (miner.chute or {}).get("hot"))

        logger.debug("Model is now hot and ready")

    asyncio.run(warmup_model())
    logger.debug("Mining setup complete. Model is live!")  


        

