#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import random
import click
import aiohttp
import asyncio
import bittensor as bt
from dotenv import load_dotenv
from collections import defaultdict
from abc import ABC, abstractmethod
from alive_progress import alive_bar
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Tuple
from .utils import validate_miners_hot, load_samples, save_samples
import types  # for SimpleNamespace used in warm-up helper
from .invalidation import invalidate  # local import to avoid cycles
import numpy as np  # lightweight dependency, safe to import anywhere
from typing import Sequence

# ── load env ─────────────────────────────────────────────────────────────────
load_dotenv(os.path.expanduser("~/.affine/config.env"), override=True)
load_dotenv(override=True)

def get_conf(key) -> Any:
    value = os.getenv(key)
    if not value:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True)
        sys.exit(1)
    return value

# ── logging setup ─────────────────────────────────────────────────────────────
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
    for noisy_logger in [ "websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ── core models ───────────────────────────────────────────────────────────────
class Miner(BaseModel):
    uid: int
    hotkey: str
    model: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]

class BaseEnv(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    async def many(self, n: int) -> List["Challenge"]:
        """Return *n* Challenge objects, using on-disk samples for caching.

        The algorithm is:
            1. Load all cached samples for this environment.
            2. Pop up to *n* of them to create Challenge objects.
            3. If we still need more, call *generate()* synchronously to fill
               the gap.
            4. Persist the *unused* remainder back to disk so the cache never
               shrinks unexpectedly.
        """
        env_name = self.__class__.__name__

        # 1 & 2 — fetch cached samples --------------------------------------
        cached = load_samples(env_name)
        used, remaining = cached[:n], cached[n:]

        challenges: List["Challenge"] = [
            Challenge(env=self, prompt=sd["prompt"], extra={k: v for k, v in sd.items() if k != "prompt"})
            for sd in used
        ]

        # 3 — top-up with freshly generated ones ---------------------------
        needed = n - len(challenges)
        if needed > 0:
            for _ in range(needed):
                challenges.append(await self.generate())

        # 4 — persist leftover cache --------------------------------------
        save_samples(env_name, remaining)

        return challenges

    @abstractmethod
    async def generate(self) -> "Challenge":
        ...

    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: Response) -> "Evaluation":
        ...

class Challenge(BaseModel):
    env: BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)

    async def evaluate(self, response: Response) -> "Evaluation":
        return await self.env.evaluate(self, response)

class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)

class Result(BaseModel):
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation

# ── fetch chute & miners ─────────────────────────────────────────────────────
async def get_chute(model: str) -> Dict[str, Any]:
    url = f"https://api.chutes.ai/chutes/{model}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            text = await r.text(errors="ignore")
            if r.status != 200:
                raise RuntimeError(f"{r.status}:{text}")
            info = await r.json()
            for k in ('readme','cords','tagline','instances'):
                info.pop(k, None)
            info.get('image', {}).pop('readme', None)
            logger.trace("Fetched chute info for %s", model)
            return info

async def miners(uids: Optional[Union[int, List[int]]] = None, no_null: bool = False) -> Dict[int, Miner]:
    NETUID = 120
    s = bt.async_subtensor()
    await s.initialize()
    meta = await s.metagraph(NETUID)
    revs = await s.get_all_revealed_commitments(NETUID)

    if uids is None:
        uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids = [uids]

    out: Dict[int, Miner] = {}
    for uid in uids:
        hk = meta.hotkeys[uid] if 0 <= uid < len(meta.hotkeys) else ""
        commits = revs.get(hk) or []
        blk, mdl = commits[-1] if commits else (None, None)
        if no_null and blk is None:
            continue
        out[uid] = Miner(uid=uid, hotkey=hk,
                         model=str(mdl) if mdl is not None else None,
                         block=int(blk) if blk is not None else None)

    with_models = [m for m in out.values() if m.model]
    if with_models:
        infos = await asyncio.gather(
            *[get_chute(m.model) for m in with_models],
            return_exceptions=True
        )
        for m, info in zip(with_models, infos):
            if not isinstance(info, Exception):
                m.chute = info

    miner_info = [
        f"\tUID: {m.uid}, Hotkey: {m.hotkey}, Model: {m.model}, Block: {m.block}"
        for m in out.values()
    ]
    logger.debug("Discovered %d miners:\n%s", len(out), "\n".join(miner_info))
    return out

# ── run challenges ────────────────────────────────────────────────────────────
HTTP_SEM = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))  # limit concurrent HTTP calls

# ── persistent last results store ─────────────────────────────────────────
RESULTS_FILE = os.path.expanduser("~/.affine/results.json")

def save_last_results(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def _reset_in_memory() -> None:  # pragma: no cover
    """Clear in-memory/Redis state for unit testing."""
    if _USE_REDIS:
        _R.flushdb()  # type: ignore[attr-defined]
    else:
        _LOCAL.clear()

async def _run_one(
    session: aiohttp.ClientSession,
    chal: Challenge,
    model: str,
    timeout: float,
    retries: int,
    backoff: float
) -> Response:
    url = "https://llm.chutes.ai/v1/chat/completions"
    token = get_conf("CHUTES_API_KEY")
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    start = time.monotonic()
    TERMINAL_STATUS = {400, 404, 410}
    # NOTE: When *retries* is provided >0 we still break early on terminal errors to avoid useless traffic.
    for attempt in range(1, retries + 2):
        try:
            async with HTTP_SEM:
                async with session.post(
                    url,
                    json={"model": model, "messages": [{"role": "user", "content": chal.prompt}]},
                    headers=hdr,
                    timeout=timeout
                ) as r:
                    text = await r.text(errors="ignore")
                    if r.status in TERMINAL_STATUS:
                        # Permanent failure: do not retry further
                        latency = time.monotonic() - start
                        logger.debug("Permanent failure (%s) for %s: %s", r.status, model, text)
                        return Response(response=None, latency_seconds=latency, attempts=attempt, model=model, error=f"{r.status}:{text}")
                    if r.status != 200:
                        raise RuntimeError(f"{r.status}:{text}")
                    data = await r.json()
                    res = data["choices"][0]["message"]["content"]
                    latency = time.monotonic() - start
                    logger.trace("Model %s answered in %.2fs on attempt %d", model, latency, attempt)
                    return Response(response=res, latency_seconds=latency, attempts=attempt, model=model, error=None)
        except Exception as e:
            logger.debug("Attempt %d for %s failed: %s", attempt, model, e)
            if attempt > retries:
                latency = time.monotonic() - start
                return Response(response=None, latency_seconds=latency, attempts=attempt, model=model, error=str(e))
            delay = backoff * (2 ** (attempt - 1)) * (1 + random.uniform(-0.1, 0.1))
            await asyncio.sleep(delay)
    return Response(response=None, latency_seconds=time.monotonic()-start, attempts=retries+1, model=model, error="unreachable")

async def run(
    challenges: Union[Challenge, List[Challenge]],
    miners: Optional[Union[Dict[int, Miner], List[Miner], int, List[int]]] = None,
    timeout: float = 120.0,
    retries: int = 3,
    backoff: float = 1.0,
    progress: bool = True,
) -> List[Result]:
    if not isinstance(challenges, list):
        challenges = [challenges]
    if isinstance(miners, dict):
        mmap = miners
    elif isinstance(miners, list) and all(isinstance(m, Miner) for m in miners):
        mmap = {m.uid: m for m in miners}
    else:
        mmap = await miners(miners)
    valid = [m for m in mmap.values() if m.model]
    results: List[Result] = []
    total_tests = len(valid) * len(challenges)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:

        if progress:
            with alive_bar(total_tests, title="Running challenges") as bar:

                async def _run_for_miner(miner: Miner) -> List[Result]:
                    """Run *all* challenges for *miner* sequentially."""
                    local_results: List[Result] = []
                    for chal in challenges:
                        resp = await _run_one(sess, chal, miner.model, timeout, retries, backoff)
                        ev = await chal.evaluate(resp)
                        local_results.append(Result(miner=miner, challenge=chal, response=resp, evaluation=ev))
                        bar()
                    return local_results

                miner_tasks = [asyncio.create_task(_run_for_miner(m)) for m in valid]
                for coro in asyncio.as_completed(miner_tasks):
                    results.extend(await coro)
        else:
            async def _run_for_miner(miner: Miner) -> List[Result]:
                local_results: List[Result] = []
                for chal in challenges:
                    resp = await _run_one(sess, chal, miner.model, timeout, retries, backoff)
                    ev = await chal.evaluate(resp)
                    local_results.append(Result(miner=miner, challenge=chal, response=resp, evaluation=ev))
                return local_results

            miner_tasks = [asyncio.create_task(_run_for_miner(m)) for m in valid]
            for coro in asyncio.as_completed(miner_tasks):
                results.extend(await coro)

    logger.info("Finished %d runs", len(results))
    return results

# ── CLI & commands ────────────────────────────────────────────────────────────
from .envs.coin import COIN
from .envs.sat import SAT
from .envs.abd import ABD
from .envs.research import RES
from .envs.gaia import GAIA

ENVS = {"COIN": COIN, "SAT": SAT, "ABD": ABD, "RES": RES, "GAIA": GAIA}

# ── persistent Elo helpers ──────────────────────────────────────────
ELO_FILE = os.path.expanduser("~/.affine/results/elo.json")

def load_elo() -> Dict[str, float]:
    """Load Elo table from disk, creating it if absent."""
    if not os.path.exists(ELO_FILE):
        # Ensure directory exists and create an empty file
        os.makedirs(os.path.dirname(ELO_FILE), exist_ok=True)
        with open(ELO_FILE, "w") as f:
            json.dump({}, f)
        return defaultdict(lambda: 1500.0)

    try:
        with open(ELO_FILE) as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        # Corrupted file: start fresh
        return defaultdict(lambda: 1500.0)

def save_elo(table: Dict[str, float]) -> None:
    """Persist Elo dict to disk."""
    os.makedirs(os.path.dirname(ELO_FILE), exist_ok=True)
    # convert defaultdict to regular dict for json
    serial = dict(table)
    with open(ELO_FILE, "w") as f:
        json.dump(serial, f, indent=2)

@click.group()
@click.option('-v', '--verbose', count=True,
              help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)

@cli.command('run')
@click.argument('uids', callback=lambda _ctx, _param, val: [int(x) for x in val.split(',')])
@click.argument('env', type=click.Choice(list(ENVS), case_sensitive=False))
@click.option('-n', '--num', 'n', default=1, show_default=True, help='Number of challenges')
def run_command(uids, env, n):
    """Run N challenges in ENV against miners."""
    logger.debug("Running %d challenges in %s for UIDs %s", n, env, uids)
    async def _coro():
        chals = await ENVS[env.upper()]().many(n)
        ms = await miners(uids)
        return await run(challenges=chals, miners=ms)
    results = asyncio.run(_coro())
    print_results(results)
    save(results)

@cli.command('deploy')
@click.argument('filename', type=click.Path(exists=True))
@click.option('--chutes-api-key', default=None, help='Chutes API key')
@click.option('--hf-user', default=None, help='HuggingFace user')
@click.option('--hf-token', default=None, help='HuggingFace token')
@click.option('--chute-user', default=None, help='Chutes user')
@click.option('--wallet-cold', default=None, help='Bittensor coldkey')
@click.option('--wallet-hot', default=None, help='Bittensor hotkey')
@click.option('--existing-repo', default=None, help='Use existing HuggingFace repository')
@click.option('--blocks-until-reveal', default=1, help='Blocks until reveal on Bittensor')
def deploy(filename, chutes_api_key, hf_user, hf_token, chute_user, wallet_cold, wallet_hot, existing_repo, blocks_until_reveal):
    """Deploy a model or file using provided credentials."""
    logger.debug("Deploying %s", filename)
    
    # Resolve configuration values
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    hf_user        = hf_user        or get_conf("HF_USER")
    hf_token       = hf_token       or get_conf("HF_TOKEN")
    chute_user     = chute_user     or get_conf("CHUTES_USER")
    wallet_cold    = wallet_cold    or get_conf("BT_COLDKEY")
    wallet_hot     = wallet_hot     or get_conf("BT_HOTKEY")
    
    # Import deployment functions
    from .deployment import deploy_model, DeploymentConfig
    
    # Create deployment configuration
    config = DeploymentConfig(
        chutes_api_key=chutes_api_key,
        hf_user=hf_user,
        hf_token=hf_token,
        chute_user=chute_user,
        wallet_cold=wallet_cold,
        wallet_hot=wallet_hot
    )
    
    # Run deployment
    async def _deploy():
        repo_id = await deploy_model(
            local_path=filename,
            config=config,
            existing_repo=existing_repo,
            blocks_until_reveal=blocks_until_reveal
        )
        click.echo(f"✅ Deployment completed successfully!")
        click.echo(f"Repository: {repo_id}")
        return repo_id
    
    try:
        asyncio.run(_deploy())
    except KeyboardInterrupt:
        click.echo("\n Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Deployment failed: {str(e)}", err=True)
        logger.error("Deployment failed", exc_info=True)
        sys.exit(1)

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

# ── simplified weight-based validator ─────────────────────────────────────
@cli.command("validate")
@click.option("--delay", "-d", default=5.0, show_default=True,
              help="Seconds between validation cycles")
def validate(delay: float):
    """Validate all miners on 5 SAT + 5 ABD, compute GRPO and announce the winner."""

    SAT_N, ABD_N = 5, 5 

    async def _cycle_once() -> None:
        miners_dict = await miners(no_null=True)
        # Initial filter: miners with a published model
        miners_live = [m for m in miners_dict.values() if m.model]

        # ── invalidation pipeline: keep only verified miners ────────────

        """if miners_live:
            entries = await asyncio.gather(*[
                invalidate(uid=m.uid,
                           commit_hash="",  # not used for check here
                           hotkey=m.hotkey,
                           block=m.block or 0,
                           model_name=m.model or "")
                for m in miners_live
            ])
            miners_live = [m for m, e in zip(miners_live, entries) if e.finetune_verified]"""

        if len(miners_live) < 1:
            click.echo("🚫 No verified miners – waiting…")
            return

        sat_chals = await SAT().many(SAT_N)
        abd_chals = await ABD().many(ABD_N)
        common_chals = sat_chals + abd_chals

        random.shuffle(miners_live)      
        random.shuffle(common_chals)     

        total_reqs = len(miners_live) * len(common_chals)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
            async def _score(miner: Miner, chal: Challenge):
                try:
                    resp = await _run_one(sess, chal, miner.model, timeout=120.0, retries=0, backoff=0.0)
                    ev = await chal.evaluate(resp)
                    return miner, chal, ev.score
                except Exception as exc:
                    logger.error("Request failed for %s: %s", miner.hotkey, exc)
                    return miner, chal, 0.0

            
            results: List[tuple] = []
            with alive_bar(total_reqs, title="Validation") as bar:
                for chal in common_chals:
                    tasks = [asyncio.create_task(_score(m, chal)) for m in miners_live]
                    for coro in asyncio.as_completed(tasks):
                        res = await coro
                        results.append(res)
                        bar()

        from collections import defaultdict

        # Map hotkey → Miner for quick access
        hk_to_miner = {m.hotkey: m for m in miners_live}

        # 1) Record raw task outcomes (0/1) per environment ------------
        miner_results: Dict[str, Dict[str, list[int]]] = defaultdict(lambda: {"SAT": [], "ABD": []})

        # 2) Collect successes per challenge to determine the winner ---
        for miner, chal, score in results:
            env_name = chal.env.__class__.__name__
            success = 1 if score >= 1.0 else 0
            miner_results[miner.hotkey][env_name].append(success)

        # 3) Compute simple counts (0-5) per miner and environment -----
        miner_scores: Dict[str, Dict[str, int]] = defaultdict(lambda: {"SAT": 0, "ABD": 0})

        for hk, env_lists in miner_results.items():
            miner_scores[hk]["SAT"] = sum(env_lists["SAT"])
            miner_scores[hk]["ABD"] = sum(env_lists["ABD"])

        # 4) Persist raw outcomes to ~/.affine/results.json ------------
        try:
            round_id = next_round_id()
            rows_to_append = []
            for hk in miner_scores:
                sat_score = miner_scores[hk]["SAT"]
                abd_score = miner_scores[hk]["ABD"]
                #
                # Store detailed per-task success (0/1) lists as well for
                # easier debugging / analytics while keeping the aggregated
                # numeric scores used elsewhere.
                sat_raw  = miner_results[hk]["SAT"]
                abd_raw  = miner_results[hk]["ABD"]
                rows_to_append.append({
                    "round": round_id,
                    "uid": hk_to_miner[hk].uid,
                    "hotkey": hk,
                    # aggregated relative scores (used by monitoring helpers)
                    "SAT": sat_score,
                    "ABD": abd_score,
                    # raw binary successes per prompt (detail requested)
                    "SAT_tasks": sat_raw,
                    "ABD_tasks": abd_raw,
                })
            append(rows_to_append)
        except Exception as _exc:
            logger.error("Failed to append results: %s", _exc, exc_info=True)

        # 5) Update 20-round rolling averages --------------------------
        try:
            from affine import round_scoring as _rs
            table20 = _rs.compute_scores_window()
            _rs.save_scores(table20)
        except Exception as _exc:
            logger.error("Failed to update score.json: %s", _exc, exc_info=True)

        # 6) Determine winner using 20-round window statistics ----------
        try:
            # Build helper dicts from 20-round table
            env_names = ("SAT", "ABD")
            if not 'table20' in locals():
                from affine import round_scoring as _rs
                table20 = _rs.compute_scores_window()
            if table20:
                # Average score per env across miners (in 0-100 scale)
                env_avg: Dict[str, float] = {
                    env: float(sum(row[env] for row in table20)) / len(table20)
                    for env in env_names
                }

                env_max: Dict[str, float] = {
                    env: max(row[env] for row in table20)
                    for env in env_names
                }

                # Compute final score per miner
                ranking: list[Dict[str, Any]] = []
                for row in table20:
                    hk = row["hotkey"]
                    # Only leaders for each environment earn points.
                    # Each earned point equals (1 - env_avg / 100).
                    bonus = sum(
                        (1.0 - env_avg[env] / 100.0)
                        for env in env_names
                        if row[env] == env_max[env]
                    )
                    # Final score is 2 ^ (total points earned).
                    final = 2 ** bonus
                    ranking.append({"hotkey": hk, "uid": row["uid"], "final": final})

                ranking.sort(key=lambda d: d["final"], reverse=True)
                top = ranking[0]
            else:
                # Fallback – use current round min(SAT,ABD)
                top = {
                    "hotkey": max(miner_scores, key=lambda h: min(miner_scores[h]["SAT"], miner_scores[h]["ABD"])) ,
                    "uid": hk_to_miner[max(miner_scores, key=lambda h: min(miner_scores[h]["SAT"], miner_scores[h]["ABD"]))].uid,
                }
        except Exception as _exc:
            logger.error("Winner computation failed: %s", _exc, exc_info=True)
            # Graceful degradation: simple min criterion
            top = {
                "hotkey": max(miner_scores, key=lambda h: min(miner_scores[h]["SAT"], miner_scores[h]["ABD"])) ,
                "uid": hk_to_miner[max(miner_scores, key=lambda h: min(miner_scores[h]["SAT"], miner_scores[h]["ABD"]))].uid,
            }

        # 7) Persist winner history (ND-JSON) ------------------
        WIN_FILE = os.path.expanduser("~/.affine/winners.json")
        os.makedirs(os.path.dirname(WIN_FILE), exist_ok=True)
        with open(WIN_FILE, "a", encoding="utf-8") as wf:
            wf.write(json.dumps({
                "round": round_id,
                "uid": top["uid"],
                "hotkey": top["hotkey"],
                "block": (hk_to_miner.get(top["hotkey"]) or types.SimpleNamespace(block=None)).block,
            }))
            wf.write("\n")

        # 8) Compute champion over last 20 rounds -------------
        try:
            recent: Dict[str, int] = defaultdict(int)
            winners: list = []
            if os.path.exists(WIN_FILE):
                with open(WIN_FILE, "r", encoding="utf-8") as rf:
                    for line in rf:
                        try:
                            winners.append(json.loads(line))
                        except Exception:
                            continue
            if winners:
                max_round = max(w["round"] for w in winners)
                window_min = max_round - 19
                for w in winners:
                    if w["round"] >= window_min:
                        recent[w["hotkey"]] += 1
                champion_hk, wins = max(recent.items(), key=lambda kv: kv[1])
                champ = next((w for w in winners if w["hotkey"] == champion_hk), None)
                if champ:
                    click.echo(f"🥇 Leader 20rds: UID {champ['uid']} – {champion_hk[:12]}… ({wins} wins)")
        except Exception as _exc:
            logger.error("Failed to compute 20-round champion: %s", _exc, exc_info=True)

        
        click.echo(
            f"\n🏆 Winner round {round_id}: UID {top['uid']} – {top['hotkey'][:12]}… (score={top.get('final', 0):.3f})"
        )

        # End of cycle -----------------------------------------------
        return

    async def _main_loop():
        """Persistent async loop – keeps the event-loop alive so that
        long-lived background tasks (e.g., finetune worker) can run."""
        while True:
            try:
                await _cycle_once()
            except KeyboardInterrupt:
                click.echo("\nStopped.")
                break
            except Exception as exc:
                logger.error("validation loop failed: %s", exc, exc_info=True)
            await asyncio.sleep(delay)

    # Run a single event-loop for the whole lifetime of the command
    asyncio.run(_main_loop())

# ── helpers ─────────────────────────────────────────────────────────────────
def print_results(results: List[Result]):
    stats = defaultdict(lambda: {'model': None, 'scores': []})
    for r in results:
        stats[r.miner.uid]['model'] = r.miner.model
        stats[r.miner.uid]['scores'].append(r.evaluation.score)

    fmt = "{:<5} {:<25} {:<5} {:>7} {:>8}"
    header = fmt.format("UID", "Model", "#", "Total", "Average")
    click.echo(header)
    click.echo("-" * len(header))
    for uid, data in stats.items():
        cnt   = len(data['scores'])
        total = sum(data['scores'])
        avg   = total / cnt if cnt else 0
        click.echo(fmt.format(uid, data['model'], cnt, f"{total:.4f}", f"{avg:.4f}"))

def save(results: List[Result]):
    path = os.path.expanduser("~/.affine/results")
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f"{int(time.time())}.json")
    serial = [r.model_dump() for r in results]
    for sr, r in zip(serial, results):
        sr['challenge']['env'] = r.challenge.env.__class__.__name__
    with open(file, "w") as f:
        json.dump(serial, f, indent=2)
    logger.info("Results saved to %s", file)
    click.echo(f"\nResults saved to: {file}\n")

# ── inline submodules (scorer & state) to minimise file count ───────────────
# These blocks replicate the previous ``affine.scorer`` and ``affine.state``
# files directly inside __init__.py.  The original standalone files can be
# safely removed once this patch is in place since we manually register the
# replacement modules in ``sys.modules`` so external imports (including the
# test-suite) continue to work unchanged.

import sys as _sys, types as _types, json as _json, os as _os, time as _time
from collections import defaultdict as _defaultdict, deque as _deque
from typing import Sequence as _Sequence, Dict as _Dict, Deque as _Deque

import numpy as _np

# ---------------------------------------------------------------------------
# 1. scorer sub-module – pure numpy helpers
# ---------------------------------------------------------------------------
_scorer_mod = _types.ModuleType(__name__ + ".scorer")

# ---- public API ------------------------------------------------------------

def window_average(window_sum: "_np.ndarray", sub_blocks: int) -> "_np.ndarray":
    """Simple average of *window_sum* over *sub_blocks* blocks."""
    return _np.asarray(window_sum, dtype=_np.float64) / float(sub_blocks or 1)


def compute_scores(window_sum: "_np.ndarray", sub_blocks: int, b: float = 0.0) -> "_np.ndarray":
    """Convert cumulative rewards to a normalised weight vector.

    Algorithm:
      1. Average over the window.
      2. Shift by baseline *b* (ghost baseline).
      3. Truncate negatives and renormalise so ∑w = 1.
    """
    avg = window_average(window_sum, sub_blocks)
    shifted = _np.maximum(avg - float(b), 0.0)
    total = shifted.sum()
    if total <= 0.0:
        return _np.full_like(shifted, 1.0 / shifted.size, dtype=_np.float64)
    return shifted / total

# Expose in sub-module + package namespace (advantage_vector removed)
for _name in ("window_average", "compute_scores"):
    setattr(_scorer_mod, _name, globals()[_name])
    globals()[_name] = globals()[_name]  # also at package level

# Register so that ``import affine.scorer`` keeps working
_sys.modules[_scorer_mod.__name__] = _scorer_mod
# Additionally add as attribute so ``from affine import scorer`` works
setattr(sys.modules[__name__], "scorer", _scorer_mod)

# ---------------------------------------------------------------------------
# 2. state sub-module – lightweight Redis-backed persistence
# ---------------------------------------------------------------------------
_state_mod = _types.ModuleType(__name__ + ".state")

# --- configuration ----------------------------------------------------------
MAX_MINERS: int = 256
WINDOW: int = int(_os.getenv("AFFINE_WINDOW", "128"))
setattr(_state_mod, "MAX_MINERS", MAX_MINERS)
setattr(_state_mod, "WINDOW", WINDOW)

# Internal generic store (either Redis or in-memory fallback)
_LOCAL: _Dict[str, _Dict[str, str]] = _defaultdict(dict)
try:  # pragma: no cover – real Redis present
    import redis as _redis  # type: ignore

    _R = _redis.Redis(host=_os.getenv("REDIS_HOST", "localhost"), db=0, decode_responses=True)
    _USE_REDIS = True
except Exception:  # pragma: no cover – fall-back to in-memory store
    _USE_REDIS = False


def _hset(name: str, key: str, val: str) -> None:
    if _USE_REDIS:
        _R.hset(name, key, val)  # type: ignore[attr-defined]
    else:
        _LOCAL[name][key] = val


def _hget(name: str, key: str) -> str | None:
    if _USE_REDIS:
        return _R.hget(name, key)  # type: ignore[attr-defined]
    return _LOCAL.get(name, {}).get(key)


def _hgetall(name: str) -> _Dict[str, str]:
    if _USE_REDIS:
        return _R.hgetall(name)  # type: ignore[attr-defined]
    return dict(_LOCAL.get(name, {}))


def _hdel(name: str, key: str) -> None:
    if _USE_REDIS:
        _R.hdel(name, key)  # type: ignore[attr-defined]
    else:
        _LOCAL.get(name, {}).pop(key, None)


# ---------------------------------------------------------------------------
# Public API – mirrors old ``affine.state``
# ---------------------------------------------------------------------------


def upsert_miner(hotkey: str, uid: int, block_height: int) -> None:
    """Register *hotkey* or refresh metadata, evicting if > MAX_MINERS."""
    meta = _json.dumps({"uid": uid, "block": block_height})
    _hset("miners", hotkey, meta)
    if _hget("vectors", hotkey) is None:
        _hset("vectors", hotkey, _json.dumps([]))


def _vector_avg(vec) -> float:  # type: ignore[explicit-any]
    if isinstance(vec, (list, tuple)) and not vec:
        return 0.0
    if hasattr(vec, "size") and vec.size == 0:  # ndarray
        return 0.0
    return float(_np.mean(vec))


def evict_lowest() -> str | None:
    """Evict the miner with the *lowest* average window score."""
    vectors = get_window_vectors()
    if not vectors:
        return None
    worst = min(vectors.items(), key=lambda kv: (_vector_avg(kv[1]), kv[0]))[0]
    for bucket in ("miners", "vectors", "last_ts"):
        _hdel(bucket, worst)
    return worst


def slide_window_update(score_map: _Dict[str, float]) -> None:
    """Append new *score* for each hotkey, trimming to WINDOW length."""
    for hk, score in score_map.items():
        raw = _hget("vectors", hk)
        hist: _Deque[float] = _deque(_json.loads(raw) if raw else [], maxlen=WINDOW)
        hist.append(float(score))
        _hset("vectors", hk, _json.dumps(list(hist)))

    # Enforce cap after insertion
    while len(_hgetall("miners")) > MAX_MINERS:
        evict_lowest()


def get_window_vectors() -> _Dict[str, _np.ndarray]:
    """Return all vectors as np.float64 arrays."""
    return {hk: _np.asarray(_json.loads(vec), dtype=_np.float64) for hk, vec in _hgetall("vectors").items()}


# Misc helpers (kept for compatibility)
# ---------------------------------------------------------------------------

def decay_factor(delta_blocks: int, half_life: int | None = None) -> float:
    hl = half_life or WINDOW
    return float(0.5 ** (delta_blocks / hl))

# Expose all public names in the sub-module & register it --------------------
for _name in [
    "upsert_miner",
    "slide_window_update",
    "get_window_vectors",
    "evict_lowest",
    "_reset_in_memory",
]:
    setattr(_state_mod, _name, globals()[_name])

_sys.modules[_state_mod.__name__] = _state_mod
setattr(sys.modules[__name__], "state", _state_mod)

# ---------------------------------------------------------------------------
# 3. result_store sub-module – ND-JSON persistence helpers
# ---------------------------------------------------------------------------
_result_store_mod = _types.ModuleType(__name__ + ".result_store")

import os as _rs_os, json as _rs_json

RESULTS_FILE = _rs_os.path.expanduser("~/.affine/results.json")

def _rs_ensure_parent(path: str) -> None:
    _rs_os.makedirs(_rs_os.path.dirname(path), exist_ok=True)


def append(rows: list[dict]) -> None:
    """Append *rows* (list of dict) as ND-JSON lines to *RESULTS_FILE*."""
    if not rows:
        return
    _rs_ensure_parent(RESULTS_FILE)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(_rs_json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load() -> list[dict]:
    """Load and return all test results contained in *RESULTS_FILE*."""
    if not _rs_os.path.exists(RESULTS_FILE):
        return []
    out: list[dict] = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(_rs_json.loads(line))
            except _rs_json.JSONDecodeError:
                continue  # skip malformed line – robustness in prod loops
    return out


def clear() -> None:
    """Truncate *RESULTS_FILE* if it exists (no-op otherwise)."""
    if _rs_os.path.exists(RESULTS_FILE):
        open(RESULTS_FILE, "w").close()


def next_round_id() -> int:
    """Return the next round identifier (monotonically increasing)."""
    rows = load()
    if not rows:
        return 0
    try:
        current_max = max(int(r.get("round", -1)) for r in rows)
        return current_max + 1
    except ValueError:
        return 0

# – expose & register --------------------------------------------------------
for _name in ("RESULTS_FILE", "append", "load", "clear", "next_round_id"):
    setattr(_result_store_mod, _name, globals()[_name])
    globals()[_name] = globals()[_name]  # also at package level

_sys.modules[_result_store_mod.__name__] = _result_store_mod
setattr(sys.modules[__name__], "result_store", _result_store_mod)

# ---------------------------------------------------------------------------
# 4. round_scoring sub-module – sliding-window helper (UI/monitoring)
# ---------------------------------------------------------------------------
_round_mod = _types.ModuleType(__name__ + ".round_scoring")

from typing import Dict as _r_Dict, List as _r_List, Any as _r_Any, Tuple as _r_Tuple
from collections import defaultdict as _r_defaultdict

ENV_NAMES: _r_Tuple[str, str] = ("SAT", "ABD")
ROUNDS_WINDOW: int = 20
SCORE_FILE: str = _rs_os.path.expanduser("~/.affine/score.json")


def compute_scores_window(round_window: int = ROUNDS_WINDOW) -> _r_List[_r_Dict[str, _r_Any]]:
    """Return average score per environment over the last *round_window* rounds."""
    rows = load()
    if not rows:
        return []

    max_round = max(int(r.get("round", -1)) for r in rows)
    min_round = max_round - round_window + 1
    rows = [r for r in rows if int(r.get("round", -1)) >= min_round]

    totals: _r_Dict[str, _r_Dict[str, float]] = _r_defaultdict(lambda: {env: 0.0 for env in ENV_NAMES})
    meta: _r_Dict[str, _r_Tuple[int, int]] = {}

    for r in rows:
        hk = r["hotkey"]
        meta.setdefault(hk, (int(r.get("uid", -1)), int(r.get("block", -1))))
        for env in ENV_NAMES:
            if env in r:
                totals[hk][env] += float(r[env])  # accumulate raw scores (0–5 per round)

    out: _r_List[_r_Dict[str, _r_Any]] = []
    for hk, env_total in totals.items():
        uid, blk = meta.get(hk, (None, None))
        out.append({
            "hotkey": hk,
            "uid": uid,
            "block": blk,
            "SAT": env_total["SAT"],  # total successes over the window (0-100)
            "ABD": env_total["ABD"],
        })

    out.sort(key=lambda d: min(d["SAT"], d["ABD"]), reverse=True)
    return out


def save_scores(table: _r_List[_r_Dict[str, _r_Any]]) -> None:
    _rs_os.makedirs(_rs_os.path.dirname(SCORE_FILE), exist_ok=True)
    with open(SCORE_FILE, "w", encoding="utf-8") as f:
        _rs_json.dump(table, f, indent=2)

# – expose & register --------------------------------------------------------
for _name in ("compute_scores_window", "save_scores", "ENV_NAMES", "ROUNDS_WINDOW", "SCORE_FILE"):
    setattr(_round_mod, _name, globals()[_name])
    globals()[_name] = globals()[_name]

_sys.modules[_round_mod.__name__] = _round_mod
setattr(sys.modules[__name__], "round_scoring", _round_mod)

# ---------------------------------------------------------------------------
# 5. ranking sub-module – decayed score ranking over commit blocks
# ---------------------------------------------------------------------------
_ranking_mod = _types.ModuleType(__name__ + ".ranking")

import math as _rk_math, asyncio as _rk_asyncio, bittensor as _rk_bt
from typing import Dict as _rk_Dict, List as _rk_List, Any as _rk_Any, Tuple as _rk_Tuple

ENV_NAMES_RK: _rk_Tuple[str, str] = ("SAT", "ABD")
PROMPTS_PER_ENV: int = 5
WINDOW_BLOCKS: int = 7_200
SCORE_FILE_RK: str = _rs_os.path.expanduser("~/.affine/score.json")


def _rk_ensure_parent(path: str) -> None:
    _rs_os.makedirs(_rs_os.path.dirname(path), exist_ok=True)


async def build_score_table(*, window_blocks: int = WINDOW_BLOCKS, current_block: int | None = None) -> _rk_List[_rk_Dict[str, _rk_Any]]:
    from . import miners as _rk_miners, Miner as _rk_Miner  # late import to avoid cycles

    tests = load()
    miners_dict = await _rk_miners(no_null=True)
    hk_to_miner: _rk_Dict[str, _rk_Miner] = {m.hotkey: m for m in miners_dict.values() if m.model}

    if not current_block:
        try:
            current_block = _rk_bt.subtensor().get_current_block()
        except Exception:
            current_block = max((m.block or 0) for m in hk_to_miner.values())

    env_totals: _rk_Dict[str, _rk_Dict[str, float]] = {hk: {env: 0.0 for env in ENV_NAMES_RK} for hk in hk_to_miner}

    for row in tests:
        hk = row.get("hotkey")
        if hk not in hk_to_miner:
            continue
        blk = row.get("current_block", row.get("block", 0))
        if current_block - int(blk) > window_blocks
        
            continue
        # RAW format
        if "env" in row and "score" in row:
            env = str(row["env"]).upper()
            if env in ENV_NAMES_RK:
                env_totals[hk][env] += float(row.get("score", 0.0))
        else:  # Aggregated
            for env in ENV_NAMES_RK:
                if env in row:
                    env_totals[hk][env] += float(row[env])

    running_max: _rk_Dict[str, float] = {env: 0.0 for env in ENV_NAMES_RK}
    table: _rk_List[_rk_Dict[str, _rk_Any]] = []

    for m in sorted(hk_to_miner.values(), key=lambda x: (x.block or _rk_math.inf, x.uid)):
        avg = env_totals.get(m.hotkey, {env: 0.0 for env in ENV_NAMES_RK})
        diff = {env: avg[env] - running_max[env] for env in ENV_NAMES_RK}
        table.append({
            "hotkey": m.hotkey,
            "uid": m.uid,
            "block": m.block,
            "avg": avg,
            "max_prev": running_max.copy(),
            "diff": diff,
        })
        for env in ENV_NAMES_RK:
            running_max[env] = max(running_max[env], avg[env])

    return table


def save_score_table(table: _rk_List[_rk_Dict[str, _rk_Any]]) -> None:
    _rk_ensure_parent(SCORE_FILE_RK)
    with open(SCORE_FILE_RK, "w", encoding="utf-8") as f:
        _rs_json.dump(table, f, indent=2)


async def rank_miners(*, window_blocks: int = WINDOW_BLOCKS) -> _rk_List[_rk_Dict[str, _rk_Any]]:
    table = await build_score_table(window_blocks=window_blocks)
    try:
        current_block = _rk_bt.subtensor().get_current_block()
    except Exception:
        current_block = max(row["block"] or 0 for row in table)

    ranked: _rk_List[_rk_Dict[str, _rk_Any]] = []
    for row in table:
        diff_values = list(row["diff"].values())
        raw_score = sum(diff_values) / len(diff_values) if diff_values else 0.0
        commit_block = row["block"] or current_block
        age_blocks = max(current_block - commit_block, 0)
        decay_factor = 2 ** (age_blocks // window_blocks)
        decayed_score = raw_score / max(decay_factor, 1)
        ranked.append({
            "hotkey": row["hotkey"],
            "uid": row["uid"],
            "score": decayed_score,
            "raw_score": raw_score,
            "decay_factor": decay_factor,
        })

    ranked.sort(key=lambda d: d["score"], reverse=True)
    return ranked


async def compute_and_save_scores(window_blocks: int = WINDOW_BLOCKS) -> _rk_Dict[str, _rk_Any] | None:
    table = await build_score_table(window_blocks=window_blocks)
    save_score_table(table)
    ranking = await rank_miners(window_blocks=window_blocks)
    return ranking[0] if ranking else None

# – expose & register --------------------------------------------------------
for _name in (
    "build_score_table",
    "save_score_table",
    "rank_miners",
    "compute_and_save_scores",
):
    setattr(_ranking_mod, _name, globals()[_name])
    globals()[_name] = globals()[_name]

_sys.modules[_ranking_mod.__name__] = _ranking_mod
setattr(sys.modules[__name__], "ranking", _ranking_mod)

# Misc helpers (kept for compatibility)
# ---------------------------------------------------------------------------

def decay_factor(delta_blocks: int, half_life: int | None = None) -> float:
    hl = half_life or WINDOW
    return float(0.5 ** (delta_blocks / hl))



if __name__ == "__main__":
    cli()
