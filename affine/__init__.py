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
from typing import Any, Dict, List, Optional, Union

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
        return [await self.generate() for _ in range(n)]

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
    for attempt in range(1, retries + 2):
        try:
            async with session.post(
                url,
                json={"model": model, "messages": [{"role": "user", "content": chal.prompt}]},
                headers=hdr,
                timeout=timeout
            ) as r:
                text = await r.text(errors="ignore")
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
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        async def run_one(m, c):
            resp = await _run_one(sess, c, m.model, timeout, retries, backoff)
            ev = await c.evaluate(resp)
            return Result(miner=m, challenge=c, response=resp, evaluation=ev)
        tasks = [asyncio.create_task(run_one(m, c)) for m in valid for c in challenges]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
    logger.info("Finished %d runs", len(results))
    return results

# ── CLI & commands ────────────────────────────────────────────────────────────
from .envs.coin import COIN
from .envs.sat import SAT

ENVS = {"COIN": COIN, "SAT": SAT}

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
@click.option('--blocks-until-reveal', default=720, help='Blocks until reveal on Bittensor')
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

@cli.command('validate')
@click.option("--k-factor", "-k", default=32, show_default=True, help="Elo K-factor")
@click.option("--challenges", "-c", default=2, show_default=True, help="SAT challenges per game")
@click.option("--delay", "-d", default=5.0, show_default=True, help="Retry delay (s)")
def validator(k_factor: int, challenges: int, delay: float):
    """Continuously pit two random miners head-to-head."""
    elo: Dict[str, float] = load_elo()
    if isinstance(elo, dict) and not isinstance(elo, defaultdict):
        elo = defaultdict(lambda: 1500.0, elo)

    def update_elo(a: Miner, b: Miner, score_a: float):
        ra, rb = elo[a.hotkey], elo[b.hotkey]
        qa, qb = 10**(ra/400), 10**(rb/400)
        ea, eb = qa/(qa+qb), 1 - qa/(qa+qb)
        elo[a.hotkey] = ra + k_factor*(score_a - ea)
        elo[b.hotkey] = rb + k_factor*((1-score_a) - eb)
        save_elo(elo)
        logger.trace("ELO updated: %s→%.1f, %s→%.1f", a.hotkey, elo[a.hotkey], b.hotkey, elo[b.hotkey])

    async def play_game(a: Miner, b: Miner, progress=None, task_id=None):
        # Announce match
        if progress is None:
            click.echo(f"▶ Match {a.uid} vs {b.uid}")
        else:
            progress.console.print(f"▶ Match {a.uid} vs {b.uid}")

        # Run the challenges for this pair
        results = await run(challenges=await SAT().many(challenges), miners=[a, b], progress=False)

        # Aggregate scores
        sa = sum(r.evaluation.score for r in results if r.miner.hotkey == a.hotkey)
        sb = sum(r.evaluation.score for r in results if r.miner.hotkey == b.hotkey)

        # Determine outcome and update Elo
        if sa != sb:
            winner = a if sa > sb else b
        else:
            winner = a if a.block < b.block else b

        loser = b if winner is a else a

        # Single-line Elo update equivalent to previous logic
        update_elo(a, b, float(sa > sb) if sa != sb else float(a.block < b.block))

        # Compose message
        if sa != sb:
            msg = f"🏆 Winner: {winner.uid} (score {sa:.2f} – {sb:.2f}) against {loser.uid}"
        else:
            msg = f"🤝 Draw on scores ({sa:.2f} – {sb:.2f}), tiebreak winner {winner.uid}"

        # Output
        if progress is None:
            click.echo(msg)
        else:
            progress.console.print(msg)

        # Mark progress bar complete if provided
        if progress is not None and task_id is not None:
            progress.update(task_id, advance=1)

    async def main_loop():
        """Run continuous validation cycles with concurrent matches.

        Each cycle:
          1. Fetch up-to-date miners (with a model).
          2. Shuffle and take up to 64 ⇒ pair into ⌊n/2⌋ matches (max 32).
          3. Launch a play_game task per pair and await them concurrently.
        """
        while True:
            all_miners_dict = await miners(no_null=True)

            # Keep only miners that have registered a model
            available = [m for m in all_miners_dict.values() if m.model]
            if len(available) < 2:
                logger.debug("Only %d miner(s) with model; retrying in %ds", len(available), delay)
                await asyncio.sleep(delay)
                continue

            random.shuffle(available)

            # Limit to first 64 miners (or less) then pair them
            sample_size = min(64, len(available))
            selected = available[:sample_size]

            pairs = [(selected[i], selected[i + 1])
                     for i in range(0, (sample_size // 2) * 2, 2)]
            if not pairs:
                await asyncio.sleep(delay)
                continue

            # Keep at most 32 matches
            pairs = pairs[:32]

            logger.debug("Validator batch: %d matches queued", len(pairs))

            try:
                from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
                rich_available = True
            except ImportError:
                rich_available = False

            if rich_available:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    transient=True,
                )

                with progress:
                    tasks = []
                    for a, b in pairs:
                        t_id = progress.add_task(f"{a.uid} vs {b.uid}", total=1)
                        tasks.append(asyncio.create_task(play_game(a, b, progress, t_id)))

                    await asyncio.gather(*tasks)
            else:
                # Fallback: no progress bars, just run concurrently
                tasks = [asyncio.create_task(play_game(a, b)) for a, b in pairs]
                await asyncio.gather(*tasks)

            # ── leaderboard ─────────────────────────────────────────
            if elo:
                sorted_elo = sorted(elo.items(), key=lambda kv: kv[1], reverse=True)
                click.echo("\n🏅 Elo leaderboard:")
                click.echo("{:<5} {:<48} {:>8}".format("#", "Hotkey", "Elo"))
                click.echo("-"*65)
                for rank, (hk, rating) in enumerate(sorted_elo, 1):
                    click.echo(f"{rank:<5} {hk:<48} {rating:>8.1f}")

            # Wait before the next batch
            await asyncio.sleep(delay)

    logger.info("Starting validator (k=%d, c=%d, d=%.1f)", k_factor, challenges, delay)
    asyncio.run(main_loop())

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

if __name__ == "__main__":
    cli()
