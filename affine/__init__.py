import os
import sys
import json
import time
import click
import random
import aiohttp
import asyncio
import bittensor as bt
from dotenv import load_dotenv
from collections import defaultdict
from abc import ABC, abstractmethod
from alive_progress import alive_bar
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

load_dotenv(os.path.expanduser("~/.affine/config.env"), override=True)
load_dotenv(override=True)
def get_conf(key) -> Any:
    value = os.getenv(key)
    if value is None:
        print( f"{key} not set.\nRun:\n\taf set {key} <your-{key}-value>\n")
        sys.exit(1)
    return value
        
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
        
    async def many(self, n:int) -> List["Challenge"]:
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

async def get_chute(model: str) -> Dict[str, Any]:
    api = f"https://api.chutes.ai/chutes/{model}"
    token = os.getenv("CHUTES_API_KEY", "")
    hdr = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(api, headers=hdr) as r:
            text = await r.text(errors="ignore")
            if r.status != 200:
                raise RuntimeError(f"{r.status}:{text}")
            chute_info =  await r.json()
            [chute_info.pop(k, None) for k in ['readme', 'cords', 'tagline', 'instances']]; chute_info.get('image', {}).pop('readme', None)
            return chute_info

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
        if no_null and blk == None: continue
        out[uid] = Miner(
            uid=uid,
            hotkey=hk,
            model=str(mdl) if mdl is not None else None,
            block=int(blk) if blk is not None else None
        )
    
    miners_with_models = [m for m in out.values() if m.model]
    if miners_with_models:
        chute_infos = await asyncio.gather(
            *[get_chute(m.model) for m in miners_with_models],
            return_exceptions=True
        )

        for miner, chute_info in zip(miners_with_models, chute_infos):
            if not isinstance(chute_info, Exception):
                miner.chute = chute_info

    return out


async def _run_one(
    session: aiohttp.ClientSession,
    chal: Challenge,
    model: str,
    timeout: float,
    retries: int,
    backoff: float
) -> Response:
    api = "https://llm.chutes.ai/v1/chat/completions"
    token = get_conf("CHUTES_API_KEY")
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    start = time.monotonic()

    for attempt in range(1, retries + 2):
        try:
            async with session.post(
                api,
                json={"model": model, "messages": [{"role": "user", "content": chal.prompt}]},
                headers=hdr,
                timeout=timeout
            ) as r:
                text = await r.text(errors="ignore")
                if r.status != 200:
                    raise RuntimeError(f"{r.status}:{text}")
                data = await r.json()
                res = data["choices"][0]["message"]["content"]
                return Response(
                    response=res,
                    latency_seconds=time.monotonic() - start,
                    attempts=attempt,
                    model=model,
                    error=None
                )
        except Exception as e:
            if attempt > retries:
                return Response(
                    response=None,
                    latency_seconds=time.monotonic() - start,
                    attempts=attempt,
                    model=model,
                    error=str(e)
                )
            delay = backoff * (2 ** (attempt - 1)) * (1 + random.uniform(-0.1, 0.1))
            await asyncio.sleep(delay)

    # should never reach
    return Response(
        response=None,
        latency_seconds=time.monotonic() - start,
        attempts=retries + 1,
        model=model,
        error="unreachable code"
    )


async def run(
    challenges: Union[Challenge, List[Challenge]],
    miners: Optional[Union[Dict[int, Miner], List[Miner], int, List[int]]] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.0
) -> List[Result]:
    challenges = challenges if isinstance(challenges, list) else [challenges]
    if isinstance(miners, dict): 
        mmap = miners
    elif isinstance(miners, list) and all(isinstance(m, Miner) for m in miners):
        mmap = {m.uid: m for m in miners}
    else:
        mmap = await miners(miners)
    valid_miners = [m for m in mmap.values() if m.model]
    total = len(valid_miners) * len(challenges)
    results: List[Result] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        async def run_one(miner, chal):
            resp = await _run_one(sess, chal, miner.model, timeout, retries, backoff)
            ev = await chal.evaluate(resp)
            return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)
        tasks = [asyncio.create_task(run_one(miner, chal))
                 for miner in valid_miners for chal in challenges]
        with alive_bar(total, title='AF') as bar:
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                bar()
    return results

from .envs.coin import COIN
from .envs.sat import SAT

ENVS = {
    "COIN": COIN,
    "SAT": SAT,
}

@click.group()
def cli():
    """Affine"""
    pass


def print_results(results):
    # build stats per UID
    stats = defaultdict(lambda: {'model': None, 'scores': []})
    for r in results:
        entry = stats[r.miner.uid]
        entry['model']  = r.miner.model
        entry['scores'].append(r.evaluation.score)
    fmt = "{:<5} {:<25} {:<5} {:>7} {:>8}"
    fifty = 5 + 1 + 25 + 1 + 5 + 1 + 7 + 1 + 8  # sum of field widths + spaces
    click.echo(fmt.format("UID", "Model", "#", "Total", "Average"))
    click.echo("-" * fifty)
    for uid, data in stats.items():
        cnt   = len(data['scores'])
        total = sum(data['scores'])
        avg   = total / cnt if cnt else 0
        click.echo(fmt.format(uid, data['model'], cnt, f"{total:.4f}", f"{avg:.4f}"))

def save_results(results):
    results_dir = os.path.expanduser("~/.affine/results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{int(time.time())}.json")
    serializable_results = [r.model_dump() for r in results]
    for sr, r in zip(serializable_results, results):
        sr['challenge']['env'] = r.challenge.env.__class__.__name__
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    click.echo(f"\nResults saved to: {results_path}\n")
    return results_path

@cli.command('run')
@click.argument('uids', callback=lambda _ctx, _param, val: [int(x) for x in val.split(',')])
@click.argument('env',  type=click.Choice(list(ENVS), case_sensitive=False))
@click.option('-n',    default=1, show_default=True, help='Number of challenges')
def run_command(uids, env, n):
    """Run N challenges in ENV against miners with UIDs."""
    async def _coro():
        chals = await ENVS[env.upper()]().many(n)
        ms    = await miners(uids)
        return await run(challenges=chals, miners=ms)
    results = asyncio.run(_coro())
    print_results(results)
    save_results(results)
    return results
    
@cli.command('deploy')
@click.argument('filename', type=click.Path(exists=True))
@click.option('--chutes-api-key', default=None, help='Chutes API key')
@click.option('--hf-user', default=None, help='HuggingFace user')
@click.option('--hf-token', default=None, help='HuggingFace token')
@click.option('--chute-user', default=None, help='Chutes user')
@click.option('--wallet-cold', default=None, help='Bittensor coldkey')
@click.option('--wallet-hot', default=None, help='Bittensor hotkey')
def deploy(
    filename,
    chutes_api_key,
    hf_user,
    hf_token,
    chute_user,
    wallet_cold,
    wallet_hot
):
    """Deploy a model or file using provided credentials and configuration."""
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    hf_user = hf_user or get_conf("HF_USER")
    hf_token = hf_token or get_conf("HF_TOKEN")
    chute_user = chute_user or get_conf("CHUTES_USER")
    wallet_cold = wallet_cold or get_conf("BT_COLDKEY")
    wallet_hot = wallet_hot or get_conf("BT_HOTKEY")
    import os

@cli.command('set')
@click.argument('key')
@click.argument('value')
def set(key: str, value: str):
    """Set a key-value pair in ~/.affine/config.env, creating the file and directory if needed."""
    p = os.path.expanduser("~/.affine/config.env")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    lines = [l for l in open(p).readlines() if not l.strip().startswith(f"{key}=")] if os.path.exists(p) else []
    lines.append(f"{key}={value}\n")
    open(p, "w").writelines(lines)
    print(f"Set {key} in {p}")
    
    
def get_average_scores(results):
    stats = defaultdict(list)
    for r in results:
        stats[r.miner.hotkey].append(r.evaluation.score)
    return {hotkey: sum(scores)/len(scores) if scores else 0 for hotkey, scores in stats.items()}

class EloRatingSystem:
    def __init__(self, players, initial_rating=1500, k_factor=32):
        """
        :param players: iterable of player IDs (any hashable)
        :param initial_rating: rating assigned to new players
        :param k_factor: the K‐factor for all rating updates
        """
        self.ratings = {player: initial_rating for player in players}
        self.k = k_factor

    def expected_score(self, rating_a, rating_b):
        """
        Compute expected score for player A against player B.
        """
        qa = 10 ** (rating_a / 400)
        qb = 10 ** (rating_b / 400)
        return qa / (qa + qb)

    def update(self, player_a, player_b, score_a):
        """
        Update ratings for a single game.
        
        :param player_a: ID of first player
        :param player_b: ID of second player
        :param score_a: actual score for player A (1=win, 0=loss, 0.5=draw)
        """
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        ea = self.expected_score(ra, rb)
        eb = 1 - ea

        # score_b is the flip of score_a
        score_b = 1 - score_a

        # New ratings
        self.ratings[player_a] = ra + self.k * (score_a - ea)
        self.ratings[player_b] = rb + self.k * (score_b - eb)
    
@cli.command('validator')
@click.argument('coldkey')
@click.argument('hotkey')
def set(coldkey: str, hotkey: str):
    
    # Elo scoring system.
    kfac = 32
    elo: Dict[str, float] = defaultdict(lambda: 1500)
    def update( miner_a: Miner, miner_b: Miner, score_a ):
        ra = elo[miner_a.hotkey]
        rb = elo[miner_b.hotkey]
        qa = 10 ** (ra / 400)
        qb = 10 ** (rb / 400)
        ea = qa / (qa + qb)
        eb = 1 - ea
        score_b = 1 - score_a
        elo[miner_a.hotkey] = ra + kfac * (score_a - ea)
        elo[miner_b.hotkey] = rb + kfac * (score_b - eb)
    
    async def game( miner_a: Miner, miner_b: Miner ):
        results = await run(
            challenges = SAT().many(1),
            miners = [miner_a, miner_b]
        )
        score_a = 0; score_b = 0
        for r in results:
            if r.miner.hotkey == miner_a.hotkey:
                score_a += r.evaluation.score
            else:
                score_b += r.evaluation.score
        if score_a > score_b:
            update( miner_a, miner_b, 1.0 )
        elif score_a == score_b:
            update( miner_a, miner_b, float( miner_a.block < miner_b.block ))
        else:
            update( miner_a, miner_b, 0.0 )
            
    async def validator_loop():
        while True:
            all_miners = await miners(no_null=True)
            if len(all_miners) < 2:
                raise RuntimeError("Not enough miners to select 2 unique miners.")
            selected_uids = random.sample(list(all_miners.keys()), 2)
            miner_a = all_miners[selected_uids[0]]
            miner_b = all_miners[selected_uids[1]]
            await game(miner_a, miner_b)

    asyncio.run(validator_loop())
 
    
