import os
import sys
import time
import click
import random
import aiohttp
import asyncio
import bittensor as bt
from dotenv import load_dotenv
from abc import ABC, abstractmethod
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
    api = f"https://api.chutes.ai/chutes/AlphaTao/{model}"
    token = os.getenv("CHUTES_API_KEY", "")
    hdr = {"Authorization": token}

    async with aiohttp.ClientSession() as session:
        async with session.get(api, headers=hdr) as r:
            text = await r.text(errors="ignore")
            if r.status != 200:
                raise RuntimeError(f"{r.status}:{text}")
            chute_info =  await r.json()
            del chute_info['readme']
            del chute_info['chords']
            return chute_info

async def miners(uids: Optional[Union[int, List[int]]] = None) -> Dict[int, Miner]:
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
    miners: Optional[Union[Dict[int, Miner], int, List[int]]] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.0
) -> List[Result]:
    if not isinstance(challenges, list):
        challenges = [challenges]

    if isinstance(miners, dict):
        mmap = miners
    else:
        mmap = await miners(miners)

    valid = {uid: m for uid, m in mmap.items() if m.model}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        tasks = [
            (uid, chal, _run_one(sess, chal, m.model, timeout, retries, backoff))
            for uid, m in valid.items()
            for chal in challenges
        ]
        responses = await asyncio.gather(*(t[2] for t in tasks))

    results: List[Result] = []
    for (uid, chal, _), resp in zip(tasks, responses):
        ev = await chal.evaluate(resp)
        results.append(Result(miner=valid[uid], challenge=chal, response=resp, evaluation=ev))

    return results



from .envs.COIN import COIN
from .envs.SAT import SAT

ENVS = {
    "COIN": COIN,
    "SAT": SAT,
}

@click.group()
def cli():
    """Affine"""
    pass

@cli.command('run')
@click.argument('uids', callback=lambda _ctx, _param, val: [int(x) for x in val.split(',')])
@click.argument('env',  type=click.Choice(list(ENVS), case_sensitive=False))
@click.option('-n',    default=1, show_default=True, help='Number of challenges')
def run_command(uids, env, n):
    """Run N challenges in ENV against miners with UIDs."""
    async def _coro():
        ch = await ENVS[env.upper()]().many(n)
        ms = await miners(uids)
        return await run(challenges=ch, miners=ms)
    print(asyncio.run(_coro()))
    
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