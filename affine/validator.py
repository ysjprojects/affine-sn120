import asyncio
import logging
import affine as af
import bittensor as bt
from math import log, sqrt
from rich.table import Table
from rich.console import Console

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_miner_stats(substrate, netuid: int, env_cfg: tuple):
    """
    Gather evaluation statistics for each unique miner model:
      - Fetch on-chain metadata and reveal commitments
      - Deduplicate by model hash (keep earliest reveal)
      - Batch-fetch trial results for each model
      - Count successes & trials
      - Compute Wilson (95%) intervals and UCB1 exploration scores
    Returns:
        uids: List[int]
        models: Dict[int,str]
        blocks: Dict[int,int]
        trials: Dict[int,int]
        successes: Dict[int,int]
        intervals: Dict[int, (float, float)]
        scores: Dict[int, float]
    """
    meta     = await substrate.metagraph(netuid)
    reveals  = await substrate.get_all_revealed_commitments(netuid)
    uids     = meta.uids

    # Deduplicate by model: keep only the oldest reveal per model hash
    models, blocks = {}, {}
    for uid in uids:
        hot = meta.hotkeys[uid]
        try:
            blk, mdl = reveals[hot][0]
            prev = next((i for i,m in models.items() if m == mdl), None)
            if prev is None or blk < blocks[prev]:
                if prev is not None:
                    models.pop(prev); blocks.pop(prev)
                models[uid], blocks[uid] = mdl, blk
        except KeyError:
            logger.debug(f"UID {uid} has no reveal")

    if not models:
        return uids, {}, {}, {}, {}, {}, {}

    # Batch-fetch affine.results for each model
    raw       = af.results(models=list(models.values()), env=af.environments.SAT1(*env_cfg))
    by_model  = dict(zip(models.values(), raw))

    # Count trials & successes
    trials, successes = {}, {}
    for uid, mdl in models.items():
        recs = by_model.get(mdl, [])
        n    = len(recs)
        s    = sum(1 for r in recs if r.get("metrics", {}).get("correct"))
        trials[uid], successes[uid] = n, s

    # Compute Wilson intervals & UCB1 scores
    total = sum(trials.values()) or 1
    z     = 1.96
    intervals, scores = {}, {}
    for uid in models:
        n, s = trials[uid], successes[uid]
        p    = s/n if n else 0.5
        d    = 1 + z*z/n
        center = (p + z*z/(2*n)) / d
        half   = (z * sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
        L, U   = center-half, center+half
        intervals[uid] = (L, U)
        scores[uid]    = (p + sqrt(2 * log(total) / n)) if n else float('inf')

    return uids, models, blocks, trials, successes, intervals, scores


async def run_validator(
    coldkey: str,
    hotkey: str,
    netuid: int = 120,
    concurrency: int = 20,
    env_cfg: tuple = (3, 2, 3),
    debug: bool = False
):
    """
    Continuously:
      1) collect stats via collect_miner_stats()
      2) eliminate hopeless miners
      3) pick next to sample via UCB1
      4) run trials
      5) compute accuracies & set on-chain weights
      6) display formatted table
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    console   = Console()
    substrate = bt.async_subtensor()
    await substrate.initialize()
    wallet    = bt.wallet(name=coldkey, hotkey=hotkey)

    while True:
        try:
            # -------------------------------------------------------
            # 1) Gather stats for all miners
            #    - on-chain UIDs & hotkeys
            #    - reveal block & model hash
            #    - dedupe identical models
            #    - fetch affine.results in one batch
            #    - count successes/trials
            #    - compute Wilson [L,U] & UCB1 scores
            uids, models, blocks, trials, successes, intervals, scores = (
                await collect_miner_stats(substrate, netuid, env_cfg)
            )

            if not models:
                console.print("No active miner models; retrying in 60s…", style="yellow")
                await asyncio.sleep(60)
                continue

            # -------------------------------------------------------
            # 2) Eliminate clearly losing miners
            #    If a miner’s U < max(other L), it cannot win—drop it.
            best_lower = max(L for L,_ in intervals.values())
            candidates = [uid for uid,(_,U) in intervals.items() if U >= best_lower]
            logger.debug(f"Candidates: {candidates}")

            # -------------------------------------------------------
            # 3) Select next miner to sample (highest UCB1)
            chosen     = max(candidates, key=lambda u: scores[u])
            chosen_mdl = models[chosen]
            logger.info(f"Sampling UID {chosen} → model {chosen_mdl}")

            # -------------------------------------------------------
            # 4) Run a batch of new trials for the chosen model
            await af.run(
                models=[chosen_mdl],
                n=10,
                c=concurrency,
                env=af.environments.SAT1(*env_cfg)
            )

            # -------------------------------------------------------
            # 5) Compute accuracies and set on-chain Dirac weights
            #    Assign weight=1.0 to the UID with highest success rate,
            #    0.0 to all others.
            accuracies = {
                uid: (successes[uid]/trials[uid] if trials[uid] else 0.0)
                for uid in models
            }
            champion = max(accuracies, key=accuracies.get)
            weights  = [1.0 if uid == champion else 0.0 for uid in uids]
            await substrate.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )

            # -------------------------------------------------------
            # 6) Display a rich table with weights included
            table = Table(
                title="Validator Results",
                header_style="bold",
                box=None,
                show_header=True
            )
            for col,align in [
                ("UID","center"),("Model","left"),("Block","center"),
                ("Trials","center"),("Success %","center"),
                ("L [95%]","center"),("U [95%]","center"),
                ("Weight","center")
            ]:
                table.add_column(col, justify=align)

            for uid in uids:
                if uid in models:
                    mdl       = models[uid]
                    n, s      = trials[uid], successes[uid]
                    L, U      = intervals[uid]
                    w         = weights[uid]
                    table.add_row(
                        str(uid),
                        mdl,
                        str(blocks[uid]),
                        str(n),
                        f"{(s/n):.1%}" if n else "N/A",
                        f"{L:.2f}",
                        f"{U:.2f}",
                        f"{w:.1f}"
                    )
            console.print(table)

            # Pause briefly before next iteration
            await asyncio.sleep(10)

        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}\nRetrying in 60s…", style="yellow")
            logger.exception("Validator loop error")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(run_validator("default", "default", debug=True))
