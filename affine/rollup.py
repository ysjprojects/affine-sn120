"""affine.rollup – periodic weight commitment driver.

This helper is intended to be executed *out-of-band* every ``W`` blocks via a
cron-job or daemon.  It rolls the sliding window, computes the final weight
vector and relays it to the (stub) on-chain committer.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Tuple

import numpy as np

from . import state, scorer, deployment


async def _compute_and_commit() -> None:
    """Assemble current window vectors → weight vector → commit."""
    vecs = state.get_window_vectors()
    if not vecs:
        return  # nothing to do yet

    keys = list(vecs)
    stacked = np.array([v.sum() for v in vecs.values()], dtype=np.float64)
    sub_blocks = np.array([len(v) for v in vecs.values()], dtype=int).max(initial=1)

    w_vec = scorer.compute_scores(stacked, int(sub_blocks), b=0.0)
    weights: List[Tuple[str, float]] = list(zip(keys, w_vec.tolist()))

    # Commit using stub (replace later with real bittensor call)
    deployment.commit_weights(weights, block=int(time.time()))


def main() -> None:
    """Synchronous entry-point."""
    asyncio.run(_compute_and_commit())


if __name__ == "__main__":  # pragma: no cover
    main() 