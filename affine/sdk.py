import asyncio
from typing import List, Optional, Dict, Any
import nest_asyncio

from affine.runner import run_llm_batch
from affine.environments.base import BaseEnv
from affine.config import settings
from affine.utils import setup_logging
from affine.results import get_results as get_results_internal

nest_asyncio.apply()

async def run(
    models: List[str],
    n: int = 1,
    c: Optional[int] = None,
    out: Optional[str] = None,
    env: BaseEnv = None,
    log_level: str = "INFO",
):
    """
    Run evaluations on LLMs.

    :param models: A list of model names to run.
    :param n: Number of questions to generate per model.
    :param c: Number of concurrent LLM queries.
    :param out: Output file path (optional).
    :param env: An environment instance.
    :param log_level: Set logging level.
    """
    if c:
        settings.app.concurrency = c
    
    settings.app.log_level = log_level.upper()
    setup_logging()

    await run_llm_batch(models, n, out, env)

def results(
    models: List[str],
    env: BaseEnv,
) -> List[List[Dict[str, Any]]]:
    """
    Get results for a given list of models and an environment class.
    
    :param models: A list of model names.
    :param env: An environment class (e.g., af.environments.SAT1).
    :return: A list of lists of result dictionaries.
    """
    return get_results_internal(models=models, env=env, display=False) 