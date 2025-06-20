import asyncio
import json
from typing import List, Optional, Dict, Any

import aiohttp
from affine.config import settings
from affine.llm import LLMClient
from affine.utils import get_output_path
from affine.environments.base import GeneratedQuestion


async def query(prompt: str, model: str) -> str:
    """
    Query a model with a given prompt.
    """
    headers = {
        "Authorization": f"Bearer {settings.llm.api_key}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        client = LLMClient(session, settings.llm)
        response, _, _ = await client.prompt(prompt, model)
        return response


def save(
    dataset: List[GeneratedQuestion],
    responses: List[str],
    results: List[Dict[str, Any]],
    out: Optional[str] = None,
    model: str = "default_model"
):
    """
    Save the dataset, responses, and results to a file.
    """
    if not dataset:
        print("Dataset is empty, nothing to save.")
        return

    env_name = dataset[0].env.name
    output_path = get_output_path(model, env_name, out)
    output_data = {
        "dataset": [d.data for d in dataset],
        "responses": responses,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}") 