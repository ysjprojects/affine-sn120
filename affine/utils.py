"""
Affine validation module - handles model validation including hot status checks
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Miner
else:
    Miner = None


async def get_chutes_info(model: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
    """Fetches additional information about a model from the Chutes.ai API."""
    from affine import get_conf  # Import here to avoid circular imports
    
    api_key = get_conf("CHUTES_API_KEY")
    if not api_key:
        print(f"No Chutes API key available for model info fetch")
        return None

    # Use the same URL pattern as existing get_chute function
    url = f"https://api.chutes.ai/chutes/{model.replace('/', '_')}"
    headers = {"Authorization": api_key}

    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✓ Fetched info for model '{model}'")
                return data
            else:
                response_text = await response.text()
                print(f"Failed to fetch info for model {model}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching info for model {model}: {e}")
        return None


async def is_model_hot(model: str, session: aiohttp.ClientSession) -> bool:
    """Checks if a model is marked as 'hot' on Chutes.ai by fetching its info."""
    chutes_info = await get_chutes_info(model, session)
    if chutes_info and isinstance(chutes_info, dict):
        is_hot = chutes_info.get("hot", False)
        print(f"Model '{model}' hot status: {is_hot}")
        return is_hot
    else:
        print(f"Could not verify hot status for model '{model}'. Defaulting to False.")
        return False


async def validate_miners_hot(miners_dict: Dict[int, Miner], require_hot: bool = True) -> Dict[int, Miner]:
    """
    Validate miners by checking if their models are hot.
    
    Args:
        miners_dict: Dictionary of miners to validate
        require_hot: If True, filter out non-hot models. If False, just log status.
    
    Returns:
        Dictionary of validated miners (filtered if require_hot=True)
    """
    if not miners_dict:
        return miners_dict
    
    miners_with_models = {uid: miner for uid, miner in miners_dict.items() if miner.model}
    
    if not miners_with_models:
        print("No miners with models to validate")
        return miners_dict
    
    print(f"Validating hot status for {len(miners_with_models)} miners...")
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        # Check hot status for all models in parallel
        hot_checks = await asyncio.gather(
            *[is_model_hot(miner.model, session) for miner in miners_with_models.values()],
            return_exceptions=True
        )
    
    validated_miners = {}
    hot_count = 0
    
    for (uid, miner), is_hot in zip(miners_with_models.items(), hot_checks):
        # Handle exceptions from hot checks
        if isinstance(is_hot, Exception):
            print(f"Error checking hot status for miner {uid} (model: {miner.model}): {is_hot}")
            is_hot = False
        
        if is_hot:
            hot_count += 1
            validated_miners[uid] = miner
        elif not require_hot:
            # Include non-hot miners if not requiring hot status
            validated_miners[uid] = miner
        else:
            print(f"Skipping miner {uid} - model '{miner.model}' is not hot")
    
    # Also include miners without models (they won't be used for inference anyway)
    for uid, miner in miners_dict.items():
        if not miner.model:
            validated_miners[uid] = miner
    
    print(f"Validation complete: {hot_count}/{len(miners_with_models)} models are hot")
    
    if require_hot and hot_count == 0:
        print("Warning: No hot models found. All miners will be filtered out.")
    
    return validated_miners


async def validate_single_model_hot(model: str) -> bool:
    """Convenience function to check if a single model is hot."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        return await is_model_hot(model, session)


async def get_chutes_model_info(model: str) -> dict:
    """Fetches vLLM config for a model from the Chutes.ai API."""
    from affine import get_conf
    api_key = get_conf("CHUTES_API_KEY")
    if not api_key:
        print(f"No Chutes API key available for model info fetch")
        return None
    url = f"https://api.chutes.ai/guess/vllm_config?model={model}"
    headers = {"Authorization": api_key}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✓ vLLM config fetched for model '{model}'")
                    return data
                else:
                    print(f"Failed to fetch vLLM config: HTTP {response.status}")
                    return None
    except Exception as e:
        print(f"Error fetching vLLM config: {e}")
        return None

async def get_model_size(model: str) -> dict:
    chutes_data = await get_chutes_model_info(model)

    model_size = chutes_data['total_model_size']

    return model_size


model_size = asyncio.run(get_model_size("Alphatao/Affine-2501551"))
print(model_size)