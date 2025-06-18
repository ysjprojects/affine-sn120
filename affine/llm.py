import os
import time
import json
import random
import asyncio
import logging
from typing import Any, Dict, Tuple

import aiohttp
from aiohttp import ClientError, ContentTypeError

from affine.config import LLMSettings
from affine.exceptions import ConfigurationError, LLMAPIError

logger = logging.getLogger("affine")

# ──────────────────────────────────────────────────────────────────────────────
# Simple LLM client using one shared session
# ──────────────────────────────────────────────────────────────────────────────
class LLMClient:
    def __init__(self, session: aiohttp.ClientSession, settings: LLMSettings):
        self.session = session
        self.settings = settings
        if not self.settings.api_key:
            raise ConfigurationError("CHUTES_API_KEY not set. Export it and retry.")

    async def prompt(self, prompt: str, model: str) -> Tuple[str, float, int]:
        attempt = 0
        start = time.monotonic()
        
        # TRACE: Log the full prompt being sent
        logger.log(logging.DEBUG - 5, f"Sending prompt to {model}: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        while True:
            # Total attempts will be attempt + 1 (since attempt is num retries)
            total_attempts = attempt + 1
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                logger.debug(f"POST → {model} (attempt {total_attempts})")
                logger.log(logging.DEBUG - 5, f"Full payload: {json.dumps(payload, indent=2)}")
                
                async with self.session.post(
                    self.settings.api_url, json=payload, timeout=self.settings.timeout
                ) as resp:
                    try:
                        text = await resp.text()
                    except (UnicodeDecodeError, aiohttp.ClientPayloadError):
                        text = "<binary response>"

                    logger.log(logging.DEBUG - 5, f"Raw response status: {resp.status}")
                    
                    if resp.status != 200:
                        logger.debug(f"HTTP {resp.status}: {text[:200]}{'...' if len(text) > 200 else ''}")
                        raise LLMAPIError(f"HTTP {resp.status}: {text}", status_code=resp.status)
                    
                    try:
                        data = await resp.json()
                    except ContentTypeError as e:
                         raise LLMAPIError(f"Failed to decode JSON response: {e}", status_code=resp.status)

                    result = data["choices"][0]["message"]["content"]
                    latency = time.monotonic() - start
                    
                    if result is None:
                        logger.debug(f"← {latency:.2f}s (No content) [attempt {total_attempts}]")
                    else:
                        logger.debug(f"← {latency:.2f}s ({len(result)} chars) [attempt {total_attempts}]")
                    logger.log(logging.DEBUG - 5, f"Full response: {result}")
                    return result, latency, total_attempts

            except Exception as e:
                attempt += 1
                logger.debug(f"Request failed: {type(e).__name__}: {e}")
                
                if attempt > self.settings.max_retries:
                    logger.error(f"Failed after {attempt} retries: {e}")
                    raise
                    
                backoff = self.settings.backoff_base * (2 ** (attempt - 1))
                jitter = random.uniform(-0.1 * backoff, 0.1 * backoff)
                wait = backoff + jitter
                logger.warning(f"retry {attempt}/{self.settings.max_retries} in {wait:.2f}s (backoff: {backoff:.2f}s)")
                await asyncio.sleep(wait) 