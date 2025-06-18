from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from affine.llm import LLMClient


class BaseEnv(ABC):
    """Abstract environment: generates questions and verifies responses."""
    @abstractmethod
    async def generate_question(self, llm_client: Optional[LLMClient] = None) -> str:
        """Return a new prompt/question to send to the LLM."""
        pass

    @abstractmethod
    async def verify(
        self, question: str, response: str, llm_client: Optional[LLMClient] = None
    ) -> Dict[str, Any]:
        """
        Inspect the LLM's response and return a dict of metrics,
        e.g. {"correct": True, "score": 0.87, ...}.
        """
        pass 