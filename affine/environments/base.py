from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel
from affine.llm import LLMClient


class GeneratedQuestion(BaseModel):
    env: "BaseEnv"
    data: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

    @property
    def prompt(self) -> str:
        return self.data["question"]

    async def validate(self, response: str, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
        return await self.env.validate(self.data, response, llm_client)


class BaseEnv(BaseModel, ABC):
    """Abstract environment: generates questions and verifies responses."""
    name: str = "Base"

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def _generate(self, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
        """Return a new prompt/question to send to the LLM."""
        pass

    async def generate(self, llm_client: Optional[LLMClient] = None) -> GeneratedQuestion:
        """Return a new prompt/question to send to the LLM."""
        generated_data = await self._generate(llm_client)
        return GeneratedQuestion(env=self, data=generated_data)

    @abstractmethod
    async def validate(
        self, generated_data: Dict[str, Any], response: str, llm_client: Optional[LLMClient] = None
    ) -> Dict[str, Any]:
        """
        Inspect the LLM's response and return a dict of metrics,
        e.g. {"correct": True, "score": 0.87, ...}.
        """
        pass 