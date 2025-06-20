import logging
from typing import Any, Dict, List, Optional
from typing_extensions import Self
import random

from pydantic import PrivateAttr
import reasoning_gym as rg
from affine.llm import LLMClient
from affine.environments.base import BaseEnv

logger = logging.getLogger("tool")

class CMPLX(BaseEnv):
    _gym: rg.algebra.complex_arithmetic.ComplexArithmeticDataset = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._gym = rg.algebra.complex_arithmetic.ComplexArithmeticDataset(
            rg.algebra.complex_arithmetic.ComplexArithmeticConfig()
        )

    async def generate_question(self, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
        question_data = self._gym[random.randint(0, len(self._gym) - 1)]
        prompt = question_data["question"] + "\nProvide your answer in the format: real + imaginaryi (e.g., 2.0 + 1.0i)"
        return {"question": prompt, "full_data": question_data}
    
    async def verify(
        self, generated_data: Dict[str, Any], response: str, llm_client: Optional[LLMClient] = None
    ) -> Dict[str, Any]:
        
        question_data = generated_data["full_data"]
        extracted = self._gym.parse_string_to_complex(response)
        score = self._gym.score_answer(
            response, question_data
        )
        return { "correct": bool(score), "extracted": extracted }
