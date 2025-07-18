import random, re
from datasets import load_dataset
import affine as af

class GPQA(af.BaseEnv):
    def __init__(self):
        super().__init__()
        self._dataset = None
    
    def _get_dataset(self):
        if self._dataset is None:
            self._dataset = list(load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train"))
        return self._dataset
    
    async def generate(self):
        sample = random.choice(self._get_dataset())
        prompt = f"""Answer this question:

{sample['Question']}

Provide your answer in <ANSWER></ANSWER> tags."""
        return af.Challenge(env=self, prompt=prompt, extra={"correct": sample['Correct Answer']})
    
    async def _llm_verify(self, question, correct, miner_answer):
            
        prompt = f"""Compare these two answers to determine if they are equivalent:

Question: {question[:200]}...

Reference Answer: {correct}
Student Answer: {miner_answer}

Are these answers equivalent in meaning? Consider:
- Chemical names/formulas that refer to the same compound
- Different but correct representations of the same answer
- Minor formatting differences

Reply ONLY with "YES" or "NO"."""
        
        url = "https://llm.chutes.ai/v1/chat/completions"
        hdr = {"Authorization": f"Bearer {af.get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
        
        async with af.aiohttp.ClientSession() as sess:
            try:
                async with sess.post(url, json={"model": "Qwen/Qwen2.5-72B-Instruct", "messages":[{"role":"user","content":prompt}]}, headers=hdr, timeout=600) as r:
                    if r.status == 200:
                        body = await r.json()
                        llm_response = body["choices"][0]["message"]["content"]
                        if llm_response:
                            clean_response = llm_response.strip().upper()
                            af.logger.debug(f"GPQA LLM response: '{clean_response}'")
                            return "YES" in clean_response
                    else:
                        af.logger.debug(f"GPQA HTTP error: {r.status}")
            except Exception as e:
                af.logger.debug(f"GPQA LLM verify error: {e}")
        
        correct_norm = correct.lower().strip()
        miner_norm = miner_answer.lower().strip()
        fallback_result = (correct_norm in miner_norm or 
                          miner_norm in correct_norm or 
                          correct_norm == miner_norm)
        af.logger.debug(f"GPQA using fallback comparison: {fallback_result}")
        return fallback_result

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        correct = challenge.extra["correct"]
        answer = re.search(r'<ANSWER>(.*?)</ANSWER>', response.response or "", re.DOTALL | re.IGNORECASE)
        if not answer:
            return af.Evaluation(env=self, score=0.0)
        
        miner_answer = answer.group(1).strip()
        is_correct = await self._llm_verify(challenge.prompt, correct, miner_answer)
        return af.Evaluation(env=self, score=float(is_correct), extra={"correct": correct, "got": miner_answer})