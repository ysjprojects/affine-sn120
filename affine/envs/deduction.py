# ToDo: Update dataset

import random
import re
from datasets import load_dataset
from typing import Any
import affine as af
from ..utils.program_executor import ProgramExecutor

PROMPT_TEMPLATE = """You are a programming expert. Given a Python program and its input, you need to determine the exact output that would be produced when the program runs with this input.

Program:
```python
{program}
```

Input:
```
{input}
```

Task: Analyze the program step by step to understand its logic, then trace through the execution with the given input to determine what output it would produce.

You can provide any explanations, analysis, or reasoning you want. However, you MUST include the output within <OUTPUT> </OUTPUT> tags.

Format the output like this:
<OUTPUT>
[program output here - each line on a separate line as the program would produce]
</OUTPUT>

I will extract only the content between these tags.

Requirements for the output data within the tags:
1. Each line of output should be on a separate line
2. Use the exact format the program would produce to stdout
3. Provide the raw output values that the program would print
4. Do not include any prefixes or extra formatting within the OUTPUT tags

Please analyze the program and provide the expected output:"""

INPUT_GENERATION_PROMPT = """Given this Python program and an example of how it works, generate a NEW valid input that would be accepted by the program:

Program:
```python
{program}
```

Example:
Input: {example_input}
Output: {example_output}

Now generate a NEW input that would work with this program. Make it different from the example but follow the same format and pattern.

Requirements:
1. Each line of input should be on a separate line
2. Use the exact format the program expects from stdin
3. Provide only raw input values
4. Do not include any prefixes or extra formatting
5. Make it different from the example input

Format your response with <INPUT> </INPUT> tags like this:
<INPUT>
[your new input data here]
</INPUT>

Please generate a valid input:"""


class DEDUCTION(af.BaseEnv):
    dataset_name: str
    
    def __init__(self, dataset_name="satpalsr/rl-python"):
        super().__init__(dataset_name=dataset_name)
        self._executor = ProgramExecutor()
        self._dataset = None
        self._dataset_iterator = None
        
    @property
    def executor(self):
        return self._executor
        
    def _get_dataset(self):
        """Load dataset as an iterator"""
        if self._dataset is None:
            try:
                self._dataset = load_dataset(self.dataset_name, streaming=True)
                if 'train' in self._dataset:
                    self._dataset_iterator = iter(self._dataset['train'])
                else:
                    split_name = list(self._dataset.keys())[0]
                    self._dataset_iterator = iter(self._dataset[split_name])
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset: {str(e)}")
        return self._dataset

    def _get_random_sample(self):
        """Get a random sample from the dataset"""
        if not self._dataset_iterator:
            self._get_dataset()
            
        try:
            return next(self._dataset_iterator)
        except StopIteration:
            # Reset iterator and try again
            if 'train' in self._dataset:
                self._dataset_iterator = iter(self._dataset['train'])
            else:
                split_name = list(self._dataset.keys())[0]
                self._dataset_iterator = iter(self._dataset[split_name])
            return next(self._dataset_iterator)

    async def _llm_generate_input(self, prompt: str) -> str:
        """Generate input using LLM API"""
        url = "https://llm.chutes.ai/v1/chat/completions"
        hdr = {"Authorization": f"Bearer {af.get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
        
        async with af.aiohttp.ClientSession() as sess:
            try:
                async with sess.post(
                    url, 
                    json={
                        "model": "Qwen/Qwen2.5-72B-Instruct", 
                        "messages": [{"role": "user", "content": prompt}]
                    }, 
                    headers=hdr, 
                    timeout=600
                ) as r:
                    if r.status == 200:
                        body = await r.json()
                        return body["choices"][0]["message"]["content"]
                    else:
                        af.logger.debug(f"LLM HTTP error: {r.status}")
            except Exception as e:
                af.logger.debug(f"LLM call error: {e}")
        
        return ""  # fallback
    
    def _validate_input_for_program(self, program: str, input_data: str) -> bool:
        """Validate that input has enough lines for the program"""
        # Count expected input() calls in the program
        input_calls = program.count('input()')
        input_lines = len(input_data.split('\n')) if input_data else 0
        
        # Simple heuristic: if program has loops, it might need more input
        # Look for common patterns that suggest dynamic input reading
        if 'for _ in range(int(input()))' in program:
            # This pattern suggests the first line determines how many test cases
            lines = input_data.split('\n')
            if lines and lines[0].isdigit():
                expected_cases = int(lines[0])
                # Very rough estimate - each case might need multiple lines
                # This is a heuristic, not perfect
                return input_lines > expected_cases
        
        return input_lines >= input_calls
    
    async def _create_challenge_from_sample(self, sample):
        """Create a challenge by generating new input first, then running program to get expected output"""
        program = sample['program']
        
        # Use the sample's existing input-output as example (if available)
        example_input = sample.get('inputs', '')
        example_output = sample.get('output', '')
        
        # If no example available, skip this sample
        if not example_input or not example_output:
            return None
        
        # Try multiple LLM-generated inputs until we get a valid output
        for _ in range(5):  # Try up to 5 different LLM attempts
            try:
                # Generate NEW input using LLM with example
                llm_prompt = INPUT_GENERATION_PROMPT.format(
                    program=program,
                    example_input=example_input,
                    example_output=example_output
                )
                llm_response = await self._llm_generate_input(llm_prompt)
                
                if not llm_response:
                    continue
                
                # Extract input from LLM response
                generated_input = self.extract_input_from_response(llm_response)
                
                if not generated_input:
                    continue
                
                # Validate input before running program
                if not self._validate_input_for_program(program, generated_input):
                    af.logger.debug(f"Generated input appears insufficient for program (attempt {_+1})")
                    continue
                
                # Run the program with generated input to get EXPECTED output
                expected_output, error = self.executor.execute_program(program, generated_input)
                
                if not error and expected_output and expected_output.strip():
                    return {
                        'program': program,
                        'input': generated_input,
                        'expected_output': expected_output
                    }
            except Exception as e:
                af.logger.debug(f"Error in challenge creation: {e}")
                continue
        
        return None

    async def generate(self) -> af.Challenge:
        """Generate a single challenge"""
        # Try multiple samples until we get a valid challenge
        for _ in range(10):  # Try up to 10 samples
            sample = self._get_random_sample()
            af.logger.info(f"Selected sample for challenge generation: {sample}")
            challenge_data = await self._create_challenge_from_sample(sample)
            
            if challenge_data:
                prompt = PROMPT_TEMPLATE.format(
                    program=challenge_data['program'],
                    input=challenge_data['input']
                )
                
                return af.Challenge(
                    env=self,
                    prompt=prompt,
                    extra={
                        "program": challenge_data['program'],
                        "input": challenge_data['input'],
                        "expected_output": challenge_data['expected_output'],
                    }
                )
        
        raise RuntimeError("Failed to generate a valid challenge after 10 attempts")

    def extract_input_from_response(self, response: str) -> str:
        """Extract input data from <INPUT> </INPUT> tags"""
        # Remove thinking tags first
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Look for <INPUT> </INPUT> tags - case insensitive
        input_pattern = r'<INPUT>(.*?)</INPUT>'
        matches = re.findall(input_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Take the last match if multiple exist
            input_data = matches[-1].strip()
            # Clean up the input data
            lines = input_data.split('\n')
            cleaned_lines = []
            
            for line in lines:
                cleaned_line = line.rstrip()
                cleaned_lines.append(cleaned_line)
            
            # Remove empty lines at the end
            while cleaned_lines and not cleaned_lines[-1].strip():
                cleaned_lines.pop()
            
            return '\n'.join(cleaned_lines)
        
        return ""

    def extract_output_from_response(self, response: str) -> str:
        """Extract output data from <OUTPUT> </OUTPUT> tags"""
        # Remove thinking tags first
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Look for <OUTPUT> </OUTPUT> tags - case insensitive
        output_pattern = r'<OUTPUT>(.*?)</OUTPUT>'
        matches = re.findall(output_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Take the last match if multiple exist
            output_data = matches[-1].strip()
            # Clean up the output data
            lines = output_data.split('\n')
            cleaned_lines = []
            
            for line in lines:
                cleaned_line = line.rstrip()
                cleaned_lines.append(cleaned_line)
            
            # Remove empty lines at the end
            while cleaned_lines and not cleaned_lines[-1].strip():
                cleaned_lines.pop()
            
            return '\n'.join(cleaned_lines)
        
        return ""
    
    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Compare expected and actual outputs"""
        # Direct comparison
        if expected == actual:
            return True
        
        # Normalize line endings
        expected_normalized = expected.strip().replace('\r\n', '\n')
        actual_normalized = actual.strip().replace('\r\n', '\n')
        
        if expected_normalized == actual_normalized:
            return True
        
        # Compare line by line (strip trailing whitespace)
        expected_lines = [line.rstrip() for line in expected_normalized.split('\n')]
        actual_lines = [line.rstrip() for line in actual_normalized.split('\n')]
        
        return expected_lines == actual_lines
    
    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        """Evaluate the response by extracting output and comparing with expected output"""
        program = challenge.extra["program"]
        input_data = challenge.extra["input"]
        expected_output = challenge.extra["expected_output"]
        
        # Extract output from the LLM response
        generated_output = self.extract_output_from_response(response.response or "")
        
        if not generated_output:
            return af.Evaluation(
                env=self,
                score=0.0,
                extra={
                    "program": program,
                    "input": input_data,
                    "expected_output": expected_output,
                    "generated_output": generated_output,
                    "error": "No output found in response"
                }
            )
        
        # Compare the generated output with expected output
        outputs_match = self.compare_outputs(expected_output, generated_output)
        score = 1.0 if outputs_match else 0.0
        
        return af.Evaluation(
            env=self,
            score=score,
            extra={
                "program": program,
                "input": input_data,
                "expected_output": expected_output,
                "generated_output": generated_output,
                "outputs_match": outputs_match,
                "error": None
            }
        )
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, '_executor'):
            self._executor.cleanup()