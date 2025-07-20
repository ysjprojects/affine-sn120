import os
import re
import sys
import random
import asyncio
import tempfile
import subprocess
import affine as af
from typing import Tuple
from threading import Lock
from typing import Any, Dict
from collections import deque
from contextlib import contextmanager

MODELS = ["unsloth/gemma-3-12b-it", "Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B"]
PROMPT_TEMPLATE = """You are a programming expert. Given a Python program and its expected output, you need to determine the exact input that would produce this output.

Program:
```python
{program}
```

Expected Output:
```
{output}
```

Task: Analyze the program to understand what input format it expects from stdin, then provide the input data that would produce the expected output.

You can provide any explanations, analysis, or reasoning you want. However, you MUST include the input data within <INPUT> </INPUT> tags.

Format the input data like this:
<INPUT>
[input data here - each line on a separate line as the program expects]
</INPUT>

I will extract only the content between these tags.

Requirements for the input data within the tags:
1. Each line of input should be on a separate line
2. Use the exact format the program expects  
3. Provide the raw input values that should be fed to stdin
4. Do not include any prefixes or extra formatting within the INPUT tags

Please analyze the program and provide the required input:"""

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

DEFAULT_PROGRAM_EXECUTION_TIMEOUT = 30

class ProgramExecutor:
    """Handles execution of Python programs with input data"""
    
    def __init__(self):
        self.temp_files = []
        self.temp_files_lock = Lock()
    
    @contextmanager
    def create_temp_file(self, content: str, suffix: str = '.py'):
        """Context manager for creating and cleaning up temporary files"""
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_file = f.name
            
            with self.temp_files_lock:
                self.temp_files.append(temp_file)
            
            yield temp_file
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file)
                    with self.temp_files_lock:
                        if temp_file in self.temp_files:
                            self.temp_files.remove(temp_file)
                except Exception:
                    pass
    
    def execute_program(self, program: str, input_data: str, timeout: int = DEFAULT_PROGRAM_EXECUTION_TIMEOUT) -> Tuple[str, str]:
        """Execute program with input data and return output and error"""
        try:
            # Clean up program code
            program = self._clean_program_code(program)
            
            # Use context manager for temp file
            with self.create_temp_file(program, '.py') as prog_file_path:
                process = subprocess.run(
                    [sys.executable, prog_file_path],
                    input=input_data,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    encoding='utf-8'
                )
                
                return process.stdout, process.stderr
                
        except subprocess.TimeoutExpired:
            return "", "Program execution timed out"
        except Exception as e:
            return "", f"Execution error: {str(e)}"
    
    def _clean_program_code(self, program: str) -> str:
        """Clean up program code by removing markdown formatting"""
        if program.startswith('```python'):
            program = program.replace('```python', '').replace('```', '').strip()
        elif program.startswith('```'):
            program = program.replace('```', '').strip()
        return program
    
    def cleanup(self):
        """Clean up temporary files"""
        with self.temp_files_lock:
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception:
                    pass
            self.temp_files.clear() 

class ABDUCTION(af.BaseEnv):
    def __init__(self):
        super().__init__()
        self._executor = ProgramExecutor()
        self._buffer: deque[Dict[str, Any]] = deque()
        self._buffer_size: int = 10
        af.logger.trace("ABDUCTION environment initialized.")
        
    async def _fill_buffer(self) -> None:
        """
        Fetch up to `self._buffer_size` rows from HF and store them locally.
        """
        while len(self._buffer) < self._buffer_size:
            offset = random.randint(0, max(0, 34_824 - 1))
            rows_url = (
                f"https://datasets-server.huggingface.co/rows?"
                f"dataset=satpalsr/rl-python&config=default"
                f"&split=train&offset={offset}&length=1"
            )
            try:
                async with af.aiohttp.ClientSession() as sess:
                    async with sess.get(rows_url, timeout=30) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
            except Exception:
                continue

            for r in data.get("rows", []):
                row = r.get("row")
                if row:
                    self._buffer.append(row)
                    break  # got one valid, go refill until full
                
    async def random_sample(self) -> Dict[str, Any] | None:
        """
        Return one row from the local buffer, refilling if empty.
        """
        if not self._buffer:
            await self._fill_buffer()
        return self._buffer.popleft() if self._buffer else None
        
    async def _create_challenge(
        self, program: str, example_in: str, example_out: str
    ) -> af.Challenge:
        """Use the LLM to propose a new input, validate & execute."""
        af.logger.trace("Creating challenge with program, example input, and example output.")
        prompt = INPUT_GENERATION_PROMPT.format(
            program=program,
            example_input=example_in,
            example_output=example_out
        )
        af.logger.trace(f"Generated prompt for LLM: {prompt[:10]}...")
        resp = await af.query(prompt, model=random.choice( MODELS ))
        llm_resp = resp.response
        if not llm_resp:
            af.logger.trace(f"No response from LLM error: {resp.error}, continuing to next iteration.")
            return None

        gen_input = self.extract_input_from_response(llm_resp)
        af.logger.trace(f"Extracted input from LLM response: {gen_input}")
        if not self._validate_input_for_program(program, gen_input):
            af.logger.trace("Generated input insufficient, retrying")
            return None

        output, error = self._executor.execute_program(program, gen_input)
        af.logger.trace(f"Executed program with generated input. Output: {output}, Error: {error}")
        if error or not output.strip():
            af.logger.trace("Generated input contains error")
            return None            

        af.logger.trace("Challenge created successfully.")
        return af.Challenge(
            env=self,
            prompt=PROMPT_TEMPLATE.format(program=program, output=output),
            extra={"program": program, "expected_output": output},
        )

    async def generate(self) -> af.Challenge:
        af.logger.trace("Generating a new challenge.")
        while True:
            sample = await self.random_sample()
            if sample is None:
                af.logger.trace(f"Failed generation with null sample, continuing...")
                await asyncio.sleep(5)
                continue

            program     = sample.get("program")
            example_in  = sample.get("inputs", "")
            example_out = sample.get("output", "")
            af.logger.trace(f"Sampled program: {program[:50]}…, example in: {example_in}, example out: {example_out}")
            if not (program and example_in and example_out):
                await asyncio.sleep(5)
                af.logger.trace(f"Failed generation with null program, continuing...")
                continue

            challenge = await self._create_challenge(program, example_in, example_out)
            if challenge is not None:
                af.logger.trace("Generated challenge successfully.")
                return challenge 
            else:
                af.logger.trace(f"Failed generation with null challenge, continuing...")
                await asyncio.sleep(5)
                continue

    def extract_input_from_response(self, response: str) -> str:
        """Pull out the last <INPUT>…</INPUT> block."""
        af.logger.trace(f"Extracting input from response: {response[:50]}...")
        # strip any think tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        matches = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)
        if not matches:
            af.logger.trace("No <INPUT> tags found in response.")
            return ""
        lines = [ln.rstrip() for ln in matches[-1].strip().splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        extracted_input = "\n".join(lines)
        af.logger.trace(f"Extracted input: {extracted_input}")
        return extracted_input

    def _validate_input_for_program(self, program: str, inp: str) -> bool:
        """Heuristic: ensure at least as many lines as input() calls."""
        af.logger.trace(f"Validating input for program. Program: {program[:50]}..., Input: {inp}")
        calls = program.count("input()")
        lines = inp.splitlines() if inp else []
        if "for _ in range(int(input()))" in program and lines and lines[0].isdigit():
            valid = len(lines) > int(lines[0])
            af.logger.trace(f"Validation result for loop-based input: {valid}")
            return valid
        valid = len(lines) >= calls
        af.logger.trace(f"Validation result: {valid}")
        return valid

    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Normalize line endings & trailing whitespace."""
        af.logger.trace(f"Comparing outputs. Expected: {expected}, Actual: {actual}")
        if expected == actual:
            af.logger.trace("Outputs match exactly.")
            return True
        exp = expected.strip().replace("\r\n", "\n")
        act = actual.strip().replace("\r\n", "\n")
        if exp == act:
            af.logger.trace("Outputs match after normalization.")
            return True
        match = [l.rstrip() for l in exp.splitlines()] == [l.rstrip() for l in act.splitlines()]
        af.logger.trace(f"Outputs match after line comparison: {match}")
        return match

    async def evaluate(self, challenge: af.Challenge, response: af.Response) -> af.Evaluation:
        af.logger.trace(f"Evaluating response. Challenge: {challenge}, Response: {response}")
        prog = challenge.extra["program"]
        expected = challenge.extra["expected_output"]
        gen_in = self.extract_input_from_response(response.response or "")
        if not gen_in:
            af.logger.trace("No input found in response.")
            return af.Evaluation(
                env=self, score=0.0,
                extra={"error": "No input found", "expected_output": expected}
            )
        out, err = self._executor.execute_program(prog, gen_in)
        af.logger.trace(f"Executed program with generated input. Output: {out}, Error: {err}")
        if err:
            af.logger.trace("Error occurred during program execution.")
            return af.Evaluation(
                env=self, score=0.0,
                extra={"error": err, "generated_output": out}
            )
        ok = self.compare_outputs(expected, out)
        af.logger.trace(f"Output comparison result: {ok}")
        return af.Evaluation(
            env=self, score=1.0 if ok else 0.0,
            extra={"outputs_match": ok, "generated_input": gen_in, "generated_output": out, 'expected_output': expected}
        )
