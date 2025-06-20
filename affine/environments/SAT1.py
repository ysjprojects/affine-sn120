import re
import random
import logging
from typing import Any, Dict, List, Optional
from typing_extensions import Self

from pydantic import model_validator, PrivateAttr

from affine.llm import LLMClient
from affine.environments.base import BaseEnv

logger = logging.getLogger("tool")

class SAT1Env(BaseEnv):
    """
    Env that generates random difficult boolean SAT problems with a planted solution.
    Difficulty can be controlled via constructor arguments.
    e.g. -- -e SAT1 -- --n 10 --k 3 --m 42
    """
    name: str = "SAT1"
    n: int = 3 # number of variables
    k: int = 2 # clause size
    m: Optional[int] = None # number of clauses
    
    # Private attributes for internal state
    _idx: int = PrivateAttr(default=0)

    @model_validator(mode='after')
    def validate_env(self) -> Self:
        if self.k > self.n:
            raise ValueError(f"Clause size k ({self.k}) cannot be larger than the number of variables n ({self.n}).")
        if self.m is None:
            self.m = int(4.26 * self.n)
        return self

    async def _generate(self, llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
        # 1) Plant a random solution
        assignment = {i: random.choice([True, False]) for i in range(1, self.n + 1)}
        
        # 2) Generate clauses ensuring each is satisfied by the planted solution
        clauses: List[List[int]] = []
        for _ in range(self.m):
            vars_in_clause = random.sample(range(1, self.n + 1), self.k)
            # pick one variable to guarantee satisfaction
            sat_var = random.choice(vars_in_clause)
            clause: List[int] = []
            for v in vars_in_clause:
                if v == sat_var:
                    # make the literal agree with the planted assignment
                    lit = v if assignment[v] else -v
                else:
                    # other literals random
                    lit = v if random.choice([True, False]) else -v
                clause.append(lit)
            clauses.append(clause)
        
        # Build a human-readable CNF string
        clause_strs = []
        for clause in clauses:
            lits = []
            for lit in clause:
                var = f"x{abs(lit)}"
                if lit < 0:
                    lits.append(f"¬{var}")
                else:
                    lits.append(var)
            clause_strs.append("(" + " ∨ ".join(lits) + ")")
        formula_str = " ∧ ".join(clause_strs)
        
        q = (
            f"Find a satisfying assignment for the following {self.k}-SAT formula over variables x1..x{self.n}:\n"
            f"{formula_str}\n"
            "Provide your answer as comma-separated assignments like `x1=True, x2=False, ...`, "
            "or respond `UNSAT` if it has no solution."
        )
        logger.debug(f"Generated SAT question #{self._idx}: n={self.n}, m={self.m}, planted={assignment}")
        self._idx += 1
        return {
            "question": q,
            "clauses": clauses,
            "assignment": assignment,
        }

    async def validate(
        self,
        generated_data: Dict[str, Any],
        response: str,
        llm_client: Optional[LLMClient] = None
    ) -> Dict[str, Any]:
        # Locate the corresponding formula & assignment
        logger.debug("Starting validation...")

        clauses = generated_data["clauses"]
        expected = generated_data["assignment"]
        logger.debug(f"Loaded expected assignment: {expected}")

        # Guard against None responses that might slip through
        if response is None:
            logger.debug("Response is None, marking as incorrect.")
            return {"correct": False, "expected": expected, "extraction": None, "reason": "No response from model"}

        resp = response.strip()
        logger.debug(f"Validating response: '{resp}'")
        
        # If user claims UNSAT, that's always incorrect (we planted a solution)
        if re.search(r"\\bunsat\\b", resp, re.IGNORECASE):
            logger.debug("Response claims UNSAT, which is incorrect as a solution was planted.")
            extraction = None
            correct = False
        else:
            # Extract assignments of the form x<number>=True/False or x<number>=1/0
            matches = re.findall(r"x(\d+)\s*=\s*(True|False|true|false|1|0)", resp)
            logger.debug(f"Found {len(matches)} variable assignments in response.")
            extraction: Dict[int, bool] = {}
            for var_str, val_str in matches:
                var_index = int(var_str)
                val = val_str.lower() in ("true", "1")
                extraction[var_index] = val
            logger.debug(f"Extracted assignment: {extraction}")

            # Must assign every variable
            if set(extraction.keys()) != set(expected.keys()):
                logger.debug(f"Extracted variables {set(extraction.keys())} do not match expected variables {set(expected.keys())}.")
                correct = False
            else:
                logger.debug("All variables assigned. Evaluating clauses...")
                # Evaluate each clause under the extracted assignment
                def clause_satisfied(cl: List[int]) -> bool:
                    for lit in cl:
                        v = abs(lit)
                        val = extraction[v]
                        satisfied = (lit > 0 and val) or (lit < 0 and not val)
                        if satisfied:
                            return True
                    return False
                
                clause_results = [clause_satisfied(cl) for cl in clauses]
                correct = all(clause_results)
                
                if not correct:
                    failed_clauses = [clauses[i] for i, satisfied in enumerate(clause_results) if not satisfied]
                    logger.debug(f"Evaluation failed. {len(failed_clauses)}/{len(clauses)} clauses were not satisfied.")
                    logger.debug(f"First failing clause: {failed_clauses[0]}")

        logger.debug(
            f"Validation result: correct={correct}"
        )
        return {"correct": correct, "expected": expected, "extraction": extraction if 'extraction' in locals() else None}
