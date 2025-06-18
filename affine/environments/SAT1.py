import re
import random
import logging
from typing import Any, Dict, List, Optional

from affine.llm import LLMClient
from affine.environments.base import BaseEnv

logger = logging.getLogger("tool")

class SAT1Env(BaseEnv):
    """
    Env that generates random difficult boolean SAT problems with a planted solution.
    Difficulty can be controlled via constructor arguments.
    e.g. -- -e SAT1 -- --n 10 --k 3 --m 42
    """
    name = "SAT1"

    def __init__(self, n: int = 3, k: int = 2, m: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.n = n # number of variables
        self.k = k # clause size
        
        # Validate that k is not greater than n
        if self.k > self.n:
            raise ValueError(f"Clause size k ({self.k}) cannot be larger than the number of variables n ({self.n}).")
            
        # If m is not provided, use the ratio for hard 3-SAT problems
        self.m = m if m is not None else int(4.26 * n) # number of clauses
        
        self._idx = 0
        self.questions: List[str] = []
        self.formulas: List[List[List[int]]] = []
        self.assignments: List[Dict[int, bool]] = []

    async def generate_question(self, llm_client: Optional[LLMClient] = None) -> str:
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
        
        # Store for later verification
        self.formulas.append(clauses)
        self.assignments.append(assignment)
        
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
        self.questions.append(q)
        logger.debug(f"Generated SAT question #{self._idx}: n={self.n}, m={self.m}, planted={assignment}")
        self._idx += 1
        return q

    async def verify(
        self,
        question: str,
        response: str,
        llm_client: Optional[LLMClient] = None
    ) -> Dict[str, Any]:
        # Locate the corresponding formula & assignment
        try:
            idx = self.questions.index(question)
        except ValueError:
            logger.debug(f"Unknown question: {question}")
            return {"correct": False, "expected": None, "extraction": None}

        clauses = self.formulas[idx]
        expected = self.assignments[idx]

        # Guard against None responses that might slip through
        if response is None:
            return {"correct": False, "expected": expected, "extraction": None, "reason": "No response from model"}

        resp = response.strip()
        # If user claims UNSAT, that's always incorrect (we planted a solution)
        if re.search(r"\\bunsat\\b", resp, re.IGNORECASE):
            extraction = None
            correct = False
        else:
            # Extract assignments of the form x<number>=True/False or x<number>=1/0
            matches = re.findall(r"x(\d+)\s*=\s*(True|False|true|false|1|0)", resp)
            extraction: Dict[int, bool] = {}
            for var_str, val_str in matches:
                var_index = int(var_str)
                val = val_str.lower() in ("true", "1")
                extraction[var_index] = val

            # Must assign every variable
            if set(extraction.keys()) != set(expected.keys()):
                correct = False
            else:
                # Evaluate each clause under the extracted assignment
                def clause_satisfied(cl: List[int]) -> bool:
                    for lit in cl:
                        v = abs(lit)
                        val = extraction[v]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            return True
                    return False

                correct = all(clause_satisfied(cl) for cl in clauses)

        logger.debug(
            f"Verification: expected={expected}, extracted={extraction if 'extraction' in locals() else None}, "
            f"correct={correct}"
        )
        return {"correct": correct, "expected": expected, "extraction": extraction if 'extraction' in locals() else None}
