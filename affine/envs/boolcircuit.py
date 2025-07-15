import random, re, affine as af

_ops = [("AND", lambda a,b: a and b), ("OR", lambda a,b: a or b)]

class BoolCircuit(af.BaseEnv):
    num_inputs: int

    def __init__(self, num_inputs=3):
        super().__init__(num_inputs=num_inputs)

    def _rand_expr(self, vars):
        if len(vars)==1:
            return vars[0], lambda inp: inp[vars[0]]
        op_name, op_fn = random.choice(_ops)
        left, f1 = self._rand_expr(vars[:len(vars)//2])
        right, f2 = self._rand_expr(vars[len(vars)//2:])
        expr=f"({left} {op_name} {right})"
        return expr, lambda inp, f1=f1,f2=f2,op_fn=op_fn: op_fn(f1(inp), f2(inp))

    async def generate(self):
        vars=[f"x{i}" for i in range(1,self.num_inputs+1)]
        expr,f=self._rand_expr(vars)
        inputs={v: random.choice([0,1]) for v in vars}
        out=int(f(inputs))
        prompt=f"Evaluate {expr} for inputs {inputs}. Return 0 or 1."
        return af.Challenge(env=self, prompt=prompt, extra={"out":out})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        m=re.search(r"[01]", response.response or "")
        got=int(m.group()) if m else -1
        ok=got==challenge.extra['out']
        return af.Evaluation(env=self, score=float(ok), extra={"expected":challenge.extra['out'], "got":got})

ENV_CLASS=BoolCircuit 