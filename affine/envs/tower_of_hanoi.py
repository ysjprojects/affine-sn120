import random, re, affine as af

class TowerOfHanoi(af.BaseEnv):
    n_disks:int
    def __init__(self,n_disks=4):
        super().__init__(n_disks=n_disks)
    async def generate(self):
        moves=2**self.n_disks-1
        prompt=f"In Tower of Hanoi with {self.n_disks} disks, what is minimal number of moves?"
        return af.Challenge(env=self,prompt=prompt,extra={"moves":moves})
    async def evaluate(self,challenge:af.Challenge,response:af.Response):
        m=re.search(r"\d+", response.response or "")
        got=int(m.group()) if m else None
        ok=got==challenge.extra['moves']
        return af.Evaluation(env=self,score=float(ok),extra={"expected":challenge.extra['moves'],"got":got})

ENV_CLASS=TowerOfHanoi 