import random, re, affine as af
from collections import deque

# reuse BFS path search -------------------------------------------------------

def _bfs(grid,start,goal):
    N=len(grid)
    q=deque([start]);prev={start:None}
    while q:
        r,c=q.popleft()
        if (r,c)==goal:
            path=[]
            while (r,c) is not None:
                path.append((r,c));r,c=prev[(r,c)] if (r,c) in prev else (None,None)
            return list(reversed(path))
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<N and 0<=nc<N and grid[nr][nc]==0 and (nr,nc) not in prev:
                prev[(nr,nc)]=(r,c);q.append((nr,nc))
    return []


def _render(g):
    return "\n".join("".join("H" if c else "." for c in row) for row in g)


def _parse(resp):
    return [(int(a),int(b)) for a,b in re.findall(r"(\d+)\s*,\s*(\d+)", resp)]


class FrozenLakeLogic(af.BaseEnv):
    size:int
    hole_prob:float
    def __init__(self,size=4,hole_prob=0.2):
        super().__init__(size=size,hole_prob=hole_prob)
    async def generate(self):
        grid=[[1 if random.random()<self.hole_prob else 0 for _ in range(self.size)] for _ in range(self.size)]
        start,goal=(0,0),(self.size-1,self.size-1)
        grid[0][0]=grid[-1][-1]=0
        path=_bfs(grid,start,goal)
        prompt=f"Frozen Lake (H hole, . ice):\n{_render(grid)}\nGive shortest safe path as list of coords."
        return af.Challenge(env=self,prompt=prompt,extra={"grid":grid,"path":path})
    async def evaluate(self,challenge:af.Challenge,response:af.Response):
        got=_parse(response.response or "")
        ok=got==challenge.extra['path']
        return af.Evaluation(env=self,score=float(ok),extra={"expected":challenge.extra['path'],"got":got})

ENV_CLASS=FrozenLakeLogic 