import random, re, affine as af
from heapq import heappush, heappop
from itertools import permutations

class GraphSP(af.BaseEnv):
    n: int
    edge_prob: float

    def __init__(self, n=8, edge_prob=0.3):
        super().__init__(n=n, edge_prob=edge_prob)

    def _dijkstra(self, edges, src, dst):
        adj = {i: [] for i in range(self.n)}
        for u,v,w in edges:
            adj[u].append((v,w))
        INF=10**9
        dist=[INF]*self.n
        dist[src]=0
        pq=[(0,src)]
        while pq:
            d,u=heappop(pq)
            if d!=dist[u]:
                continue
            if u==dst:
                return d
            for v,w in adj[u]:
                nd=d+w
                if nd<dist[v]:
                    dist[v]=nd
                    heappush(pq,(nd,v))
        return INF

    async def generate(self):
        nodes=list(range(self.n))
        edges=[(i,j,random.randint(1,10)) for i,j in permutations(nodes,2) if random.random()<self.edge_prob]
        src,dst=random.sample(nodes,2)
        dist=self._dijkstra(edges,src,dst)
        prompt=f"Edges (u,v,w): {edges}. From {src} to {dst}: min total weight?"
        return af.Challenge(env=self, prompt=prompt, extra={"edges":edges,"src":src,"dst":dst,"dist":dist})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        m=re.search(r"\d+", response.response or "")
        got=int(m.group()) if m else None
        ok=got==challenge.extra['dist']
        return af.Evaluation(env=self, score=float(ok), extra={"expected":challenge.extra['dist'], "got":got})

ENV_CLASS=GraphSP 