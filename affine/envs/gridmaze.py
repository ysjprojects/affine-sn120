import random, re, affine as af

# minimal BFS helper ---------------------------------------------------------

def _bfs(grid, start, goal):
    H, W = len(grid), len(grid[0])
    q = [start]
    prev = {start: None}
    while q:
        r, c = q.pop(0)
        if (r, c) == goal:
            path = []
            while (r, c) is not None:
                path.append((r, c))
                r, c = prev[(r, c)] if (r, c) in prev else (None, None)
            return list(reversed(path))
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and grid[nr][nc]==0 and (nr,nc) not in prev:
                prev[(nr,nc)] = (r,c)
                q.append((nr,nc))
    return []


def _render(grid):
    return "\n".join("".join("#" if c else "." for c in row) for row in grid)


def _parse(resp):
    return [(int(a), int(b)) for a,b in re.findall(r"(\d+)\s*,\s*(\d+)", resp)]


class GridMaze(af.BaseEnv):
    width: int
    height: int
    obstacle_prob: float

    def __init__(self, width=5, height=5, obstacle_prob=0.2):
        super().__init__(width=width, height=height, obstacle_prob=obstacle_prob)

    async def generate(self):
        grid = [[0 if random.random()>self.obstacle_prob else 1 for _ in range(self.width)] for _ in range(self.height)]
        start, goal = (0,0), (self.height-1, self.width-1)
        grid[0][0]=grid[-1][-1]=0
        path = _bfs(grid, start, goal)
        prompt = f"Grid (# obstacle, . free):\n{_render(grid)}\nProvide shortest path as list of 'r,c' pairs."
        return af.Challenge(env=self, prompt=prompt, extra={"grid":grid, "path":path})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        proposed = _parse(response.response or "")
        ok = proposed==challenge.extra["path"]
        return af.Evaluation(env=self, score=float(ok), extra={"expected":challenge.extra["path"], "got":proposed})

ENV_CLASS = GridMaze 