import random, re, affine as af

# 4x4 Sudoku helper ---------------------------------------------------------

_GRID_ALL = [[(i*2 + i//2 + j) % 4 + 1 for j in range(4)] for i in range(4)]
# _GRID_ALL is a valid 4x4 sudoku solution.

_DEF_MASKS = [
    [(0,0),(0,3),(1,1),(2,2),(3,0),(3,3)],
    [(0,2),(1,0),(1,3),(2,1),(3,2)],
]

def _mask_grid(grid):
    g = [row[:] for row in grid]
    rm = random.choice(_DEF_MASKS)
    for r,c in rm:
        g[r][c] = 0
    return g


def _render(grid):
    rows = [" ".join(str(v or '.') for v in row) for row in grid]
    return "\n".join(rows)


def _parse(resp):
    nums = list(map(int, re.findall(r"\d", resp)))
    return [nums[i*4:(i+1)*4] for i in range(4)] if len(nums)==16 else None


class MiniSudoku(af.BaseEnv):
    async def generate(self):
        grid = [row[:] for row in _GRID_ALL]
        random.shuffle(grid)
        for r in grid:
            random.shuffle(r)
        puzzle = _mask_grid(grid)
        prompt = (
            "Fill the 4x4 Sudoku so each row, column and 2x2 block contains 1-4 exactly once.\n" +
            _render(puzzle)
        )
        return af.Challenge(env=self, prompt=prompt, extra={"solution": grid})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        filled = _parse(response.response or "")
        ok = filled == challenge.extra["solution"]
        return af.Evaluation(env=self, score=float(ok), extra={"expected": challenge.extra["solution"], "got": filled})

ENV_CLASS = MiniSudoku 