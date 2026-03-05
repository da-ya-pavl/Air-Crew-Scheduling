import numpy as np
import random


# rows are 1-indexed in the file, stored 0-indexed here
class SPPInstance:
    def __init__(self, filepath: str):
        with open(filepath) as f:
            tokens = f.read().split()

        idx = 0
        self.m = int(tokens[idx]); idx += 1
        self.n = int(tokens[idx]); idx += 1

        self.costs = np.empty(self.n, dtype=np.float64)
        self.col_rows: list[list[int]] = []
        self.row_cols: list[list[int]] = [[] for _ in range(self.m)]

        for j in range(self.n):
            cost = float(tokens[idx]); idx += 1
            num_covered = int(tokens[idx]); idx += 1
            rows = []
            for _ in range(num_covered):
                row = int(tokens[idx]) - 1
                idx += 1
                rows.append(row)
            self.costs[j] = cost
            self.col_rows.append(rows)
            for r in rows:
                self.row_cols[r].append(j)

        self.A = np.zeros((self.m, self.n), dtype=np.bool_)
        for j, rows in enumerate(self.col_rows):
            self.A[rows, j] = True



def compute_coverage(x: np.ndarray, inst: SPPInstance) -> np.ndarray:
    return inst.A @ x.astype(np.int32)


def raw_cost(x: np.ndarray, inst: SPPInstance) -> float:
    return float(inst.costs @ x)


def evaluate(x: np.ndarray, inst: SPPInstance, lam: float) -> tuple[float, int]:
    cover = compute_coverage(x, inst)
    u = int(np.sum(cover != 1))
    return raw_cost(x, inst) + lam * u, u


def heuristic_improvement(x: np.ndarray, inst: SPPInstance) -> np.ndarray:
    # algorithm 1 (chu & beasley p331): drop phase then add phase
    col_rows = inst.col_rows
    row_cols = inst.row_cols
    costs = inst.costs
    m = inst.m
    n = inst.n

    xl = x.tolist()
    cover = [0] * m
    for j in range(n):
        if xl[j]:
            for r in col_rows[j]:
                cover[r] += 1

    # drop phase: random order, drop if ANY row covered by j has cover >= 2
    selected = [j for j in range(n) if xl[j]]
    random.shuffle(selected)
    for j in selected:
        if not xl[j]:
            continue
        if any(cover[r] >= 2 for r in col_rows[j]):
            xl[j] = 0
            for r in col_rows[j]:
                cover[r] -= 1

    # add phase: visit uncovered rows in random order, each row once
    # only add column j if beta_j ⊆ U (all rows of j are uncovered)
    uncovered = set(i for i in range(m) if cover[i] == 0)
    visit = list(uncovered)
    random.shuffle(visit)
    for i in visit:
        if cover[i] != 0:
            continue
        best_j = -1
        best_score = float('inf')
        for j in row_cols[i]:
            if xl[j]:
                continue
            # beta_j ⊆ U: all rows of j must be uncovered
            if all(cover[r] == 0 for r in col_rows[j]):
                score = costs[j] / len(col_rows[j])
                if score < best_score:
                    best_score = score
                    best_j = j
        if best_j != -1:
            xl[best_j] = 1
            for r in col_rows[best_j]:
                cover[r] += 1
                uncovered.discard(r)

    return np.array(xl, dtype=np.int8)


def pseudo_random_init(inst: SPPInstance) -> np.ndarray:
    # algorithm 2: randomly pick cols to cover uncovered rows, then repair
    x = np.zeros(inst.n, dtype=np.int8)
    uncovered = set(range(inst.m))

    while uncovered:
        i = random.choice(list(uncovered))
        j = random.choice(inst.row_cols[i])
        x[j] = 1
        for r in inst.col_rows[j]:
            uncovered.discard(r)

    return heuristic_improvement(x, inst)


def feasible_init(inst: SPPInstance) -> np.ndarray:
    # retry pseudo_random_init until feasible (all rows covered exactly once)
    while True:
        x = pseudo_random_init(inst)
        cover = compute_coverage(x, inst)
        if np.all(cover == 1):
            return x


if __name__ == "__main__":
    import os

    base = os.path.dirname(__file__)
    inst = SPPInstance(os.path.join(base, "sppnw41.txt"))

    assert inst.m == 17
    assert inst.n == 197
    assert inst.costs[0] == 2259.0
    assert inst.col_rows[0] == [0, 2, 3, 7, 9]
    print("parser assertions: pass")

    x_ones = np.ones(inst.n, dtype=np.int8)
    x_fixed = heuristic_improvement(x_ones, inst)
    cover = compute_coverage(x_fixed, inst)
    assert np.all(cover <= 1), "over-coverage after heuristic_improvement"
    print(f"heuristic_improvement(all-ones): pass  cost={raw_cost(x_fixed, inst):.0f}  uncovered={int(np.sum(cover == 0))}")

    feasible_count = 0
    for _ in range(100):
        x = pseudo_random_init(inst)
        c = compute_coverage(x, inst)
        assert np.all(c <= 1), "over-coverage after pseudo_random_init"
        if np.all(c == 1):
            feasible_count += 1
    print(f"pseudo_random_init (100 runs): pass  feasible={feasible_count}/100")

    # feasible_init: must be 100% feasible on all problems
    problems = ["sppnw41.txt", "sppnw42.txt", "sppnw43.txt"]
    for pfile in problems:
        ppath = os.path.join(base, pfile)
        if not os.path.exists(ppath):
            continue
        pinst = SPPInstance(ppath)
        for _ in range(100):
            x = feasible_init(pinst)
            c = compute_coverage(x, pinst)
            assert np.all(c == 1), f"infeasible after feasible_init on {pfile}"
        print(f"feasible_init on {pfile} (100 runs): pass  100% feasible")
