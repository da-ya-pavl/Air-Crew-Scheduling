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
    # algorithm 1 (chu & beasley p331): remove redundant cols, fix uncovered rows, repeat.
    # extended repair loop handles interlocked over-coverage using a banned-col set.
    x = x.copy()
    cover = inst.A @ x.astype(np.int32)  # shape (m,)

    def remove_redundant() -> bool:
        changed = False
        selected = np.where(x == 1)[0]
        order = selected[np.argsort(-inst.costs[selected])]
        for j in order:
            if x[j] == 0:
                continue
            if all(cover[r] >= 2 for r in inst.col_rows[j]):
                x[j] = 0
                for r in inst.col_rows[j]:
                    cover[r] -= 1
                changed = True
        return changed

    def fix_uncovered():
        # greedy global: pick col with min cost per newly-covered uncovered row
        while True:
            unc = np.where(cover == 0)[0]
            if len(unc) == 0:
                break
            uncovered_set = set(unc.tolist())
            cands: dict[int, float] = {}
            for i in uncovered_set:
                for j in inst.row_cols[i]:
                    if x[j] == 0 and j not in cands:
                        newly = sum(1 for r in inst.col_rows[j] if r in uncovered_set)
                        if newly > 0:
                            cands[j] = inst.costs[j] / newly
            if not cands:
                break
            j_star = min(cands, key=cands.__getitem__)
            x[j_star] = 1
            for r in inst.col_rows[j_star]:
                cover[r] += 1

    remove_redundant()
    fix_uncovered()
    remove_redundant()

    # repair loop: for any over-covered row, drop the most expensive covering col
    # and re-run fix_uncovered with a banned set to avoid cycling
    banned: set[int] = set()
    for _iter in range(inst.n * 2):
        over = np.where(cover > 1)[0]
        if len(over) == 0:
            break

        # easiest row to fix: fewest selected cols covering it
        r_target = min(over.tolist(), key=lambda r: sum(1 for j in inst.row_cols[r] if x[j] == 1))
        cols_covering = [j for j in inst.row_cols[r_target] if x[j] == 1]

        j_drop = max(cols_covering, key=lambda j: inst.costs[j])
        x[j_drop] = 0
        banned.add(j_drop)
        for r in inst.col_rows[j_drop]:
            cover[r] -= 1

        while True:
            unc = np.where(cover == 0)[0]
            if len(unc) == 0:
                break
            uncovered_set = set(unc.tolist())
            cands: dict[int, float] = {}
            for i in uncovered_set:
                for j in inst.row_cols[i]:
                    if x[j] == 0 and j not in cands and j not in banned:
                        newly = sum(1 for r in inst.col_rows[j] if r in uncovered_set)
                        if newly > 0:
                            cands[j] = inst.costs[j] / newly
            if not cands:
                # relax ban if stuck
                banned.clear()
                for i in uncovered_set:
                    for j in inst.row_cols[i]:
                        if x[j] == 0 and j not in cands:
                            newly = sum(1 for r in inst.col_rows[j] if r in uncovered_set)
                            if newly > 0:
                                cands[j] = inst.costs[j] / newly
                if not cands:
                    break
            j_star = min(cands, key=cands.__getitem__)
            x[j_star] = 1
            for r in inst.col_rows[j_star]:
                cover[r] += 1

        remove_redundant()

    return x


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
    assert np.all(compute_coverage(x_fixed, inst) == 1)
    print(f"heuristic_improvement(all-ones): pass  cost={raw_cost(x_fixed, inst):.0f}")

    for _ in range(100):
        x = pseudo_random_init(inst)
        assert np.all(compute_coverage(x, inst) == 1)
    print("pseudo_random_init (100 runs): pass")
