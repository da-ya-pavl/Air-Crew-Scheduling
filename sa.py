import math
import random
import numpy as np

from spp import SPPInstance, feasible_init


def _sa_single_run(inst, max_iter, x_init):
    # single SA run from a given starting solution
    n = inst.n
    A = inst.A
    costs = inst.costs
    col_rows = inst.col_rows
    row_cols = inst.row_cols
    lam_max = float(costs.max())

    T0 = float(costs.mean()) * 5.0
    T_final = 0.1
    alpha = (T_final / T0) ** (1.0 / max_iter)

    x = x_init.copy()
    cover = A @ x.astype(np.int32)
    cur_raw = float(costs @ x)
    cur_u = int(np.sum(cover != 1))

    best_x = x.copy() if cur_u == 0 else None
    best_cost = cur_raw if cur_u == 0 else math.inf

    T = T0
    _random = random.random
    _randrange = random.randrange
    _choice = random.choice

    for it in range(max_iter):
        lam = lam_max * (0.5 + 0.5 * it / max_iter)
        cur_pen = cur_raw + lam * cur_u

        if _random() < 0.5:
            # swap move: remove a selected col, add an unselected col sharing a row
            selected = np.where(x == 1)[0]
            if len(selected) == 0:
                T *= alpha
                continue
            j_out = selected[_randrange(len(selected))]
            row = _choice(col_rows[j_out])
            candidates = [c for c in row_cols[row] if x[c] == 0]
            if not candidates:
                T *= alpha
                continue
            j_in = _choice(candidates)

            col_out = A[:, j_out]
            col_in = A[:, j_in]
            new_cover = cover - col_out + col_in
            new_u = int(np.sum(new_cover != 1))
            new_raw = cur_raw - costs[j_out] + costs[j_in]
            new_pen = new_raw + lam * new_u

            delta = new_pen - cur_pen
            if delta <= 0 or _random() < math.exp(-delta / T):
                x[j_out] = 0
                x[j_in] = 1
                cover = new_cover
                cur_raw = new_raw
                cur_u = new_u
                if cur_u == 0 and cur_raw < best_cost:
                    best_x = x.copy()
                    best_cost = cur_raw
        else:
            # single bit flip
            j = _randrange(n)
            flip = 1 - 2 * int(x[j])
            col_j = A[:, j]

            new_cover = cover + flip * col_j
            new_u = int(np.sum(new_cover != 1))
            new_raw = cur_raw + flip * costs[j]
            new_pen = new_raw + lam * new_u

            delta = new_pen - cur_pen
            if delta <= 0 or _random() < math.exp(-delta / T):
                x[j] ^= 1
                cover = new_cover
                cur_raw = new_raw
                cur_u = new_u
                if cur_u == 0 and cur_raw < best_cost:
                    best_x = x.copy()
                    best_cost = cur_raw

        T *= alpha

    return best_x, best_cost


def simulated_annealing(
    inst: SPPInstance,
    max_iter: int = 500_000,
    n_restarts: int = 10,
) -> tuple[np.ndarray | None, float]:
    # multi-restart SA: split total iterations across restarts, keep global best
    iter_per_run = max_iter // n_restarts
    best_x = None
    best_cost = math.inf

    for _ in range(n_restarts):
        x_init = feasible_init(inst)
        run_x, run_cost = _sa_single_run(inst, iter_per_run, x_init)
        if run_cost < best_cost:
            best_cost = run_cost
            best_x = run_x

    return best_x, best_cost


if __name__ == "__main__":
    import os, time

    base = os.path.dirname(__file__)
    inst = SPPInstance(os.path.join(base, "sppnw41.txt"))

    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    best_x, best_cost = simulated_annealing(inst)
    elapsed = time.time() - t0

    if best_x is None:
        print("no feasible solution found")
    else:
        from spp import compute_coverage
        assert int(np.sum(compute_coverage(best_x, inst) != 1)) == 0
        optimal = 11307
        gap = (best_cost - optimal) / optimal * 100
        print(f"best cost : {best_cost:.0f}")
        print(f"optimal   : {optimal}")
        print(f"gap       : {gap:.2f}%")
        print(f"time      : {elapsed:.1f}s")
