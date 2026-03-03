import math
import random
import numpy as np

from spp import SPPInstance


def simulated_annealing(
    inst: SPPInstance,
    max_iter: int = 500_000,
) -> tuple[np.ndarray | None, float]:
    # lambda grows linearly from 0.1*lam_max to lam_max so the search can
    # explore infeasible regions early and converge to feasibility late.
    # cover is maintained incrementally to avoid a full matrix multiply per step.
    n = inst.n
    A = inst.A
    costs = inst.costs
    lam_max = float(costs.max())

    T0 = float(costs.mean()) * 2.0
    T_final = 0.1
    alpha = (T_final / T0) ** (1.0 / max_iter)

    x = np.random.randint(0, 2, size=n, dtype=np.int8)
    cover = A @ x.astype(np.int32)
    cur_raw = float(costs @ x)
    cur_u = int(np.sum(cover != 1))

    best_x: np.ndarray | None = None
    best_cost = math.inf

    if cur_u == 0:
        best_x = x.copy()
        best_cost = cur_raw

    T = T0

    for it in range(max_iter):
        lam = lam_max * (0.1 + 0.9 * it / max_iter)
        cur_pen = cur_raw + lam * cur_u

        j = random.randrange(n)
        flip = 1 - 2 * int(x[j])
        col_j = A[:, j]

        new_cover = cover + flip * col_j
        new_u = int(np.sum(new_cover != 1))
        new_raw = cur_raw + flip * costs[j]
        new_pen = new_raw + lam * new_u

        delta = new_pen - cur_pen

        if delta <= 0 or random.random() < math.exp(-delta / T):
            x[j] ^= 1
            cover = new_cover
            cur_raw = new_raw
            cur_u = new_u
            if cur_u == 0 and cur_raw < best_cost:
                best_x = x.copy()
                best_cost = cur_raw

        T *= alpha

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
