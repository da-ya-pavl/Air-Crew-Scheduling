import random
import numpy as np

from spp import SPPInstance, compute_coverage, raw_cost, heuristic_improvement, pseudo_random_init
from bga import uniform_crossover


def stochastic_ranking(fitnesses, violations, pf=0.45):
    # bubble-sort ranking from runarsson & yao [2], fig 1
    # N sweeps with early termination on no-swap sweep
    n = len(fitnesses)
    idx = list(range(n))
    f = fitnesses
    v = violations
    _random = random.random  # local binding for speed
    all_feasible = all(vi == 0 for vi in v)
    if all_feasible:
        for _ in range(n):
            swapped = False
            for i in range(n - 1):
                if _random() < pf and f[idx[i]] > f[idx[i + 1]]:
                    idx[i], idx[i + 1] = idx[i + 1], idx[i]
                    swapped = True
            if not swapped:
                break
    else:
        for _ in range(n):
            swapped = False
            for i in range(n - 1):
                a, b = idx[i], idx[i + 1]
                if (v[a] == 0 and v[b] == 0) or _random() < pf:
                    if f[a] > f[b]:
                        idx[i], idx[i + 1] = b, a
                        swapped = True
                else:
                    if v[a] > v[b]:
                        idx[i], idx[i + 1] = b, a
                        swapped = True
            if not swapped:
                break
    return idx


def ranking_replace(pop, fitnesses, violations, child, child_cost, child_viol, pop_hash=None):
    # deterministic ranking replacement from [1]
    # find worst by (violation, fitness) lexicographic max
    worst_i = 0
    for i in range(1, len(pop)):
        if (violations[i], fitnesses[i]) > (violations[worst_i], fitnesses[worst_i]):
            worst_i = i
    if (child_viol, child_cost) < (violations[worst_i], fitnesses[worst_i]):
        if pop_hash is not None:
            pop_hash.replace(pop[worst_i], child)
        pop[worst_i] = child
        fitnesses[worst_i] = child_cost
        violations[worst_i] = child_viol


def fixed_k_mutation(x, k):
    bits = random.sample(range(len(x)), k)
    for j in bits:
        x[j] ^= 1
    return x


class PopHashSet:
    # tracks population member byte-strings incrementally
    def __init__(self, pop):
        self._counts: dict[bytes, int] = {}
        self._n_dups = 0
        for x in pop:
            b = x.tobytes()
            c = self._counts.get(b, 0)
            if c >= 1:
                self._n_dups += 1
            self._counts[b] = c + 1

    def has_duplicates(self) -> bool:
        return self._n_dups > 0

    def contains(self, child_bytes: bytes) -> bool:
        return child_bytes in self._counts

    def replace(self, old: np.ndarray, new: np.ndarray):
        # remove old
        ob = old.tobytes()
        c = self._counts[ob]
        if c > 1:
            self._counts[ob] = c - 1
            self._n_dups -= 1  # one fewer duplicate pair
        else:
            del self._counts[ob]
        # add new
        nb = new.tobytes()
        c = self._counts.get(nb, 0)
        if c >= 1:
            self._n_dups += 1
        self._counts[nb] = c + 1


def improved_bga(
    inst: SPPInstance,
    pop_size: int = 100,
    t_max: int = 100_000,
    ms: int = 3,
    ma: int = 5,
    pf: float = 0.45,
) -> tuple[np.ndarray, float]:
    # init population with pseudo-random init (may have violations after alg 1 fix)
    pop = [pseudo_random_init(inst) for _ in range(pop_size)]
    fitnesses = [raw_cost(pop[i], inst) for i in range(pop_size)]
    violations = [int(np.sum(compute_coverage(pop[i], inst) != 1)) for i in range(pop_size)]

    best_cost = float('inf')
    best_x = None
    for i in range(pop_size):
        if violations[i] == 0 and fitnesses[i] < best_cost:
            best_cost = fitnesses[i]
            best_x = pop[i].copy()

    pop_hash = PopHashSet(pop)
    children_count = 0
    rank_interval = 10
    ranked = None
    iter_count = 0
    while children_count < t_max:
        k = ma if pop_hash.has_duplicates() else ms

        if iter_count % rank_interval == 0 or ranked is None:
            ranked = stochastic_ranking(fitnesses, violations, pf)
        iter_count += 1
        half = pop_size // 2
        p1 = pop[ranked[random.randrange(half)]]
        p2 = pop[ranked[random.randrange(half)]]

        child, _ = uniform_crossover(p1, p2)
        child = fixed_k_mutation(child, k)
        child = heuristic_improvement(child, inst)

        child_cost = raw_cost(child, inst)
        cover = compute_coverage(child, inst)
        child_viol = int(np.sum(cover != 1))

        if child_viol == 0 and child_cost < best_cost:
            best_cost = child_cost
            best_x = child.copy()

        dup = pop_hash.contains(child.tobytes())
        ranking_replace(pop, fitnesses, violations, child, child_cost, child_viol, pop_hash)

        if not dup:
            children_count += 1

    return best_x, best_cost


if __name__ == "__main__":
    import os, time

    base = os.path.dirname(__file__)
    inst = SPPInstance(os.path.join(base, "sppnw41.txt"))

    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    best_x, best_cost = improved_bga(inst)
    elapsed = time.time() - t0

    assert int(np.sum(compute_coverage(best_x, inst) != 1)) == 0
    optimal = 11307
    gap = (best_cost - optimal) / optimal * 100
    print(f"best cost : {best_cost:.0f}")
    print(f"optimal   : {optimal}")
    print(f"gap       : {gap:.2f}%")
    print(f"time      : {elapsed:.1f}s")
