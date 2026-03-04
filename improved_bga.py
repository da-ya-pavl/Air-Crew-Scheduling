import random
import numpy as np

from spp import SPPInstance, compute_coverage, raw_cost, heuristic_improvement, pseudo_random_init
from bga import uniform_crossover


def stochastic_ranking(fitnesses, violations, pf=0.45):
    # bubble-sort ranking from runarsson & yao [2]
    n = len(fitnesses)
    idx = list(range(n))
    swapped = True
    while swapped:
        swapped = False
        for i in range(n - 1):
            a, b = idx[i], idx[i + 1]
            if (violations[a] == 0 and violations[b] == 0) or random.random() < pf:
                # compare by fitness
                if fitnesses[a] > fitnesses[b]:
                    idx[i], idx[i + 1] = idx[i + 1], idx[i]
                    swapped = True
            else:
                # compare by violation
                if violations[a] > violations[b]:
                    idx[i], idx[i + 1] = idx[i + 1], idx[i]
                    swapped = True
    return idx


def ranking_replace(pop, fitnesses, violations, child, child_cost, child_viol):
    # deterministic ranking replacement from [1]
    # find worst by (violation, fitness) lexicographic max
    worst_i = 0
    for i in range(1, len(pop)):
        if (violations[i], fitnesses[i]) > (violations[worst_i], fitnesses[worst_i]):
            worst_i = i
    if (child_viol, child_cost) < (violations[worst_i], fitnesses[worst_i]):
        pop[worst_i] = child
        fitnesses[worst_i] = child_cost
        violations[worst_i] = child_viol


def fixed_k_mutation(x, k):
    bits = random.sample(range(len(x)), k)
    for j in bits:
        x[j] ^= 1
    return x


def has_duplicates(pop):
    seen = set()
    for x in pop:
        b = x.tobytes()
        if b in seen:
            return True
        seen.add(b)
    return False


def is_duplicate_of_pop(child, pop):
    cb = child.tobytes()
    for x in pop:
        if x.tobytes() == cb:
            return True
    return False


def improved_bga(
    inst: SPPInstance,
    pop_size: int = 100,
    t_max: int = 100_000,
    ms: int = 3,
    ma: int = 5,
    pf: float = 0.45,
) -> tuple[np.ndarray, float]:
    # init population with pseudo-random init (all feasible)
    pop = [pseudo_random_init(inst) for _ in range(pop_size)]
    fitnesses = [raw_cost(pop[i], inst) for i in range(pop_size)]
    violations = [0] * pop_size  # all feasible after alg 2

    best_i = int(np.argmin(fitnesses))
    best_x = pop[best_i].copy()
    best_cost = fitnesses[best_i]

    children_count = 0
    while children_count < t_max:
        k = ma if has_duplicates(pop) else ms

        ranked = stochastic_ranking(fitnesses, violations, pf)
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

        dup = is_duplicate_of_pop(child, pop)
        ranking_replace(pop, fitnesses, violations, child, child_cost, child_viol)

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
