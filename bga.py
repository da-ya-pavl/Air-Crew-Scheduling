import math
import random
import numpy as np

from spp import SPPInstance, compute_coverage, raw_cost


def standard_bga(
    inst: SPPInstance,
    pop_size: int = 200,
    t_max: int = 100_000,
) -> tuple[np.ndarray | None, float]:
    n = inst.n
    lam = float(inst.costs.max())

    # penalized fitness: raw cost + lam * count(rows not covered exactly once)
    def penalized(x):
        cover = compute_coverage(x, inst)
        u = int(np.sum(cover != 1))
        return raw_cost(x, inst) + lam * u, u

    # sparse random init: density ~ 2*m/n
    p_one = min(2.0 * inst.m / n, 0.5)
    pop = [(np.random.random(n) < p_one).astype(np.int8) for _ in range(pop_size)]
    fitnesses = np.empty(pop_size)
    for i in range(pop_size):
        fitnesses[i], _ = penalized(pop[i])

    best_x: np.ndarray | None = None
    best_cost = math.inf

    for i in range(pop_size):
        _, u = penalized(pop[i])
        if u == 0:
            c = raw_cost(pop[i], inst)
            if c < best_cost:
                best_cost = c
                best_x = pop[i].copy()

    # generational loop with truncation replacement
    children_count = 0
    while children_count < t_max:
        offspring = []
        off_fits = []

        gen_size = min(pop_size, t_max - children_count)
        for _ in range(gen_size):
            p1 = tournament_select(pop, fitnesses)
            p2 = tournament_select(pop, fitnesses)
            child, _ = uniform_crossover(p1, p2)
            single_bit_mutation(child)

            child_fit, child_u = penalized(child)
            offspring.append(child)
            off_fits.append(child_fit)

            if child_u == 0:
                child_cost = raw_cost(child, inst)
                if child_cost < best_cost:
                    best_cost = child_cost
                    best_x = child.copy()

        children_count += gen_size

        # merge pop + offspring, keep best pop_size
        merged = pop + offspring
        merged_fits = np.concatenate([fitnesses, np.array(off_fits)])
        keep = np.argsort(merged_fits)[:pop_size]
        pop = [merged[i] for i in keep]
        fitnesses = merged_fits[keep].copy()

    return best_x, best_cost


def tournament_select(pop, fitnesses, k=2):
    idxs = random.sample(range(len(pop)), k)
    winner = min(idxs, key=lambda i: fitnesses[i])
    return pop[winner]


def uniform_crossover(p1, p2):
    mask = np.random.randint(0, 2, size=len(p1), dtype=np.int8)
    c1 = np.where(mask, p1, p2).astype(np.int8)
    c2 = np.where(mask, p2, p1).astype(np.int8)
    return c1, c2


def single_bit_mutation(x):
    j = random.randrange(len(x))
    x[j] ^= 1


if __name__ == "__main__":
    import os, time

    base = os.path.dirname(__file__)
    inst = SPPInstance(os.path.join(base, "sppnw41.txt"))

    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    best_x, best_cost = standard_bga(inst)
    elapsed = time.time() - t0

    if best_x is None:
        print("no feasible solution found")
    else:
        assert int(np.sum(compute_coverage(best_x, inst) != 1)) == 0
        optimal = 11307
        gap = (best_cost - optimal) / optimal * 100
        print(f"best cost : {best_cost:.0f}")
        print(f"optimal   : {optimal}")
        print(f"gap       : {gap:.2f}%")
        print(f"time      : {elapsed:.1f}s")
