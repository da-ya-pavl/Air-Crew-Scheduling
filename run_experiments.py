import os
import json
import math
import time
import random
import statistics
import numpy as np

from spp import SPPInstance
from sa import simulated_annealing
from bga import standard_bga
from improved_bga import improved_bga

PROBLEMS = [("sppnw41.txt", 11307), ("sppnw42.txt", 7656), ("sppnw43.txt", 8904)]
ALGORITHMS = [("SA", simulated_annealing), ("Standard_BGA", standard_bga), ("Improved_BGA", improved_bga)]
NUM_RUNS = 30


def run_experiments():
    base = os.path.dirname(__file__)
    all_results = {}

    for problem_file, optimal in PROBLEMS:
        inst = SPPInstance(os.path.join(base, problem_file))
        print(f"\nProblem: {problem_file}  (optimal={optimal})")
        print(f"  {'Algorithm':<14} {'Mean':>10} {'Std':>10} {'Best':>10} {'Gap(%)':>8} {'Time(s)':>8}")
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

        for algo_name, algo_fn in ALGORITHMS:
            costs = []
            total_time = 0.0

            for run in range(NUM_RUNS):
                random.seed(42 + run)
                np.random.seed(42 + run)

                t0 = time.time()
                _, cost = algo_fn(inst)
                elapsed = time.time() - t0
                total_time += elapsed

                # sa can return inf if no feasible solution found
                if cost is None or math.isinf(cost):
                    costs.append(math.inf)
                else:
                    costs.append(cost)

                print(f"    {algo_name} run {run+1:2d}/30: {cost:>10.0f}  ({elapsed:.1f}s)", flush=True)

            finite = [c for c in costs if not math.isinf(c)]
            if finite:
                mean = statistics.mean(finite)
                std = statistics.stdev(finite) if len(finite) > 1 else 0.0
                best = min(finite)
                gap = (best - optimal) / optimal * 100
            else:
                mean = std = best = gap = math.inf

            avg_time = total_time / NUM_RUNS
            print(f"  {algo_name:<14} {mean:>10.1f} {std:>10.1f} {best:>10.0f} {gap:>7.2f}% {avg_time:>7.1f}s")

            key = f"{algo_name}_{problem_file}"
            all_results[key] = {
                "algorithm": algo_name,
                "problem": problem_file,
                "optimal": optimal,
                "costs": [c if not math.isinf(c) else None for c in costs],
                "mean": mean if not math.isinf(mean) else None,
                "std": std if not math.isinf(std) else None,
                "best": best if not math.isinf(best) else None,
                "gap_pct": gap if not math.isinf(gap) else None,
                "avg_time_s": round(avg_time, 2),
                "feasible_runs": len(finite),
            }

    results_path = os.path.join(base, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_experiments()
