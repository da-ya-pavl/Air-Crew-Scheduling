# Air Crew Scheduling — SPP Metaheuristics

Three metaheuristic algorithms for the Set Partitioning Problem (SPP), applied to airline crew scheduling instances from OR-Library.

## Requirements

```
pip install numpy
```

Python 3.10+.

## Files

| File | Description |
|------|-------------|
| `spp.py` | Data loader, evaluation helpers, Algorithm 1 (heuristic improvement), Algorithm 2 (pseudo-random init), feasible init wrapper |
| `sa.py` | Simulated Annealing |
| `bga.py` | Standard Binary Genetic Algorithm |
| `improved_bga.py` | Improved BGA (stochastic ranking, ranking replacement, adaptive mutation) |
| `run_experiments.py` | Runs 30 trials × 3 algorithms × 3 problems, prints results table, saves `results.json` |
| `report.md` | Assignment report: algorithm descriptions, results, discussion |

## Running

Verify data loading and shared utilities:

```bash
python3 spp.py
```

Run Simulated Annealing on sppnw41 (single run):

```bash
python3 sa.py
```

```
best cost : 13209
optimal   : 11307
gap       : 16.82%
time      : 2.3s
```

Run Standard BGA on sppnw41 (single run):

```bash
python3 bga.py
```

```
best cost : 14622
optimal   : 11307
gap       : 29.32%
time      : 1.2s
```

Run Improved BGA on sppnw41 (single run):

```bash
python3 improved_bga.py
```

```
best cost : 11307
optimal   : 11307
gap       : 0.00%
time      : 5.8s
```

Run all 30-trial experiments (3 algorithms × 3 problems):

```bash
python3 run_experiments.py
```

Results are printed to stdout and saved to `results.json`.

