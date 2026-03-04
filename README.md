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
| `spp.py` | Data loader, evaluation helpers, Algorithm 1 (heuristic improvement), Algorithm 2 (pseudo-random init) |
| `sa.py` | Simulated Annealing |
| `bga.py` | Standard Binary Genetic Algorithm |

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
best cost : 11430
optimal   : 11307
gap       : 1.09%
time      : 1.3s
```

Run Standard BGA on sppnw41 (single run):

```bash
python3 bga.py
```

```
best cost : 13053
optimal   : 11307
gap       : 15.44%
time      : 1.2s
```

