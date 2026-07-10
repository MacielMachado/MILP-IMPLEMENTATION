# Branch & Bound MILP Solver — Execution Guide

This repository contains a didactic implementation of a **Branch & Bound (B&B) solver for Mixed-Integer Linear Programming (MILP)** problems. Gurobi is used **exclusively as an LP relaxation solver** at each node — all B&B logic, presolving, primal heuristics, and cutting planes are implemented from scratch.

---

## 📁 Repository Structure

```
.
├── branch_and_bound.py      # Core B&B engine (BranchAndBound class, Node, etc.)
├── branching.py             # Branch variable rules (FirstFractional, StrongBranching, PseudoCost, ...)
├── node_selection.py        # Node selection strategies (DepthFirst, BestFirst, BreadthFirst)
├── presolve.py              # Presolver and presolve rules
├── primal_heuristics.py     # Primal heuristics (Diving, FeasibilityPump, RINS, RENS, LNS, ...)
├── cuts.py                  # Cutting planes (Gomory, MIR, Cover, LiftingExtending, ...)
├── milp.py                  # Single-instance runner (verbose terminal output, no report files)
├── execution_examples.py    # Multi-config experiment runner (generates CSV + LaTeX reports)
└── problems/                # Directory with .mps instance files
```

---

## ⚙️ Requirements

- Python 3.10+
- [Gurobi](https://www.gurobi.com/) with a valid license (`gurobipy`)
- NumPy

Install Python dependencies:

```bash
pip install gurobipy numpy
```

> **Note:** A valid Gurobi license is required. Academic licenses are available for free at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## 🚀 How to Run

There are **two ways** to run the solver, depending on your goal:

---

### 1️⃣ Single Instance — `milp.py` (Quick, Terminal Output)

Use this file when you want to **test a single instance interactively**. Results are **not saved to any report folder** — instead, the full B&B tree traversal is printed live to the terminal via `verbose=True`.

Simply edit the `model_path` and the solver configuration directly in the file, then run:

```bash
python milp.py
```

**Example output in the terminal:**

```
============================================================
  Branch & Bound — estratégia: BestFirst
  Direção: ceil_first | Sentido: MIN
  Branching: StrongBranchingRule
  Variáveis: 42 (30 inteiras)
  Heurísticas: ['RoundingHeuristic', 'FeasibilityPump', ...]
============================================================
  Nó    1 (prof= 0) | incumbent=inf    | bound=134.2100 | branch em x3=2.714
  Nó    2 (prof= 1) | incumbent=140.00 | bound=136.5000 | branch em x7=1.333
  ...
  Valor total: 138.0000
```

After solving, stats are exported to a JSON file:

```python
bb.export_stats("results_stress_17.json")
```

**To change the instance or configuration**, edit `milp.py` directly. For example, to switch the branching rule:

```python
# In milp.py
model_path = "problems/your_instance.mps"

bb = BranchAndBound(
    model_path=model_path,
    strategy=BestFirst(),
    direction=BranchDirection.CEIL_FIRST,
    branch_rule=StrongBranchingRule(max_candidates=8, score_mode="product"),
    presolver=Presolver([
        BoundTightening(),
        RowFeasibilityAndRedundancy(),
        SubstituteFixedVariables(),
        ImplicitFreeVariableDetection(),
        SingletonRowReduction(),
        EuclideanReduction(),
        CoefficientStrengthening(),
        ChvatalGomoryStrengthening(),
        ParallelRows(),
    ], verbose=True),
    heuristics=[
        RoundingHeuristic(only_root=False,  probability=1.0),
        FixAndPropagate(only_root=False,    probability=1.0),
        RENS(only_root=True,               probability=1.0),
        RINS(only_root=False,              probability=0.3),
        FractionalDiving(only_root=False,  probability=0.3),
        FeasibilityPump(only_root=False,   probability=0.3),
        FeasibilityJump(only_root=False,   probability=0.3),
    ],
    cutting_planes=[
        LiftingExtending(only_root=True, max_cuts=20),
    ],
    verbose=True,   # <-- prints the full tree traversal to terminal
)

result = bb.solve()

if result["status"] == "optimal":
    print(f"  Valor total: {result['obj_value']:.4f}")

bb.export_stats("results_stress_17.json")
```

---

### 2️⃣ Full Experiment Suite — `execution_examples.py` (All Configs, Reports Generated)

Use this file to **run all 10 B&B configurations** across all `.mps` instances in the problems folder. Results are automatically saved as **CSV and LaTeX tables** organized by configuration.

```bash
python execution_examples.py
```

This will execute the following pipeline:

#### Stage 1 — Run all instances for each configuration
Each configuration is run against every `.mps` file found in `problems_nl/`. JSON stats are saved per configuration in separate subfolders:

```
results_json/
  cfg0_baseline/        instance_a.json, instance_b.json, ...
  cfg1_presolve/        ...
  cfg2_depth_first/     ...
  cfg3_best_first/      ...
  cfg4_strong_branch/   ...
  cfg5_pseudocost/      ...
  cfg6_cuts/            ...
  cfg7_heuristics/      ...
  cfg8_nonlinear/       ...
  cfg9_complete/        ...
```

> ⚡ **Instances that already have a JSON file are automatically skipped**, making it safe to interrupt and resume the experiment.

#### Stage 2 — Extract metrics and generate per-config tables
For each configuration, the following tables are generated in both `.csv` and `.tex` formats:

| File | Content |
|---|---|
| `table_1_resultado_relaxacao` | Final status, objective value, LP relaxation quality, root gaps |
| `table_2_arvore` | B&B tree size: nodes explored/pruned, max depth, open nodes |
| `table_3_tempos` | Runtime breakdown: total, LP, heuristics, cuts |
| `table_4_heuristicas_cortes` | Incumbents found, heuristic calls/improvements, cuts added |
| `table_p1_presolve_summary` | Presolve summary: bounds tightened, rows/vars removed, runtime |
| `table_p2_presolve_rules` | Per-rule presolve activity and impact |

Tables are saved under:

```
results_tables/
  cfg0_baseline/        table_1_*.csv, table_2_*.csv, ...
  cfg1_presolve/        ...
  ...
  comparison_all_configs.csv   ← cross-config comparison table

results_latex/
  cfg0_baseline/        table_1_*.tex, table_2_*.tex, ...
  ...
  comparison_all_configs.tex   ← cross-config comparison in LaTeX
```

#### Stage 3 — Generate cross-configuration comparison table
A single flat table (`comparison_all_configs`) is generated with one row per `(config, instance)` pair, showing key metrics side by side for easy comparison:

| Column | Description |
|---|---|
| `config` | Configuration name (e.g. `cfg0_baseline`) |
| `instance` | Instance name |
| `status` | Solve status (`optimal`, `infeasible`, ...) |
| `obj_value` | Best objective value found |
| `runtime_total` | Total wall-clock time (seconds) |
| `nodes_explored` | Total B&B nodes explored |
| `total_cuts_added` | Total cutting planes added |
| `total_heur_solutions` | Total feasible solutions found by heuristics |
| `gap_root` | Relative gap closed at the root node |

---

## 🔧 The 10 B&B Configurations

| ID | Name | Strategy | Branch Rule | Presolver | Heuristics | Cuts |
|---|---|---|---|---|---|---|
| 0 | `cfg0_baseline` | BreadthFirst | FirstFractional | ✗ | ✗ | ✗ |
| 1 | `cfg1_presolve` | BreadthFirst | FirstFractional | ✓ | ✗ | ✗ |
| 2 | `cfg2_depth_first` | DepthFirst | FirstFractional | ✗ | ✗ | ✗ |
| 3 | `cfg3_best_first` | BestFirst | FirstFractional | ✗ | ✗ | ✗ |
| 4 | `cfg4_strong_branch` | BreadthFirst | StrongBranching | ✗ | ✗ | ✗ |
| 5 | `cfg5_pseudocost` | BreadthFirst | PseudoCost | ✗ | ✗ | ✗ |
| 6 | `cfg6_cuts` | BreadthFirst | FirstFractional | ✗ | ✗ | ✓ |
| 7 | `cfg7_heuristics` | BreadthFirst | FirstFractional | ✗ | ✓ (full) | ✓ |
| 8 | `cfg8_nonlinear` | BestFirst | FirstFractional | ✗ | ✗ | ✗ |
| 9 | `cfg9_complete` | DepthFirst | PseudoCost | ✓ | ✓ (selected) | ✓ |

---

## 🧩 Available Components

### Node Selection Strategies
| Class | Description |
|---|---|
| `DepthFirst()` | DFS — explores deepest nodes first |
| `BestFirst()` | Best-bound — explores node with best LP bound |
| `BreadthFirst()` | BFS — explores nodes level by level |

### Branch Variable Rules
| Class | Description |
|---|---|
| `FirstFractionalRule()` | Picks the first fractional integer variable |
| `MostFractionalRule()` | Picks the variable closest to 0.5 |
| `StrongBranchingRule(max_candidates, score_mode)` | Evaluates LP degradation for top candidates |
| `PseudoCostRule()` | Uses historical branching gains to estimate scores |

### Presolver Rules
| Class | Description |
|---|---|
| `BoundTightening()` | Tightens variable bounds via constraint propagation |
| `RowFeasibilityAndRedundancy()` | Detects infeasible or redundant rows |
| `SubstituteFixedVariables()` | Substitutes variables fixed by bounds |
| `ImplicitFreeVariableDetection()` | Detects implicitly free variables |
| `SingletonRowReduction()` | Converts singleton rows into bound updates |
| `EuclideanReduction()` | Divides rows by GCD of coefficients |
| `CoefficientStrengthening()` | Strengthens coefficients of integer variables |
| `ChvatalGomoryStrengthening()` | Applies Chvátal-Gomory rounding to rows |
| `ParallelRows()` | Detects and removes parallel/duplicate rows |

### Primal Heuristics
| Class | Description |
|---|---|
| `RoundingHeuristic()` | Rounds fractional LP solution |
| `FixAndPropagate()` | Fixes variables by confidence and propagates |
| `FeasibilityPump()` | Alternates between LP projection and integer rounding |
| `FeasibilityJump()` | Jump-based feasibility search |
| `FractionalDiving()` | Diving guided by fractionality |
| `GuidedDiving()` | Diving guided by objective coefficients |
| `LockDiving()` | Diving guided by variable locking scores |
| `RENS()` | Relaxation Enforced Neighborhood Search |
| `RINS()` | Relaxation Induced Neighborhood Search |
| `LocalSearch()` | Local search around incumbent |
| `LocalBranching()` | Hamming-ball neighborhood around incumbent |
| `LNS(destroy_rule)` | Large Neighborhood Search with pluggable destroy rules |
| `ALNS()` | Adaptive LNS with operator selection |

### Cutting Planes
| Class | Description |
|---|---|
| `MixedIntegerGomoryCut()` | Gomory mixed-integer cuts from simplex tableau |
| `SimpleMIR()` | Simple mixed-integer rounding cuts |
| `LiftingExtending()` | Lifting and extending cuts |
| `CoverCut()` | Cover cuts for binary knapsack-like constraints |

---

## 📄 Instance Format

Instances must be in **MPS format** (`.mps`), compatible with the MIPLIB standard. Place them in the `problems/` (for `milp.py`) or `problems_nl/` (for `execution_examples.py`) directory.

---

## 📊 Output Files

| File | Generated by | Description |
|---|---|---|
| `*.json` | Both scripts | Full B&B statistics per instance |
| `table_*.csv` | `execution_examples.py` | Metric tables per configuration |
| `table_*.tex` | `execution_examples.py` | LaTeX-ready tables per configuration |
| `comparison_all_configs.csv/.tex` | `execution_examples.py` | Cross-config comparison |
