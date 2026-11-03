# Dynamical systems experiments

This directory contains the dynamical systems experiments: convergence, computational complexity, physical systems, and white dwarf modelling.

# Installation

```bash
uv sync
```

# Experiments

```bash
# Convergence figure
(cd experiments/convergence && uv run python convergence.py)

# Runtime and checkpointing comparison
(cd experiments/complexity && uv run bash run_complexity.sh)

# Physical systems
(cd experiments/physical_systems && uv run python oscillator.py)
(cd experiments/physical_systems && uv run bash run_checkpointed.sh)

# White dwarf modeling
(cd experiments/white_dwarf && uv run bash run_checkpointed.sh)

# Plot the tracked complexity results
(cd experiments/complexity/plotting && uv run python plot_complexity.py)
```
