# Efficient, Accurate and Stable Gradients for Neural ODEs

Code for the dynamical systems and timeseries experiments in the [paper](https://arxiv.org/abs/2410.11648).

To preserve the original state used to run the experiments, we seperate the code into dynamical systems and timeseries sub-directories.

| Directory | Experiments |
| --- | --- |
| [`dynamical_systems/`](dynamical_systems/README.md) | Convergence, computational complexity, physical systems, and white dwarf modelling |
| [`timeseries/`](timeseries/README.md) | EigenWorms, PhysioNet 2019 Sepsis, and Speech Commands classification |

```text
reversible-solvers/
├── README.md
├── dynamical_systems/
│   ├── README.md
│   ├── pyproject.toml
│   ├── reversible/
│   └── experiments/
└── timeseries/
    ├── README.md
    ├── pyproject.toml
    ├── uv.lock
    ├── reversible_solvers_timeseries/
    └── experiments/
```

## Installation

First clone the repo:

```bash
git clone https://github.com/sammccallum/reversible-solvers.git
cd reversible-solvers
```

Then, follow the `README.md` instructions in each sub-directory.
