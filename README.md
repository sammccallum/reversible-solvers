# Efficient, Accurate and Stable Gradients for Neural ODEs

This repository accompanies the Reversible Solver method introduced [here](https://arxiv.org/abs/2410.11648).

## Diffrax
The reversible method is implemented in [diffrax](https://github.com/patrick-kidger/diffrax). This is a work in progress - see the fork [here](https://github.com/sammccallum/diffrax). To install and checkout to the arxiv branch, run
```bash
git clone https://github.com/sammccallum/diffrax.git
pip install -e diffrax
cd diffrax/
git checkout arxiv
```

### Example
The reversible solvers can be used by passing `adjoint=diffrax.ReversibleAdjoint()` to `diffrax.diffeqsolve`:
```python
import jax.numpy as jnp
import diffrax

vf = lambda t, y, args: y
y0 = jnp.array([1.0])
term = diffrax.ODETerm(vf)
solver = diffrax.Tsit5()
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=0,
    t1=5,
    dt0=0.01,
    y0=y0,
    adjoint=diffrax.ReversibleAdjoint(),
)
```
The base solver `diffrax.Tsit5()` will be automatically wrapped into a reversible version and gradient calculation will follow the reversible backpropagation algorithm.

## Experiments
The experiments presented in the paper can be found in the `experiments` directory. The experiments require an installation of the `reversible` and `diffrax` libraries. To install, run
```bash
git clone https://github.com/sammccallum/reversible-solvers.git
pip install -e reversible

git clone https://github.com/sammccallum/diffrax.git
pip install -e diffrax
cd diffrax
git checkout arxiv
```
Note that the `arxiv` branch in `diffrax` contains the archived code used to run the experiments.