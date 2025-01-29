# Efficient, Accurate and Stable Gradients for Neural ODEs

This repository accompanies the Reversible Solver method introduced [here](https://arxiv.org/abs/2410.11648).

## Diffrax
The reversible method is implemented in [diffrax](https://github.com/patrick-kidger/diffrax). This is a work in progress - see the fork [here](https://github.com/sammccallum/diffrax). To install and checkout to the reversible branch, run
```bash
git clone https://github.com/sammccallum/diffrax.git
pip install -e diffrax
cd diffrax/
git checkout reversible 
```

## Experiments
The experiments presented in the paper can be found in the `experiments` directory. The experiments require an installation of the `reversible` and `diffrax` libraries. To install, run
```bash
git clone https://github.com/sammccallum/reversible-ICML.git
pip install -e reversible

git clone https://github.com/sammccallum/diffrax.git
pip install -e diffrax
cd diffrax
git checkout arxiv
```
Note that the `arxiv` branch in `diffrax` contains the archived code used to run the experiments.