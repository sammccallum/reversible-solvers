import argparse
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)


def linear(t, y, args):
    alpha = args
    return alpha * y


# Simple neural vector field
class VectorField(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(1, 10, use_bias=True, key=key1),
            jnp.tanh,
            eqx.nn.Linear(10, 1, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers:
            y = layer(y)
        return y


@eqx.filter_value_and_grad
def grad_loss(y0__term, adjoint):
    y0, term = y0__term
    solver = dfx.Tsit5()
    t0 = 0
    t1 = 5
    dt0 = 0.01
    max_steps = int((t1 - t0) / dt0)
    ys = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        adjoint=adjoint,
        max_steps=max_steps,
    ).ys

    return jnp.sum(ys**2)


def run(adjoint):
    term = dfx.ODETerm(VectorField(jr.PRNGKey(0)))
    y0 = jnp.array([1.0])

    tic = time.time()
    grad_loss((y0, term), adjoint)
    toc = time.time()
    runtime = toc - tic

    return runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint", required=True)
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()

    adjoint_name = script_args.adjoint
    checkpoints = int(script_args.checkpoints)
    filename = f"../data/results/complexity/{adjoint_name}_complexity.csv"

    if adjoint_name == "reversible":
        adjoint = dfx.ReversibleAdjoint()

    elif adjoint_name == "recursive":
        adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)

    num_repeats = 10
    data = np.zeros((1, num_repeats + 1))
    data[:, 0] = checkpoints
    for repeat in range(num_repeats):
        # Compile step
        if repeat == 0:
            run(adjoint)

        # Run
        data[:, repeat + 1] = run(adjoint)
    with open(filename, "a") as file:
        np.savetxt(file, data, fmt="%.5f", delimiter=",")
