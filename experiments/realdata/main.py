import argparse

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from reversible import train

jax.config.update("jax_enable_x64", True)


def linear_interpolation(ts, xs, ts_grid):
    interp_fn = interp1d(ts, xs)
    xs_interp = interp_fn(ts_grid)
    return ts_grid, xs_interp


def load_double_linear():
    data = np.loadtxt("data/real_double_linear_h_1.txt", skiprows=1)
    data = data[:, 1:]
    ts = data[:, 0]
    ts = ts - np.min(ts)
    xs = data[:, 1:]

    n = 500
    ts_grid = np.linspace(0, 3.0, num=n)
    xs_interp = np.zeros((n, xs.shape[1]))
    for i in range(xs.shape[1]):
        ts_grid, xs_interp[:, i] = linear_interpolation(ts, xs[:, i], ts_grid)
    return ts_grid, xs_interp / 100


class VectorField(eqx.Module):
    layers: list

    def __init__(self, y_dim, hidden_size, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(y_dim, hidden_size, use_bias=True, key=key1),
            eqx.nn.Linear(hidden_size, y_dim, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers[:-1]:
            y = layer(y)
            y = jnp.tanh(y)
        y = self.layers[-1](y)
        return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--key", required=True)
    script_args = parser.parse_args()

    adjoint_name = script_args.adjoint
    checkpoints = int(script_args.checkpoints)
    key = int(script_args.key)

    if adjoint_name == "reversible":
        adjoint = dfx.ReversibleAdjoint()

    elif adjoint_name == "recursive":
        adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)

    ts, xs = load_double_linear()

    vf = VectorField(y_dim=4, hidden_size=10, key=jr.PRNGKey(key))
    y0 = xs[0]
    t1 = ts[-1]
    dt0 = ts[1] - ts[0]
    solver = dfx.RK3Simple()

    ts, ys_pred = train(
        vf,
        y0,
        t1,
        dt0,
        xs[1:],
        adjoint,
        solver,
        args=None,
        ode_model_name="double_linear_RK3",
        steps=10000,
    )

    plt.plot(ts, ys_pred)
    plt.savefig("double_linear_pred.png", dpi=300)
