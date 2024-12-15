import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

from reversible import SEIRS, VectorField, plot_SEIRS, solve, train

jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint", required=True)
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()

    adjoint_name = script_args.adjoint
    checkpoints = int(script_args.checkpoints)

    if adjoint_name == "reversible":
        adjoint = dfx.ReversibleAdjoint()

    elif adjoint_name == "recursive":
        adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)

    scale_factor = 1 / 25
    args = [1 / 76, 0.15 * 365.25, 1.0, 365.25 / 7, 365.25 / 14, 0.0]
    scaled_args = []
    for arg in args:
        scaled_arg = scale_factor * arg
        scaled_args.append(scaled_arg)

    vf = VectorField(y_dim=4, hidden_size=10, key=jr.PRNGKey(0))
    y0 = jnp.array([0.99, 0.01, 0.0, 0.0])
    t1 = 10.0
    dt0 = 0.01
    ts, data = solve(SEIRS, y0, t1, dt0, adjoint, scaled_args)
    # plot_SEIRS(ts, data)
    train(
        vf,
        y0,
        t1,
        dt0,
        data,
        adjoint,
        args=scaled_args,
        ode_model_name="seirs",
        steps=1000,
    )
    # source: https://www.nature.com/articles/s41592-020-0856-2
