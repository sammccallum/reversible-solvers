import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from reversible import VectorField, lorenz, solve, train

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

    args = (10.0, 28.0, 8 / 3)
    vf = VectorField(y_dim=3, hidden_size=10, key=jr.PRNGKey(0))
    # adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    adjoint = dfx.ReversibleAdjoint()
    y0 = jnp.array([1.0, 1.0, 1.0])
    t1 = 1
    dt0 = 0.001
    ts, data = solve(lorenz, y0, t1, dt0, adjoint, args)
    # plt.plot(ts, data[:, 0], ".-", color="tab:red")
    # plt.plot(ts, data[:, 1], ".-", color="tab:blue")
    # plt.plot(ts, data[:, 2], ".-", color="tab:blue")
    # plt.savefig("lorenz.png", dpi=300)
    train(
        vf,
        y0,
        t1,
        dt0,
        data,
        adjoint,
        args=args,
        ode_model_name="lorenz",
        steps=1000,
    )
