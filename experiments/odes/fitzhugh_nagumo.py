import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from reversible import VectorField, fitzhugh_nagumo, solve, train

jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)

    args = (0.5, 0.7, 0.8, 12.5)
    vf = VectorField(y_dim=2, hidden_size=10, key=jr.PRNGKey(0))
    # adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    adjoint = dfx.ReversibleAdjoint()
    y0 = jnp.array([1.0, 1.0])
    t1 = 20
    dt0 = 0.02
    ts, data = solve(fitzhugh_nagumo, y0, t1, dt0, adjoint, args)
    # plt.plot(ts, data[:, 0], ".-", color="tab:red")
    # plt.plot(ts, data[:, 1], ".-", color="tab:blue")
    # plt.savefig("fitzhugh_nagumo.png", dpi=300)
    train(
        vf,
        y0,
        t1,
        dt0,
        data,
        adjoint,
        args=args,
        ode_model_name="fitzhugh_nagumo",
        steps=1000,
    )
