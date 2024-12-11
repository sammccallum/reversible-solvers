import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from reversible import TimeDependentVectorField, solve, train, white_dwarf

jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)

    args = 0.001
    vf = TimeDependentVectorField(y_dim=2, hidden_size=10, key=jr.PRNGKey(0))
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    # adjoint = dfx.ReversibleAdjoint()
    y0 = jnp.array([1.0, 0.0])
    t1 = 5
    dt0 = 0.005
    ts, data = solve(white_dwarf, y0, t1, dt0, adjoint, args)
    # plt.plot(ts, data[:, 0], ".-", color="tab:red")
    # plt.plot(ts, data[:, 1], ".-", color="tab:blue")
    # plt.savefig("white_dwarf.png", dpi=300)
    train(
        vf,
        y0,
        t1,
        dt0,
        data,
        adjoint,
        args=args,
        ode_model_name="white_dwarf",
        steps=1000,
    )
