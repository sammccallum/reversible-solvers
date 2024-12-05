import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

from reversible import VectorField, lotka_volterra, solve, train

jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)

    args = (0.5, 0.05, 2.0, 0.05)
    vf = VectorField(y_dim=2, hidden_size=10, key=jr.PRNGKey(0))
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    # adjoint = dfx.ReversibleAdjoint()
    y0 = jnp.array([10.0, 10.0])
    ts, data = solve(lotka_volterra, y0, adjoint, args)
    # plot_lotka_volterra(ts, data)
    train(vf, y0, data, adjoint, args=None, ode_model_name="lotka_volterra", steps=1000)
