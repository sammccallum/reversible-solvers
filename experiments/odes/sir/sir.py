import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

from reversible import SIR, VectorField, solve, train

jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)

    ode = "sir"
    args = (jnp.asarray(1.5), jnp.asarray(0.2))
    vf = VectorField(y_dim=3, hidden_size=10, key=jr.PRNGKey(0))
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    _, data = solve(SIR, adjoint, args)

    train(vf, data, adjoint, args=None, ode_model_name="sir", steps=1000)
