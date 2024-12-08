import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

from reversible import VectorField, pendulum, solve, train

jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)

    args = (9.81, 10.0)
    vf = VectorField(y_dim=2, hidden_size=10, key=jr.PRNGKey(0))
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    y0 = jnp.array([jnp.pi / 2, 0.0])
    ts, data = solve(pendulum, y0, adjoint, args)
    train(vf, y0, data, adjoint, args=None, ode_model_name="pendulum", steps=1000)
