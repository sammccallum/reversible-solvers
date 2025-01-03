import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr

from reversible import TimeDependentVectorField, solve, train, white_dwarf

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

    args = 0.001
    vf = TimeDependentVectorField(y_dim=2, hidden_size=10, key=jr.PRNGKey(0))
    y0 = jnp.array([1.0, 0.0])
    t1 = 5
    dt0 = 0.005
    solver = dfx.RK3Simple()
    ts, data = solve(white_dwarf, y0, t1, dt0, solver, adjoint, args)

    train(
        vf,
        y0,
        t1,
        dt0,
        data,
        adjoint,
        solver,
        args=args,
        ode_model_name="white_dwarf_Ralston3",
        steps=1000,
    )
