import argparse
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from reversible import SIR, SIR_model

jax.config.update("jax_enable_x64", True)


def solve(vf, adjoint, args):
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(steps=True)
    term = dfx.ODETerm(vf)
    y0 = jnp.array([1.0, 0.1, 0.0])
    t0 = 0
    t1 = 10
    dt0 = 0.01
    n_steps = int((t1 - t0) / dt0)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
    )
    ts = sol.ts[:n_steps]
    ys = sol.ys[:n_steps]

    return ts, ys


@eqx.filter_value_and_grad
def grad_loss(vf, data, adjoint, args):
    _, ys = solve(vf, adjoint, args)
    return jnp.mean((data - ys) ** 2)


@eqx.filter_jit
def make_step(vf, data, adjoint, args, optim, opt_state):
    loss, grads = grad_loss(vf, data, args, adjoint)
    updates, opt_state = optim.update(grads, opt_state, vf)
    vf = eqx.apply_updates(vf, updates)
    return loss, vf, opt_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)

    ode = "sir"
    args = (jnp.asarray(1.5), jnp.asarray(0.2))
    vf = SIR_model(jr.PRNGKey(0))
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    # adjoint = dfx.ReversibleAdjoint()
    _, data = solve(SIR, adjoint, args)

    lr = 1e-2
    weight_decay = 1e-5
    optim = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optim.init(eqx.filter(vf, eqx.is_inexact_array))

    tic = time.time()
    loss, vf, opt_state = make_step(vf, data, adjoint, optim, opt_state)
    toc = time.time()
    print(f"Compilation time: {toc - tic}")

    tic = time.time()
    steps = 1000
    for step in range(steps):
        loss, vf, opt_state = make_step(vf, data, adjoint, optim, opt_state)
        if step % 100 == 0 or step == steps - 1:
            print(f"Step: {step}, Loss: {loss}")
    toc = time.time()
    with open(f"data/{ode}.txt", "a") as file:
        print(f"{adjoint}, runtime: {toc - tic}, loss: {loss:.8f}", file=file)

    eqx.tree_serialise_leaves("../../data/models/vf.eqx", vf)
