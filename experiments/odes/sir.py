import argparse
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

from reversible import SIR

jax.config.update("jax_enable_x64", True)


class VectorField(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(3, 10, use_bias=True, key=key1),
            eqx.nn.Linear(10, 3, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers[:-1]:
            y = layer(y)
            y = jnp.tanh(y)
        y = self.layers[-1](y)
        return y


def solve_data(args):
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(steps=True)
    term = dfx.ODETerm(SIR)
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
        adjoint=dfx.ReversibleAdjoint(),
    )
    ts = sol.ts[:n_steps]
    ys = sol.ys[:n_steps]

    return ts, ys


def solve(vf, adjoint):
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


def plot(model, adjoint):
    ts, SIR = solve(model, adjoint)
    # args = (jnp.asarray(1.5), jnp.asarray(0.2))
    # ts, SIR = solve_data(args)
    plt.plot(ts, SIR[:, 0], ".-", color="tab:blue", label="S")
    plt.plot(ts, SIR[:, 1], ".-", color="tab:red", label="I")
    plt.plot(ts, SIR[:, 2], ".-", color="tab:green", label="R")

    plt.legend()
    plt.savefig("../../data/imgs/sir_pred.png", dpi=300)


@eqx.filter_value_and_grad
def grad_loss(vf, data, adjoint):
    _, ys = solve(vf, adjoint)
    return jnp.mean((data - ys) ** 2)


@eqx.filter_jit
def make_step(vf, data, adjoint, optim, opt_state):
    loss, grads = grad_loss(vf, data, adjoint)
    updates, opt_state = optim.update(grads, opt_state, vf)
    vf = eqx.apply_updates(vf, updates)
    return loss, vf, opt_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)

    args = (jnp.asarray(1.5), jnp.asarray(0.2))
    _, data = solve_data(args)

    vf = VectorField(jr.PRNGKey(0))
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)
    # adjoint = dfx.ReversibleAdjoint()

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
    with open("sir.txt", "a") as file:
        print(f"{adjoint}, runtime: {toc - tic}, loss: {loss:.8f}", file=file)

    eqx.tree_serialise_leaves("../../data/models/vf_sir.eqx", vf)
    plot(vf, adjoint)
