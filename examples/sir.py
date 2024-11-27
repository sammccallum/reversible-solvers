import math
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from reversible import MemoryTracker

jax.config.update("jax_enable_x64", True)


def calculate_checkpoints(n_steps):
    return math.floor(-1.5 + math.sqrt(2 * n_steps + 0.25)) + 1


def vector_field(t, y, args):
    beta, gamma = args
    S, I, R = y
    dyS = -beta * I * S
    dyI = beta * I * S - gamma * I
    dyR = gamma * I
    return (dyS, dyI, dyR)


def solve(args):
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t1=True)
    term = dfx.ODETerm(vector_field)
    y0 = (1.0, 0.1, 0.0)
    t0 = 0
    t1 = 2
    dt0 = 0.001
    n_steps = int((t1 - t0) / dt0)
    # n_checkpoints = calculate_checkpoints(n_steps)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        # adjoint=dfx.RecursiveCheckpointAdjoint(n_checkpoints),
        adjoint=dfx.ReversibleAdjoint(),
    )
    ts = sol.ts
    S = sol.ys[0][:n_steps]
    I = sol.ys[1][:n_steps]
    R = sol.ys[2][:n_steps]

    return ts, (S, I, R)


def plot():
    args = (jnp.asarray(10.0), jnp.asarray(2.0))
    ts, (S, I, R) = solve(args)
    plt.plot(ts, S, ".-", color="tab:blue", label="S")
    plt.plot(ts, I, ".-", color="tab:red", label="I")
    plt.plot(ts, R, ".-", color="tab:green", label="R")

    plt.legend()
    plt.savefig("../data/imgs/sir.png", dpi=300)


@eqx.filter_value_and_grad
def grad_loss(args, data):
    S_data, I_data, R_data = data
    _, (S, I, R) = solve(args)
    S_error = jnp.sum((S - S_data) ** 2)
    I_error = jnp.sum((I - I_data) ** 2)
    R_error = jnp.sum((R - R_data) ** 2)
    return S_error + I_error + R_error


@eqx.filter_jit
def make_step(args, data, optim, opt_state):
    loss, grads = grad_loss(args, data)
    updates, opt_state = optim.update(grads, opt_state)
    args = eqx.apply_updates(args, updates)
    return loss, args, opt_state


if __name__ == "__main__":
    mem = MemoryTracker()
    args = (jnp.asarray(10.0), jnp.asarray(2.0))
    _, data = solve(args)

    args = jnp.asarray(8.0), jnp.asarray(3.0)

    lr = 3e-3
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(args, eqx.is_inexact_array))

    tic = time.time()
    loss, args, opt_state = make_step(args, data, optim, opt_state)
    toc = time.time()
    print(f"Compilation time: {toc - tic}")

    mem.start()
    tic = time.time()
    steps = 10
    for step in range(steps):
        loss, args, opt_state = make_step(args, data, optim, opt_state)
        print(f"Step: {step:.2f}, loss: {loss:.2f}")
    toc = time.time()
    mem_usage = mem.end()
    print(f"Runtime: {toc - tic}")
    print(f"Memory: {mem_usage} MB")
