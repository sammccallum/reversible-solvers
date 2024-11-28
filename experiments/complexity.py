import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from reversible import SIR, MemoryTracker

jax.config.update("jax_enable_x64", True)


def solve(args, n, adjoint):
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(t1=True)
    term = dfx.ODETerm(SIR)
    y0 = (1.0, 0.1, 0.0)
    t0 = 0
    t1 = 2
    dt0 = 0.01

    if adjoint == "checkpointed":
        adjoint = dfx.RecursiveCheckpointAdjoint(n)
    elif adjoint == "reversible":
        adjoint = dfx.ReversibleAdjoint()
    else:
        raise ValueError(f"{adjoint} is not a valid adjoint.")

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

    S = sol.ys[0][-1]
    I = sol.ys[1][-1]
    R = sol.ys[2][-1]

    return S, I, R


@eqx.filter_value_and_grad
def grad_loss(args, n, adjoint, data):
    S_data, I_data, R_data = data
    S, I, R = solve(args, n, adjoint)
    S_error = jnp.sum((S - S_data) ** 2)
    I_error = jnp.sum((I - I_data) ** 2)
    R_error = jnp.sum((R - R_data) ** 2)
    return S_error + I_error + R_error


@eqx.filter_jit
def make_step(args, n, adjoint, data, optim, opt_state):
    loss, grads = grad_loss(args, n, adjoint, data)
    updates, opt_state = optim.update(grads, opt_state)
    args = eqx.apply_updates(args, updates)
    return loss, args, opt_state


if __name__ == "__main__":
    mem = MemoryTracker()
    true_args = (jnp.asarray(10.0), jnp.asarray(2.0))
    data = solve(true_args, n=62, adjoint="checkpointed")

    args = jnp.asarray(8.0), jnp.asarray(3.0)
    num = 6
    ns = np.logspace(1, 6, num, base=2, dtype=np.int64)

    adjoint = "checkpointed"
    times = np.zeros(num)
    mems = np.zeros(num)

    lr = 3e-3
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(args, eqx.is_inexact_array))

    num_steps = 10
    for i in range(num):
        n = int(ns[i])
        print(f"Checkpoints: {n}")

        tic = time.time()
        loss, args, opt_state = make_step(args, n, adjoint, data, optim, opt_state)
        toc = time.time()
        print(f"Compilation time: {toc - tic}")

        mem.start()
        tic = time.time()
        for step in range(num_steps):
            loss, args, opt_state = make_step(args, n, adjoint, data, optim, opt_state)
        toc = time.time()
        print(f"Runtime: {toc - tic}")
        mem_usage = mem.end()
        print(f"Memory: {mem_usage} MB")
        times[i] = toc - tic
        mems[i] = mem_usage

    np.save(f"../data/results/complexity/{adjoint}_times", times)
    np.save(f"../data/results/complexity/{adjoint}_mems", mems)
