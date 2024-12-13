import time

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import optax

from reversible import plot_lorenz


def solve(vf, y0, t1, dt0, adjoint, args):
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(steps=True)
    term = dfx.ODETerm(vf)
    t0 = 0
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
        max_steps=n_steps,
    )
    ts = sol.ts[:n_steps]
    ys = sol.ys[:n_steps]

    return ts, ys


def train(
    vf,
    y0,
    t1,
    dt0,
    data,
    adjoint,
    args,
    ode_model_name,
    steps=1000,
    lr=1e-2,
    weight_decay=1e-5,
):
    @eqx.filter_value_and_grad
    def grad_loss(vf):
        _, ys = solve(vf, y0, t1, dt0, adjoint, args)
        return jnp.mean((data - ys) ** 2)

    @eqx.filter_jit
    def make_step(vf, optim, opt_state):
        loss, grads = grad_loss(vf)
        updates, opt_state = optim.update(grads, opt_state, vf)
        vf = eqx.apply_updates(vf, updates)
        return loss, vf, opt_state

    optim = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optim.init(eqx.filter(vf, eqx.is_inexact_array))

    tic = time.time()
    loss, vf, opt_state = make_step(vf, optim, opt_state)
    toc = time.time()
    print(f"Compilation time: {toc - tic}")

    tic = time.time()
    for step in range(steps):
        loss, vf, opt_state = make_step(vf, optim, opt_state)
        if step % 100 == 0 or step == steps - 1:
            print(f"Step: {step}, Loss: {loss}")
    toc = time.time()

    data_file = f"data/{ode_model_name}.txt"
    with open(data_file, "a") as file:
        print(f"{adjoint}, runtime: {toc - tic}, loss: {loss:.8f}", file=file)

    ts, ys_pred = solve(vf, y0, t1, dt0, adjoint, args)
    plot_lorenz(ts, ys_pred)
