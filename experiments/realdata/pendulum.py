import argparse
import time as time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.interpolate import interp1d

jax.config.update("jax_enable_x64", True)


def linear_interpolation(ts, xs, ts_grid):
    interp_fn = interp1d(ts, xs)
    xs_interp = interp_fn(ts_grid)
    return ts_grid, xs_interp, interp_fn


def load_double_pend():
    data = np.loadtxt("data/real_double_pend_h_1.txt", skiprows=1)
    data = data[:, 1:]
    ts = data[:, 0]
    ts = ts - np.min(ts)
    xs = data[:, 1:5] / 10

    n = 500
    ts_grid = np.linspace(0, 2.0, num=n)
    interp_fn = dfx.LinearInterpolation(ts, xs)
    xs_interp = eqx.filter_vmap(interp_fn.evaluate)(ts_grid)
    return ts_grid, xs_interp, interp_fn


class VectorField(eqx.Module):
    layers: list

    def __init__(self, y_dim, hidden_size, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(y_dim, hidden_size, use_bias=True, key=key1),
            eqx.nn.Linear(hidden_size, y_dim, use_bias=True, key=key2),
        ]

    def __call__(self, t, y, args):
        for layer in self.layers[:-1]:
            y = layer(y)
            y = jnp.tanh(y)
        y = self.layers[-1](y)
        return y


def solve(vf, y0, t1, dt0, solver, adjoint, args, stepsize_controller):
    saveat = dfx.SaveAt(steps=True)
    term = dfx.ODETerm(vf)
    t0 = 0
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
        max_steps=500,
        stepsize_controller=stepsize_controller,
    )
    ts = sol.ts
    ys = sol.ys

    return ts, ys


def train(
    vf,
    y0,
    t1,
    dt0,
    ts_grid,
    ys_data,
    adjoint,
    solver,
    stepsize_controller,
    args,
    ode_model_name,
    steps=1000,
    lr=1e-2,
    weight_decay=1e-5,
):
    @eqx.filter_value_and_grad
    def grad_loss(vf):
        ts, ys = solve(vf, y0, t1, dt0, solver, adjoint, args, stepsize_controller)
        # slighty weird hack to manage the jnp.inf padded ts, ys arrays from diffrax
        # we set the inf ts to 10 and the inf ys to 1 which is well beyond our data range
        # so this has no impact on linear interpolation
        ts = jnp.where(jnp.isfinite(ts), ts, 10 * jnp.ones_like(ts))
        ys = jnp.where(jnp.isfinite(ys), ys, jnp.zeros_like(ys))
        interp_fn = dfx.LinearInterpolation(ts, ys)
        ys = eqx.filter_vmap(interp_fn.evaluate)(ts_grid)
        return jnp.mean((ys_data - ys) ** 2)

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

    best_loss = 100
    tic = time.time()
    for step in range(steps):
        loss, vf, opt_state = make_step(vf, optim, opt_state)
        if step % 100 == 0 or step == steps - 1:
            print(f"Step: {step}, Loss: {loss}")
        if loss < best_loss:
            best_loss = loss
    toc = time.time()

    data_file = f"data/{ode_model_name}.txt"
    with open(data_file, "a") as file:
        print(f"{adjoint}, runtime: {toc - tic}, loss: {best_loss:.8f}", file=file)

    ts, ys_pred = solve(
        vf,
        y0,
        t1,
        None,
        solver,
        adjoint,
        args,
        stepsize_controller=dfx.StepTo(np.linspace(0, 2.0, num=100)),
    )
    return ts, ys_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--key", required=True)
    script_args = parser.parse_args()

    adjoint_name = script_args.adjoint
    checkpoints = int(script_args.checkpoints)
    key = int(script_args.key)

    if adjoint_name == "reversible":
        adjoint = dfx.ReversibleAdjoint()

    elif adjoint_name == "recursive":
        adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)

    ts, xs, interp_fn = load_double_pend()
    plt.plot(ts, xs)
    plt.savefig("pend.png", dpi=300)
    plt.clf()

    vf = VectorField(y_dim=4, hidden_size=10, key=jr.PRNGKey(key))
    y0 = xs[0]
    t1 = ts[-1]
    dt0 = ts[1] - ts[0]
    solver = dfx.Dopri5()
    stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-8)

    ts, ys_pred = train(
        vf,
        y0,
        t1,
        dt0,
        ts,
        xs,
        adjoint,
        solver,
        stepsize_controller,
        args=None,
        ode_model_name="double_pend",
        steps=10000,
    )

    plt.plot(ts, ys_pred)
    plt.savefig("double_pend_pred.png", dpi=300)
