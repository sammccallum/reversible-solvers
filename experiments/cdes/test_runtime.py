import os
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)


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


class QuadraticPath(dfx.AbstractPath):
    alpha: jnp.array

    def __init__(self, alpha):
        self.alpha = alpha

    @property
    def t0(self):
        return 0

    @property
    def t1(self):
        return 3

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        return self.alpha[0] * t0**2


def solve(alpha, model, adjoint):
    control = QuadraticPath(alpha)
    term = dfx.ControlTerm(model, control).to_ode()
    solver = dfx.MidpointSimple()
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=1,
        dt0=0.001,
        y0=jnp.ones((1000,)),
        adjoint=adjoint,
        max_steps=1000,
    )
    return sol.ys


@eqx.filter_grad
def grad_loss(alpha, model, adjoint):
    ys = solve(alpha, model, adjoint)
    # ys = jax.vmap(solve, in_axes=(0, None, None))(alphas, model, adjoint)
    return jnp.sum(ys**2)


@eqx.filter_jit
def time_backprop(alpha, model, adjoint):
    tic = time.time()
    grads1 = grad_loss(alpha, model, adjoint)
    toc = time.time()
    print(f"Compile time: {toc-tic}")

    tic = time.time()
    grads2 = grad_loss(alpha, model, adjoint)
    toc = time.time()
    print(f"Runtime: {toc - tic}")
    return grads1, grads2


if __name__ == "__main__":
    key = jr.PRNGKey(1)
    vf_key, cvf_key = jr.split(key, 2)
    vf = VectorField(y_dim=1000, hidden_size=10000, key=vf_key)
    alpha = jnp.array([1.0])
    print("Recursive")
    grads1, grads2 = time_backprop(alpha, vf, dfx.RecursiveCheckpointAdjoint())
    print("Reversible")
    grads1, grads2 = time_backprop(alpha, vf, dfx.ReversibleAdjoint())
