import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, y_dim, width_size, depth, key):
        self.mlp = eqx.nn.MLP(y_dim, y_dim, width_size, depth, key=key)

    def __call__(self, t, y, args):
        return self.mlp(y)


@eqx.filter_jit
def solve(model, y0, adjoint):
    term = dfx.ODETerm(model)
    solver = dfx.Euler()
    t0 = 0.0
    t1 = 10.0
    dt0 = 0.01
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=adjoint,
        max_steps=1000,
    )
    return sol.ys


@eqx.filter_value_and_grad
def grad_loss(model, y0, adjoint):
    ys = eqx.filter_vmap(solve, in_axes=(None, 0, None))(model, y0, adjoint)
    return jnp.mean(ys**2)


def measure_runtime(y0, model, adjoint):
    tic = time.time()
    loss, grads = grad_loss(model, y0, adjoint)
    toc = time.time()
    print(f"Compile time: {(toc - tic):.5f}")

    repeats = 10
    tic = time.time()
    for i in range(repeats):
        loss, grads = jax.block_until_ready(grad_loss(model, y0, adjoint))
    toc = time.time()
    print(f"Runtime: {((toc - tic) / repeats):.5f}")


y_dim = 100
width_size = 100
depth = 4
model = VectorField(y_dim, width_size, depth, key=jr.PRNGKey(10))

print("Batch Size = 1")
print("--------------")
y0 = jnp.ones((1, y_dim))
print("Recursive")
adjoint = dfx.RecursiveCheckpointAdjoint()
measure_runtime(y0, model, adjoint)
print("Reversible")
adjoint = dfx.ReversibleAdjoint()
measure_runtime(y0, model, adjoint)

print("\nBatch Size = 1000")
print("-----------------")
y0 = jnp.ones((1000, y_dim))
print("Recursive")
adjoint = dfx.RecursiveCheckpointAdjoint()
measure_runtime(y0, model, adjoint)
print("Reversible")
adjoint = dfx.ReversibleAdjoint()
measure_runtime(y0, model, adjoint)
