import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax

from reversible import load_from_numpy

jax.config.update("jax_enable_x64", False)


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    initial: eqx.nn.MLP
    linear: eqx.nn.Linear
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key):
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=jnn.tanh,
            key=fkey,
        )
        self.linear = eqx.nn.Linear(hidden_size, 10, key=lkey)

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    vector_field: Func
    initial: eqx.nn.MLP
    linear: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, width_size, depth, *, key):
        ikey, fkey, lkey = jr.split(key, 3)
        self.vector_field = Func(data_size, hidden_size, width_size, depth, key=key)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.linear = eqx.nn.Linear(hidden_size, 10, key=lkey)

    def __call__(self, ts, coeffs):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.vector_field, control).to_ode()
        solver = diffrax.MidpointSimple()
        dt0 = 0.32
        y0 = self.initial(control.evaluate(ts[0]))
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=500,
        )
        prediction = jnn.softmax(self.linear(solution.ys[-1]))
        return prediction


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main():
    ts, train_coeffs, test_coeffs, y_train, y_test = load_from_numpy()
    key = jr.PRNGKey(10)
    model_key, loader_key = jr.split(key, 2)

    model = NeuralCDE(
        data_size=21,
        hidden_size=21,
        width_size=21,
        depth=4,
        key=model_key,
    )

    def loss(model, ts, coeffs, ys):
        pred = model(ts, coeffs)
        pred_argmax = jnp.argmax(pred, axis=0)
        acc = jnp.mean(ys == pred_argmax)
        # pred = jnp.take_along_axis(pred, jnp.expand_dims(ys, 1), axis=1)
        pred = jnp.take_along_axis(pred, ys[None], axis=0)
        return -jnp.mean(pred), acc

    grad_loss = eqx.filter_value_and_grad(has_aux=True)(loss)

    @eqx.filter_jit
    def make_step(model, ts, coeffs, ys, optim, opt_state):
        (value, acc), grads = grad_loss(model, ts, coeffs, ys)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, value, acc, opt_state

    steps = 1000
    batch_size = 1
    lr = 1e-3
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    for step, (*coeffs, ys) in zip(
        range(steps), dataloader((*train_coeffs, y_train), batch_size, key=loader_key)
    ):
        coeffs = jax.tree.map(lambda x: x[0], coeffs)

        print(f"Step: {step}")
        tic = time.time()
        model, value, acc, opt_state = jax.block_until_ready(
            make_step(model, ts, coeffs, ys[0], optim, opt_state)
        )
        toc = time.time()
        print(f"Time: {toc - tic}")
        if step == 5:
            break


if __name__ == "__main__":
    main()
