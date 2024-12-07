import argparse
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)


def load_data(permutation_key, split):
    X = np.load("data/character_trajectories.npy")
    y = np.load("data/labels.npy")
    dataset_size, ts_length, _ = X.shape
    ts = np.broadcast_to(np.linspace(0, 1, num=ts_length), (dataset_size, ts_length))
    Xs = np.concatenate([ts[:, :, None], X], axis=-1)

    Xs = jnp.array(Xs)
    y = jnp.array(y, dtype="int32") - 1  # [1 ... 20] to [0 ... 19]

    Xs = jr.permutation(permutation_key, Xs)
    y = jr.permutation(permutation_key, y)

    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test


class BaseVectorField(eqx.Module):
    data_size: int
    hidden_size: int
    mlp: eqx.nn.MLP

    def __init__(self, data_size, hidden_size, width_size, depth, key):
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.relu,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: BaseVectorField
    linear: eqx.nn.Linear
    adjoint: dfx.AbstractAdjoint

    def __init__(self, data_size, hidden_size, width_size, depth, adjoint, key):
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = BaseVectorField(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, 20, key=lkey)
        self.adjoint = adjoint

    def __call__(self, xs):
        ts = xs[:, 0]
        coeffs = dfx.backward_hermite_coefficients(ts, xs)
        control = dfx.CubicInterpolation(ts, coeffs)
        term = dfx.ControlTerm(self.func, control).to_ode()
        solver = dfx.Tsit5()
        dt0 = 0.01
        y0 = self.initial(control.evaluate(ts[0]))
        solution = dfx.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            adjoint=self.adjoint,
        )
        prediction = jax.nn.softmax(self.linear(solution.ys[-1]))
        return prediction


def dataloader(X, y, batch_size, key):
    dataset_size = X.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield (X[batch_perm], y[batch_perm])
            start = end
            end = start + batch_size


@eqx.filter_jit
def calculate_pred_and_accuracy(model, xs, ys):
    pred = jax.vmap(model)(xs)
    pred_argmax = jnp.argmax(pred, axis=1)
    acc = jnp.mean(ys == pred_argmax)
    return pred, acc


@eqx.filter_value_and_grad(has_aux=True)
def grad_loss(model, xs, ys):
    pred, acc = calculate_pred_and_accuracy(model, xs, ys)
    pred = jnp.take_along_axis(pred, jnp.expand_dims(ys, 1), axis=1)
    return -jnp.mean(pred), acc


@eqx.filter_jit
def make_step(model, xs, ys, optim, opt_state):
    (loss, acc), grads = grad_loss(model, xs, ys)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, acc, model, opt_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--adjoint", required=True)
    script_args = parser.parse_args()
    checkpoints = int(script_args.checkpoints)
    adjoint = script_args.adjoint

    key = jr.PRNGKey(100)
    model_key, load_key, dataloader_key = jr.split(key, 3)
    batch_size = 32
    X_train, X_test, y_train, y_test = load_data(load_key, split=70 * batch_size)

    if adjoint == "reversible":
        adjoint = dfx.ReversibleAdjoint()
    elif adjoint == "recursive":
        adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints)

    model = NeuralCDE(
        data_size=4,
        hidden_size=32,
        width_size=32,
        depth=4,
        adjoint=adjoint,
        key=model_key,
    )

    lr = 1e-3
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    tic = time.time()
    steps = 1000
    for step, (xs, ys) in zip(
        range(steps), dataloader(X_train, y_train, batch_size, dataloader_key)
    ):
        loss, acc, model, opt_state = make_step(model, xs, ys, optim, opt_state)
        if step % 100 == 0:
            print(f"step: {step}, loss: {loss}, accuracy: {acc}")
    toc = time.time()

    pred, acc = calculate_pred_and_accuracy(model, X_test, y_test)
    data_file = "data/runtime.txt"
    with open(data_file, "a") as file:
        print(f"{adjoint}, runtime: {toc - tic}, test accuracy: {acc:.8f}", file=file)
