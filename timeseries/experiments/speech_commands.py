"""Neural CDE Speech Commands classification example.

Uses the Speech Commands v0.02 dataset (10-class keyword spotting from 1-second
audio clips). Preprocessing follows Kidger 2020 (NeurIPS) Appendix D.4:
raw 16kHz WAV → MFCC (20 coefficients) → Hermite cubic interpolation.
"""

import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax

import wandb
from reversible_solvers_timeseries.datasets import get_speech_commands

SEQ_LEN = 161  # MFCC frames from 16000 samples, n_fft=200, hop_length=100


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.relu,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    checkpoints: int
    max_steps: int

    def __init__(
        self, data_size, hidden_size, width_size, depth, num_classes, checkpoints,
        *, key, max_steps=1024, **kwargs
    ):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, num_classes, key=lkey)
        self.checkpoints = checkpoints
        self.max_steps = max_steps

    def __call__(self, ts, coeffs):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.Bosh3Simple()
        controller = diffrax.PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-3, atol=1e-3
        )
        dt0 = ts[1] - ts[0]
        y0 = self.initial(control.evaluate(ts[0]))
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            max_steps=self.max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint(self.checkpoints),
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
        )
        num_steps = solution.stats["num_accepted_steps"]
        return self.linear(solution.ys[-1]), num_steps


class ReversibleNeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    max_steps: int

    def __init__(
        self, data_size, hidden_size, width_size, depth, num_classes,
        *, key, max_steps=1024, **kwargs
    ):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, num_classes, key=lkey)
        self.max_steps = max_steps

    def __call__(self, ts, coeffs):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.UReversible(diffrax.Bosh3Simple())
        controller = diffrax.PIDController(
            pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-3, atol=1e-3
        )
        dt0 = ts[1] - ts[0]
        y0 = self.initial(control.evaluate(ts[0]))
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            max_steps=self.max_steps,
            adjoint=diffrax.ReversibleAdjoint(),
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
        )
        num_steps = solution.stats["num_accepted_steps"]
        return self.linear(solution.ys[-1]), num_steps


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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


def main(
    batch_size=1024,
    lr=0.0016,
    epochs=200,
    hidden_size=90,
    width_size=40,
    depth=4,
    num_classes=10,
    l2_weight=1e-4,
    seed=5678,
    method="checkpointed",
    checkpoints=64,
    max_steps=1024,
):
    wandb.init(
        project="Reversible Solvers Timeseries",
        config={
            "experiment": "speech_commands",
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "width_size": width_size,
            "depth": depth,
            "num_classes": num_classes,
            "l2_weight": l2_weight,
            "seed": seed,
            "method": method,
            "checkpoints": checkpoints if method == "checkpointed" else None,
            "max_steps": max_steps,
        },
    )
    run_start = time.time()

    key = jr.PRNGKey(seed)
    model_key, loader_key = jr.split(key, 2)

    print("Loading data...")
    (
        _times,
        train_ts,
        train_coeffs,
        val_ts,
        val_coeffs,
        test_ts,
        test_coeffs,
        train_labels,
        val_labels,
        test_labels,
        _train_fi,
        _val_fi,
        _test_fi,
        data_size,
    ) = get_speech_commands(intensity_data=False)

    if method == "reversible":
        model = ReversibleNeuralCDE(
            data_size, hidden_size, width_size, depth, num_classes, key=model_key,
            max_steps=max_steps,
        )
    elif method == "checkpointed":
        model = NeuralCDE(
            data_size,
            hidden_size,
            width_size,
            depth,
            num_classes,
            checkpoints,
            key=model_key,
            max_steps=max_steps,
        )
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'reversible' or 'checkpointed'."
        )

    linear_lr = lr * 100

    def batch_loss(model, ti, label_i, coeff_i):
        pred_logits, num_steps = jax.vmap(model)(ti, coeff_i)
        loss = jnp.mean(
            -jax.nn.log_softmax(pred_logits)[jnp.arange(len(label_i)), label_i]
        )
        # L2 regularisation on vector field parameters
        func_params = eqx.filter(model.func, eqx.is_inexact_array)
        l2 = sum(jnp.sum(p**2) for p in jax.tree.leaves(func_params))
        loss = loss + l2_weight * l2
        acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == label_i)
        return loss, (acc, jnp.mean(num_steps))

    @eqx.filter_jit
    def make_step(model, data_i, opt_state):
        ti, label_i, *coeff_i = data_i
        coeff_i = tuple(coeff_i)
        (bxe, aux), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(
            model, ti, label_i, coeff_i
        )
        acc, num_steps = aux
        updates, opt_state = optim.update(grads, opt_state, value=bxe)
        model = eqx.apply_updates(model, updates)
        return bxe, acc, num_steps, model, opt_state

    @eqx.filter_jit
    def eval_loss(model, ti, label_i, coeff_i):
        pred_logits, num_steps = jax.vmap(model)(ti, coeff_i)
        loss = jnp.mean(
            -jax.nn.log_softmax(pred_logits)[jnp.arange(len(label_i)), label_i]
        )
        acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == label_i)
        return loss, acc, jnp.mean(num_steps)

    # Per-layer learning rates + reduce-on-plateau
    params = eqx.filter(model, eqx.is_inexact_array)
    label_tree = jax.tree.map(lambda _: "rest", params)
    label_tree = eqx.tree_at(
        lambda t: t.linear,
        label_tree,
        jax.tree.map(lambda _: "linear", params.linear),
    )

    train_size = train_ts.shape[0]
    steps_per_epoch = train_size // batch_size

    optim = optax.chain(
        optax.multi_transform(
            {"rest": optax.adam(lr), "linear": optax.adam(linear_lr)},
            lambda _: label_tree,
        ),
        optax.contrib.reduce_on_plateau(
            factor=0.1,
            patience=10,
            accumulation_size=steps_per_epoch,
        ),
    )
    opt_state = optim.init(params)

    # Training loop
    loader = dataloader(
        (train_ts, train_labels) + train_coeffs, batch_size, key=loader_key
    )
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for _step in range(steps_per_epoch):
            data_i = next(loader)
            bxe, acc, num_steps, model, opt_state = make_step(model, data_i, opt_state)
            epoch_loss += float(bxe)
            epoch_acc += float(acc)

        epoch_loss /= steps_per_epoch
        epoch_acc /= steps_per_epoch
        epoch_time = time.time() - epoch_start

        # Evaluate on validation set
        val_loss, val_acc, num_steps = eval_loss(model, val_ts, val_labels, val_coeffs)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/acc": epoch_acc,
                "val/loss": float(val_loss),
                "val/acc": float(val_acc),
                "val/num_steps": num_steps,
                "epoch_time": epoch_time,
            }
        )

        print(
            f"Epoch {epoch}: Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f}, "
            f"Val loss: {float(val_loss):.4f}, Val acc: {float(val_acc):.4f}, Num steps: {int(num_steps)}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Early stopping
        if float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 100:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test evaluation
    test_loss, test_acc, num_steps = eval_loss(model, test_ts, test_labels, test_coeffs)
    total_time = time.time() - run_start

    wandb.log(
        {
            "test/loss": float(test_loss),
            "test/acc": float(test_acc),
            "test/num_steps": num_steps,
            "total_time": total_time,
        }
    )
    wandb.summary.update(
        {
            "test/loss": float(test_loss),
            "test/acc": float(test_acc),
            "total_time": total_time,
        }
    )

    print(f"Test loss: {float(test_loss):.4f}, Test Accuracy: {float(test_acc):.4f}")

    wandb.finish()


if __name__ == "__main__":
    main(method="reversible", seed=1)
    for checkpoints in (16, 32, 64, 128):
        main(method="checkpointed", checkpoints=checkpoints, seed=1)
