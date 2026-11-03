"""Neural RDE EigenWorms classification example.

Uses the EigenWorms dataset (5-class worm behaviour classification from
6-channel motion time series, 17984 timesteps). Preprocessing computes
log-signatures on a coarse grid; the model is a Neural ODE whose vector
field is driven by the piecewise-constant log-signature.
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

from reversible_solvers_timeseries.datasets import get_eigenworms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    logsig_size: int
    hidden_size: int

    def __init__(self, logsig_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.logsig_size = logsig_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * logsig_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.relu,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        logsigs, interval_times = args
        n_intervals = logsigs.shape[0]

        # Find which interval t belongs to
        j = jnp.searchsorted(interval_times, t, side="right") - 1
        j = jnp.clip(j, 0, n_intervals - 1)

        logsig_j = logsigs[j]  # (L,)

        return self.mlp(y).reshape(self.hidden_size, self.logsig_size) @ logsig_j


class NeuralRDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    solver: diffrax.AbstractSolver

    def __init__(
        self, in_channels, logsig_size, hidden_size, width_size, depth,
        num_classes, solver, *, key, **kwargs,
    ):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(in_channels, hidden_size, 128, 1, key=ikey)
        self.func = Func(logsig_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, num_classes, key=lkey)
        self.solver = solver

    def __call__(self, logsigs, interval_times, x0):
        term = diffrax.ODETerm(self.func)
        solver = self.solver
        t0 = interval_times[0]
        t1 = interval_times[-1]
        dt0 = interval_times[1] - interval_times[0]
        y0 = self.initial(x0)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args=(logsigs, interval_times),
            max_steps=logsigs.shape[0],
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            saveat=diffrax.SaveAt(t1=True),
        )
        return self.linear(solution.ys[-1])


class ReversibleNeuralRDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    solver: diffrax.AbstractSolver

    def __init__(
        self, in_channels, logsig_size, hidden_size, width_size, depth,
        num_classes, solver, *, key, **kwargs,
    ):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(in_channels, hidden_size, 128, 1, key=ikey)
        self.func = Func(logsig_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, num_classes, key=lkey)
        self.solver = solver

    def __call__(self, logsigs, interval_times, x0):
        term = diffrax.ODETerm(self.func)
        solver = self.solver
        t0 = interval_times[0]
        t1 = interval_times[-1]
        dt0 = interval_times[1] - interval_times[0]
        y0 = self.initial(x0)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args=(logsigs, interval_times),
            max_steps=logsigs.shape[0],
            adjoint=diffrax.ReversibleAdjoint(),
            saveat=diffrax.SaveAt(t1=True),
        )
        return self.linear(solution.ys[-1])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main(
    lr=1e-3,
    epochs=200,
    hidden_size=64,
    width_size=128,
    depth=4,
    step_size=32,
    logsig_depth=3,
    seed=5678,
    base_solver=None,
    method="checkpointed",
    lr_patience=15,
    lr_factor=0.1,
    lr_threshold=1e-6,
    min_lr=1e-7,
):
    print("Loading data...")
    (
        interval_times,
        train_logsigs, val_logsigs, test_logsigs,
        train_X0, val_X0, test_X0,
        train_labels, val_labels, test_labels,
        in_channels, logsig_size, num_classes,
    ) = get_eigenworms(step_size=step_size, depth=logsig_depth)

    # Derive n_intervals from the data (= maxlen // step_size)
    n_intervals = train_logsigs.shape[1]

    # Normalize time grid to [0, n_intervals] with unit steps (matching
    # reference implementation).  The logsigs are independent of this
    # parameterization; unit time keeps solver steps and network outputs O(1).
    interval_times = jnp.arange(n_intervals + 1, dtype=jnp.float32)

    wandb.init(
        project="Reversible Solvers Timeseries",
        config={
            "experiment": "eigenworms",
            "model": "nrde",
            "lr": lr,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "width_size": width_size,
            "depth": depth,
            "step_size": step_size,
            "n_intervals": n_intervals,
            "logsig_depth": logsig_depth,
            "logsig_size": logsig_size,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "seed": seed,
            "method": method,
            "lr_patience": lr_patience,
            "lr_factor": lr_factor,
            "min_lr": min_lr,
        },
    )
    run_start = time.time()

    key = jr.PRNGKey(seed)
    model_key = jr.split(key, 1)[0]

    if base_solver is None:
        base_solver = diffrax.MidpointSimple()

    if method == "reversible":
        solver = diffrax.UReversible(base_solver)
        model = ReversibleNeuralRDE(
            in_channels, logsig_size, hidden_size, width_size, depth,
            num_classes, solver, key=model_key,
        )
    elif method == "checkpointed":
        model = NeuralRDE(
            in_channels, logsig_size, hidden_size, width_size, depth,
            num_classes, base_solver, key=model_key,
        )
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'reversible' or 'checkpointed'."
        )

    def batch_loss(model, logsig_i, x0_i, label_i):
        pred_logits = jax.vmap(model, in_axes=(0, None, 0))(
            logsig_i, interval_times, x0_i,
        )
        loss = jnp.mean(
            -jax.nn.log_softmax(pred_logits)[jnp.arange(len(label_i)), label_i]
        )
        acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == label_i)
        return loss, acc

    @eqx.filter_jit
    def make_step(model, data_i, opt_state):
        logsig_i, x0_i, label_i = data_i
        (bxe, acc), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(
            model, logsig_i, x0_i, label_i,
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return bxe, acc, model, opt_state

    @eqx.filter_jit
    def eval_loss(model, logsig_i, x0_i, label_i):
        pred_logits = jax.vmap(model, in_axes=(0, None, 0))(
            logsig_i, interval_times, x0_i,
        )
        loss = jnp.mean(
            -jax.nn.log_softmax(pred_logits)[jnp.arange(len(label_i)), label_i]
        )
        acc = jnp.mean(jnp.argmax(pred_logits, axis=-1) == label_i)
        return loss, acc

    optim = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optim.init(params)

    # Full-batch training (dataset is tiny: ~181 train samples)
    train_data = (train_logsigs, train_X0, train_labels)
    best_val_loss = float("inf")
    patience_counter = 0
    best_val_loss_for_lr = float("inf")
    lr_patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        epoch_loss, epoch_acc, model, opt_state = make_step(
            model, train_data, opt_state,
        )
        epoch_loss = float(epoch_loss)
        epoch_acc = float(epoch_acc)
        epoch_time = time.time() - epoch_start

        # Evaluate on validation set
        val_loss, val_acc = eval_loss(model, val_logsigs, val_X0, val_labels)

        # ReduceLROnPlateau on val loss
        current_val = float(val_loss)
        if current_val < best_val_loss_for_lr - lr_threshold:
            best_val_loss_for_lr = current_val
            lr_patience_counter = 0
        else:
            lr_patience_counter += 1
            if lr_patience_counter >= lr_patience:
                current_lr = float(opt_state.hyperparams["learning_rate"])
                new_lr = max(current_lr * lr_factor, min_lr)
                opt_state.hyperparams["learning_rate"] = jnp.array(
                    new_lr, dtype=jnp.float32
                )
                lr_patience_counter = 0
                best_val_loss_for_lr = current_val
                print(f"  Reducing LR to {new_lr:.2e}")

        current_lr = float(opt_state.hyperparams["learning_rate"])

        wandb.log({
            "epoch": epoch,
            "train/loss": epoch_loss,
            "train/acc": epoch_acc,
            "val/loss": float(val_loss),
            "val/acc": float(val_acc),
            "lr": current_lr,
            "epoch_time": epoch_time,
        })

        print(
            f"Epoch {epoch}: Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f}, "
            f"Val loss: {float(val_loss):.4f}, Val acc: {float(val_acc):.4f}, "
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
    test_loss, test_acc = eval_loss(model, test_logsigs, test_X0, test_labels)
    total_time = time.time() - run_start

    wandb.log({
        "test/loss": float(test_loss),
        "test/acc": float(test_acc),
        "total_time": total_time,
    })
    wandb.summary.update({
        "test/loss": float(test_loss),
        "test/acc": float(test_acc),
        "total_time": total_time,
    })

    print(
        f"Test loss: {float(test_loss):.4f}, Test Accuracy: {float(test_acc):.4f}"
    )

    wandb.finish()


def run_single_solver(base_solver, method, step_sizes):
    seeds = [10, 20, 30, 40, 50]
    for step_size in step_sizes:
        for seed in seeds:
            main(base_solver=base_solver, method=method, step_size=step_size, seed=seed)


if __name__ == "__main__":
    run_single_solver(
        base_solver=diffrax.RK4Simple(),
        method="reversible",
        step_sizes=[16, 64, 128],
    )

