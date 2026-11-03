"""Neural CDE PhysioNet 2019 Sepsis classification example.

Uses the PhysioNet Computing in Cardiology Challenge 2019 dataset
(binary sepsis prediction, 34 time-varying + static clinical features,
irregular NaN-heavy time series).

Preprocessing matches the NeuralCDE reference implementation (Kidger 2020):
static features handled separately, ICULOS excluded, proper gap filling.
"""

import time

import diffrax
import wandb
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from reversible_solvers_timeseries.datasets import get_sepsis

MAX_SEQ_LEN = 72


def compute_auc(labels, scores):
    """Compute AUC using the trapezoidal rule (numpy, not JAX)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    desc_order = np.argsort(-scores)
    labels_sorted = labels[desc_order]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg

    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    auc = np.trapezoid(tpr, fpr)
    return float(auc)


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
    solver: diffrax.AbstractSolver
    exact: bool

    def __init__(self, data_size, hidden_size, width_size, depth, static_size, *, solver, exact=True, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(static_size, hidden_size, 256, 1, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)
        self.solver = solver
        self.exact = exact

    def __call__(self, ts, coeffs, static, fi):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        dt0 = ts[1] - ts[0]
        y0 = self.initial(static)
        t1 = ts[fi]
        if self.exact:
            adjoint = diffrax.RecursiveCheckpointAdjoint()
        else:
            adjoint = diffrax.BacksolveAdjoint()
        solution = diffrax.diffeqsolve(
            term,
            self.solver,
            ts[0],
            t1,
            dt0,
            y0,
            max_steps=MAX_SEQ_LEN,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(t1=True),
        )
        (prediction,) = jnn.sigmoid(self.linear(solution.ys[-1]))
        return prediction


class ReversibleNeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    solver: diffrax.AbstractSolver

    def __init__(self, data_size, hidden_size, width_size, depth, static_size, *, solver, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        self.initial = eqx.nn.MLP(static_size, hidden_size, 256, 1, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)
        self.solver = solver

    def __call__(self, ts, coeffs, static, fi):
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        dt0 = ts[1] - ts[0]
        y0 = self.initial(static)
        t1 = ts[fi]
        solution = diffrax.diffeqsolve(
            term,
            self.solver,
            ts[0],
            t1,
            dt0,
            y0,
            max_steps=MAX_SEQ_LEN,
            adjoint=diffrax.ReversibleAdjoint(),
            saveat=diffrax.SaveAt(t1=True),
        )
        (prediction,) = jnn.sigmoid(self.linear(solution.ys[-1]))
        return prediction


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
    lr=0.0032,
    epochs=200,
    hidden_size=64,
    width_size=64,
    depth=4,
    seed=5678,
    pos_weight=10.0,
    l2_weight=0.03,
    method="checkpointed",
    solver=diffrax.RK4Simple(),
    exact=True,
):
    solver_name = type(solver).__name__
    wandb.init(
        project="Reversible Solvers Timeseries",
        config={
            "experiment": "sepsis",
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "width_size": width_size,
            "depth": depth,
            "seed": seed,
            "pos_weight": pos_weight,
            "l2_weight": l2_weight,
            "method": method,
            "solver": solver_name,
            "exact": exact,
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
        train_fi,
        val_fi,
        test_fi,
        data_size,
        train_static,
        val_static,
        test_static,
    ) = get_sepsis(static_intensity=True, time_intensity=True)

    static_size = train_static.shape[-1]

    if method == "reversible":
        rev_solver = diffrax.UReversible(solver)
        model = ReversibleNeuralCDE(
            data_size, hidden_size, width_size, depth, static_size,
            solver=rev_solver, key=model_key,
        )
    elif method == "checkpointed":
        model = NeuralCDE(
            data_size, hidden_size, width_size, depth, static_size,
            solver=solver, exact=exact, key=model_key,
        )
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose 'reversible' or 'checkpointed'."
        )

    def batch_loss(model, ti, label_i, static_i, fi_i, coeff_i):
        pred = jax.vmap(model)(ti, coeff_i, static_i, fi_i)
        label_f = label_i.astype(jnp.float32)
        weight = jnp.where(label_i == 1, pos_weight, 1.0)
        bxe = -(label_f * jnp.log(pred + 1e-7) + (1 - label_f) * jnp.log(1 - pred + 1e-7))
        loss = jnp.mean(weight * bxe)
        # Per-tensor Frobenius norm regularisation on vector field parameters
        func_params = eqx.filter(model.func, eqx.is_inexact_array)
        reg = sum(jnp.sqrt(jnp.sum(p ** 2)) for p in jax.tree.leaves(func_params))
        loss = loss + l2_weight * reg
        acc = jnp.mean((pred > 0.5) == (label_i == 1))
        return loss, acc

    @eqx.filter_jit
    def make_step(model, data_i, opt_state):
        ti, label_i, static_i, fi_i, *coeff_i = data_i
        coeff_i = tuple(coeff_i)
        (bxe, acc), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(
            model, ti, label_i, static_i, fi_i, coeff_i
        )
        updates, opt_state = optim.update(grads, opt_state, value=bxe)
        model = eqx.apply_updates(model, updates)
        return bxe, acc, model, opt_state

    @eqx.filter_jit
    def eval_loss(model, ti, label_i, static_i, fi_i, coeff_i):
        pred = jax.vmap(model)(ti, coeff_i, static_i, fi_i)
        label_f = label_i.astype(jnp.float32)
        bxe = -(label_f * jnp.log(pred + 1e-7) + (1 - label_f) * jnp.log(1 - pred + 1e-7))
        loss = jnp.mean(bxe)
        acc = jnp.mean((pred > 0.5) == (label_i == 1))
        # Per-class accuracy
        pos_mask = label_i == 1
        neg_mask = label_i == 0
        pos_acc = jnp.where(
            jnp.sum(pos_mask) > 0,
            jnp.sum((pred > 0.5) & pos_mask) / jnp.sum(pos_mask),
            0.0,
        )
        neg_acc = jnp.where(
            jnp.sum(neg_mask) > 0,
            jnp.sum((pred <= 0.5) & neg_mask) / jnp.sum(neg_mask),
            0.0,
        )
        return loss, acc, pos_acc, neg_acc

    @eqx.filter_jit
    def eval_predictions(model, ti, static_i, fi_i, coeff_i):
        return jax.vmap(model)(ti, coeff_i, static_i, fi_i)

    # Per-layer learning rates + reduce-on-plateau
    params = eqx.filter(model, eqx.is_inexact_array)
    label_tree = jax.tree.map(lambda _: "rest", params)
    label_tree = eqx.tree_at(
        lambda t: t.linear, label_tree,
        jax.tree.map(lambda _: "linear", params.linear),
    )

    train_size = train_ts.shape[0]
    steps_per_epoch = train_size // batch_size
    total_steps = steps_per_epoch * epochs

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps)
    optim = optax.adam(schedule)
    opt_state = optim.init(params)

    # Training loop
    loader = dataloader(
        (train_ts, train_labels, train_static, train_fi) + train_coeffs, batch_size, key=loader_key
    )
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0

        for _step in range(steps_per_epoch):
            data_i = next(loader)
            bxe, acc, model, opt_state = make_step(model, data_i, opt_state)
            epoch_loss += float(bxe)
            epoch_acc += float(acc)

        epoch_loss /= steps_per_epoch
        epoch_acc /= steps_per_epoch
        epoch_time = time.time() - epoch_start

        # Evaluate on validation set
        val_loss, val_acc, val_pos_acc, val_neg_acc = eval_loss(
            model, val_ts, val_labels, val_static, val_fi, val_coeffs
        )
        val_scores = eval_predictions(model, val_ts, val_static, val_fi, val_coeffs)
        val_auc = compute_auc(np.array(val_labels), np.array(val_scores))

        wandb.log({
            "epoch": epoch,
            "train/loss": epoch_loss,
            "train/acc": epoch_acc,
            "val/loss": float(val_loss),
            "val/acc": float(val_acc),
            "val/pos_acc": float(val_pos_acc),
            "val/neg_acc": float(val_neg_acc),
            "val/auc": val_auc,
            "epoch_time": epoch_time,
        })

        print(
            f"Epoch {epoch}: Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f}, "
            f"Val loss: {float(val_loss):.4f}, Val AUC: {val_auc:.4f}, "
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
    test_loss, test_acc, test_pos_acc, test_neg_acc = eval_loss(
        model, test_ts, test_labels, test_static, test_fi, test_coeffs
    )
    test_scores = eval_predictions(model, test_ts, test_static, test_fi, test_coeffs)
    test_auc = compute_auc(np.array(test_labels), np.array(test_scores))

    total_time = time.time() - run_start

    wandb.log({
        "test/loss": float(test_loss),
        "test/acc": float(test_acc),
        "test/auc": test_auc,
        "test/pos_acc": float(test_pos_acc),
        "test/neg_acc": float(test_neg_acc),
        "total_time": total_time,
    })
    wandb.summary.update({
        "test/loss": float(test_loss),
        "test/acc": float(test_acc),
        "test/auc": test_auc,
        "test/pos_acc": float(test_pos_acc),
        "test/neg_acc": float(test_neg_acc),
        "total_time": total_time,
    })

    print(
        f"Test loss: {float(test_loss):.4f}, Test Accuracy: {float(test_acc):.4f}, "
        f"Test AUC: {test_auc:.4f}, "
        f"Positive (sepsis) accuracy: {float(test_pos_acc):.4f}, "
        f"Negative (no sepsis) accuracy: {float(test_neg_acc):.4f}"
    )

    wandb.finish()


def run_all(method, solver, exact=True, num_repeats=5, **kwargs):
    """Run multiple repeats with different seeds."""
    seeds = [1, 2, 3, 4, 5][:num_repeats]
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{num_repeats} | method={method} | solver={type(solver).__name__} | exact={exact} | seed={seed}")
        print(f"{'='*60}\n")
        main(method=method, solver=solver, exact=exact, seed=seed, **kwargs)


if __name__ == "__main__":
    for solver in (
        diffrax.Euler(), diffrax.MidpointSimple(),
        diffrax.RK3Simple(), diffrax.RK4Simple(),
    ):
        run_all(method="reversible", solver=solver)
        run_all(method="checkpointed", solver=solver)
