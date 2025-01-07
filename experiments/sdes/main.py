import argparse
import time
from typing import Union

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

jax.config.update("jax_enable_x64", True)


def lipswish(x):
    return 0.909 * jnn.silu(x)


class VectorField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, scale, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(scale_key, (hidden_size,), minval=0.9, maxval=1.1)
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y]))


class ControlledVectorField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(
        self, control_size, hidden_size, width_size, depth, scale, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(
                scale_key, (hidden_size, control_size), minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = control_size
        self.hidden_size = hidden_size

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )


class NeuralSDE(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField  # drift
    cvf: ControlledVectorField  # diffusion
    readout: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int
    adjoint: diffrax.AbstractAdjoint

    def __init__(
        self,
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        adjoint,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)

        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=True, key=vf_key)
        self.cvf = ControlledVectorField(
            noise_size, hidden_size, width_size, depth, scale=True, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, data_size, key=readout_key)

        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size
        self.adjoint = adjoint

    def __call__(self, ts, *, key):
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = ts[1] - ts[0]
        init_key, bm_key = jr.split(key, 2)
        init = jr.normal(init_key, (self.initial_noise_size,))
        control = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=dt0, shape=(self.noise_size,), key=bm_key
        )
        vf = diffrax.ODETerm(self.vf)  # Drift term
        cvf = diffrax.ControlTerm(self.cvf, control)  # Diffusion term
        terms = diffrax.MultiTerm(vf, cvf)
        solver = diffrax.EulerHeun()
        y0 = self.initial(init)

        saveat = diffrax.SaveAt(t0=True, steps=True)
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            dt0,
            y0,
            saveat=saveat,
            max_steps=ts.shape[0] - 1,
            adjoint=self.adjoint,
        )
        return jax.vmap(self.readout)(sol.ys)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField
    cvf: ControlledVectorField
    readout: eqx.nn.Linear
    adjoint: diffrax.AbstractAdjoint

    def __init__(
        self, data_size, hidden_size, width_size, depth, adjoint, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)

        self.initial = eqx.nn.MLP(
            data_size + 1, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=False, key=vf_key)
        self.cvf = ControlledVectorField(
            data_size, hidden_size, width_size, depth, scale=False, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, 1, key=readout_key)
        self.adjoint = adjoint

    def __call__(self, ts, ys):
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        solver = diffrax.MidpointSimple()
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = ts[1] - ts[0]
        y0 = self.initial(init)

        saveat = diffrax.SaveAt(t1=True)
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            dt0,
            y0,
            saveat=saveat,
            max_steps=ts.shape[0] - 1,
            adjoint=self.adjoint,
        )
        ys = jnp.concatenate([y0[None, :], sol.ys], axis=0)
        return jax.vmap(self.readout)(sol.ys)

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_util.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)


@jax.jit
@jax.vmap
def get_data(key):
    bm_key, y0_key, drop_key = jr.split(key, 3)

    mu = 0.02
    theta = 0.1
    sigma = 0.4

    t0 = 0.0
    t1 = 50.0
    t_size = 501

    def drift(t, y, args):
        return mu * t - theta * y

    def diffusion(t, y, args):
        return 2 * sigma * t / t1

    bm = diffrax.UnsafeBrownianPath(shape=(), key=bm_key)
    drift = diffrax.ODETerm(drift)
    diffusion = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift, diffusion)
    solver = diffrax.EulerHeun()
    dt0 = 0.1

    y0 = jr.uniform(y0_key, (1,), minval=-1, maxval=1)
    ts = jnp.linspace(t0, t1, t_size)
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms, solver, t0, t1, dt0, y0, saveat=saveat, adjoint=diffrax.DirectAdjoint()
    )
    ys = sol.ys

    return ts, ys


def dataloader(arrays, batch_size, loop, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        key = jr.split(key, 1)[0]
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        if not loop:
            break


@eqx.filter_jit
def loss(generator, discriminator, ts_i, ys_i, key, step=0):
    batch_size, _ = ts_i.shape
    key = jr.fold_in(key, step)
    key = jr.split(key, batch_size)
    fake_ys_i = jax.vmap(generator)(ts_i, key=key)
    real_score = jax.vmap(discriminator)(ts_i, ys_i)
    fake_score = jax.vmap(discriminator)(ts_i, fake_ys_i)
    return jnp.mean(real_score - fake_score)


@eqx.filter_grad
def grad_loss(g_d, ts_i, ys_i, key, step):
    generator, discriminator = g_d
    return loss(generator, discriminator, ts_i, ys_i, key, step)


def increase_update_initial(updates):
    get_initial_leaves = lambda u: jax.tree_util.tree_leaves(u.initial)
    return eqx.tree_at(get_initial_leaves, updates, replace_fn=lambda x: x * 10)


@eqx.filter_jit
def make_step(
    generator,
    discriminator,
    g_opt_state,
    d_opt_state,
    g_optim,
    d_optim,
    ts_i,
    ys_i,
    key,
    step,
):
    g_grad, d_grad = grad_loss((generator, discriminator), ts_i, ys_i, key, step)
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
    g_updates = increase_update_initial(g_updates)
    d_updates = increase_update_initial(d_updates)
    generator = eqx.apply_updates(generator, g_updates)
    discriminator = eqx.apply_updates(discriminator, d_updates)
    discriminator = discriminator.clip_weights()
    return generator, discriminator, g_opt_state, d_opt_state


def main(
    adjoint,
    filename,
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-5,
    discriminator_lr=1e-4,
    batch_size=1024,
    steps=10000,
    steps_per_print=200,
    dataset_size=8192,
    seed=5678,
):
    key = jr.PRNGKey(seed)
    (
        data_key,
        generator_key,
        discriminator_key,
        dataloader_key,
        train_key,
        evaluate_key,
        sample_key,
    ) = jr.split(key, 7)
    data_key = jr.split(data_key, dataset_size)

    ts, ys = get_data(data_key)
    _, _, data_size = ys.shape

    generator = NeuralSDE(
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        adjoint=adjoint,
        key=generator_key,
    )
    discriminator = NeuralCDE(
        data_size,
        hidden_size,
        width_size,
        depth,
        adjoint=adjoint,
        key=discriminator_key,
    )

    g_optim = optax.rmsprop(generator_lr)
    d_optim = optax.rmsprop(-discriminator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_inexact_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_inexact_array))

    infinite_dataloader = dataloader(
        (ts, ys), batch_size, loop=True, key=dataloader_key
    )

    tic = time.time()
    for step, (ts_i, ys_i) in zip(range(steps), infinite_dataloader):
        step = jnp.asarray(step)
        generator, discriminator, g_opt_state, d_opt_state = make_step(
            generator,
            discriminator,
            g_opt_state,
            d_opt_state,
            g_optim,
            d_optim,
            ts_i,
            ys_i,
            key,
            step,
        )
        if (step % steps_per_print) == 0 or step == steps - 1:
            total_score = 0
            num_batches = 0
            for ts_i, ys_i in dataloader(
                (ts, ys), batch_size, loop=False, key=evaluate_key
            ):
                score = loss(generator, discriminator, ts_i, ys_i, sample_key)
                total_score += score.item()
                num_batches += 1
            print(f"Step: {step}, Loss: {total_score / num_batches}")
    toc = time.time()

    data_file = "runtime.txt"
    with open(data_file, "a") as file:
        print(f"{adjoint}, runtime: {toc - tic}", file=file)

    # Save generator model
    eqx.tree_serialise_leaves(filename, generator)

    # Plot samples
    fig, ax = plt.subplots()
    num_samples = min(50, dataset_size)
    ts_to_plot = ts[:num_samples]
    ys_to_plot = ys[:num_samples]

    def _interp(ti, yi):
        return diffrax.linear_interpolation(
            ti, yi, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )

    ys_to_plot = jax.vmap(_interp)(ts_to_plot, ys_to_plot)[..., 0]
    ys_sampled = jax.vmap(generator)(ts_to_plot, key=jr.split(sample_key, num_samples))[
        ..., 0
    ]
    kwargs = dict(label="Real")
    for ti, yi in zip(ts_to_plot, ys_to_plot):
        ax.plot(ti, yi, c="dodgerblue", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    kwargs = dict(label="Generated")
    for ti, yi in zip(ts_to_plot, ys_sampled):
        ax.plot(ti, yi, c="crimson", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    ax.set_title(f"{num_samples} samples from both real and generated distributions.")
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"{filename}_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint", required=True)
    parser.add_argument("--checkpoints", required=True)
    script_args = parser.parse_args()

    adjoint_name = script_args.adjoint
    checkpoints = int(script_args.checkpoints)

    if adjoint_name == "reversible":
        adjoint = diffrax.ReversibleAdjoint()

    elif adjoint_name == "recursive":
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    filename = f"{adjoint_name}_{checkpoints}_new.eqx"
    main(adjoint, filename)
