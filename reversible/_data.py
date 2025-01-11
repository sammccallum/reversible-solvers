import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def split_data(tensor, stratify):
    (train_tensor, test_tensor, train_stratify, test_stratify) = train_test_split(
        tensor,
        stratify,
        train_size=0.8,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    return train_tensor, test_tensor, train_stratify, test_stratify


def normalise_data(X, y):
    X = torch.tensor(X)
    y = torch.tensor(y)
    train_X, _, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    normalised_X = torch.stack(out, dim=-1)
    return np.array(normalised_X)


def include_time(ts, Xs):
    ts = ts.reshape((len(ts), 1))
    ts = np.repeat(ts[np.newaxis, :, :], Xs.shape[0], axis=0)
    aug_Xs = jnp.concatenate([ts, Xs], axis=2)
    return aug_Xs


def load_from_numpy():
    ts = np.load("ts.npy")
    Xs = np.load("Xs.npy")
    ys = np.load("ys.npy")
    Xs = normalise_data(Xs, ys)
    Xs = include_time(ts, Xs)
    train_X, test_X, train_y, test_y = split_data(Xs, ys)
    train_coeffs = jax.vmap(dfx.backward_hermite_coefficients, in_axes=(None, 0))(
        ts, train_X
    )
    test_coeffs = jax.vmap(dfx.backward_hermite_coefficients, in_axes=(None, 0))(
        ts, test_X
    )

    return ts, train_coeffs, test_coeffs, train_y, test_y
