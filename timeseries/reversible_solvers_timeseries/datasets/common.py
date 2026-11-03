"""Shared preprocessing utilities matching NeuralCDE's preprocessing pipeline.

Ported from the NeuralCDE reference implementation (Kidger 2020), adapted from
PyTorch/natural-cubic-splines to numpy/JAX/Hermite interpolation.
"""

import pathlib

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import sklearn.model_selection
import tqdm


def split_data(array, stratify):
    """70/15/15 train/val/test split matching NeuralCDE exactly.

    Uses sklearn.model_selection.train_test_split with the same random_state
    values as the reference implementation.
    """
    train_array, testval_array, train_stratify, testval_stratify = (
        sklearn.model_selection.train_test_split(
            array,
            stratify,
            train_size=0.7,
            random_state=0,
            shuffle=True,
            stratify=stratify,
        )
    )
    val_array, test_array = sklearn.model_selection.train_test_split(
        testval_array,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_array, val_array, test_array


def normalise_data(X, y):
    """Per-channel normalization using training set statistics only.

    Matches NeuralCDE: for each channel, compute mean/std from training data
    (masking NaN), then normalize all data as (x - mean) / (std + 1e-5).
    """
    train_X, _, _ = split_data(X, y)
    out = []
    for i in range(X.shape[-1]):
        Xi = X[..., i]
        train_Xi = train_X[..., i]
        train_Xi_nonan = train_Xi[~np.isnan(train_Xi)]
        mean = train_Xi_nonan.mean()
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    return np.stack(out, axis=-1)


def _interpolate_nans(X, times):
    """Interpolate NaN values using diffrax backward Hermite, per sample and channel.

    For each scalar feature, drops NaN values, computes backward Hermite
    coefficients on the observed points, then evaluates the cubic interpolation
    at all original time points. Boundary NaN are imputed with the nearest
    observed value before fitting (avoids extrapolation artifacts).

    Args:
        X: numpy array of shape (N, T, C)
        times: numpy array of shape (T,)
    Returns:
        Array with NaN replaced by interpolated values.
    """
    X = X.copy()
    N, T, C = X.shape

    for n in tqdm.tqdm(range(N), desc="Interpolating NaN"):
        for c in range(C):
            series = X[n, :, c]
            observed = ~np.isnan(series)
            n_observed = observed.sum()

            if n_observed == 0:
                X[n, :, c] = 0.0
            elif n_observed == T:
                continue
            elif n_observed == 1:
                X[n, :, c] = series[observed][0]
            else:
                # Boundary imputation (matching NeuralCDE reference)
                if not observed[0]:
                    series[0] = series[observed][0]
                    observed[0] = True
                if not observed[-1]:
                    series[-1] = series[observed][-1]
                    observed[-1] = True

                t_obs = jnp.array(times[observed], dtype=jnp.float32)
                v_obs = jnp.array(series[observed], dtype=jnp.float32)[:, None]

                coeffs = diffrax.backward_hermite_coefficients(t_obs, v_obs)
                interp = diffrax.CubicInterpolation(t_obs, coeffs)

                times_jax = jnp.array(times, dtype=jnp.float32)
                filled = jax.vmap(interp.evaluate)(times_jax)
                X[n, :, c] = np.array(filled[:, 0])

    return X


def preprocess_data(times, X, y, final_index, append_times, append_intensity, cache_dir=None):
    """Full preprocessing pipeline matching NeuralCDE.

    1. Normalize via normalise_data
    2. Augment channels in order: [time, intensity (optional), data]
    3. Split X, y, final_index via split_data
    4. Interpolate NaN per sample/channel using backward Hermite coefficients
    5. Compute diffrax backward_hermite_coefficients per split

    Args:
        times: 1-D numpy array of time points, shape (T,)
        X: numpy array of shape (N, T, C)
        y: numpy array of shape (N,) — labels (used for stratified split)
        final_index: numpy array of shape (N,) — last valid time index per sample
        append_times: bool — whether to prepend a time channel
        append_intensity: bool — whether to prepend intensity channels
        cache_dir: optional path — directory to cache interpolated arrays

    Returns:
        (times_jax, train_coeffs, val_coeffs, test_coeffs,
         train_y, val_y, test_y,
         train_final_index, val_final_index, test_final_index,
         in_channels)
    """
    X = normalise_data(X, y)

    # Augment channels in NeuralCDE order: time, intensity, original data
    augmented = []
    if append_times:
        time_channel = np.broadcast_to(times, (X.shape[0], len(times)))[..., None]
        augmented.append(time_channel)
    if append_intensity:
        intensity = (~np.isnan(X)).astype(np.float64)
        intensity = np.cumsum(intensity, axis=1)
        augmented.append(intensity)
    augmented.append(X)
    if len(augmented) == 1:
        X = augmented[0]
    else:
        X = np.concatenate(augmented, axis=-1)

    train_X, val_X, test_X = split_data(X, y)
    train_y, val_y, test_y = split_data(y, y)
    train_final_index, val_final_index, test_final_index = split_data(
        final_index, y
    )

    # Interpolate NaN using backward Hermite (per-sample, per-channel)
    if cache_dir is not None:
        interp_dir = pathlib.Path(cache_dir)
        interp_dir.mkdir(parents=True, exist_ok=True)
        train_path = interp_dir / "interp_train.npy"
        val_path = interp_dir / "interp_val.npy"
        test_path = interp_dir / "interp_test.npy"

        if train_path.exists() and val_path.exists() and test_path.exists():
            train_X = np.load(train_path)
            val_X = np.load(val_path)
            test_X = np.load(test_path)
        else:
            train_X = _interpolate_nans(train_X, times)
            val_X = _interpolate_nans(val_X, times)
            test_X = _interpolate_nans(test_X, times)
            np.save(train_path, train_X)
            np.save(val_path, val_X)
            np.save(test_path, test_X)
    else:
        train_X = _interpolate_nans(train_X, times)
        val_X = _interpolate_nans(val_X, times)
        test_X = _interpolate_nans(test_X, times)

    in_channels = X.shape[-1]

    # Convert to JAX and compute Hermite coefficients
    times_jax = jnp.array(times, dtype=jnp.float32)

    def _to_jax_coeffs(data):
        data_jax = jnp.array(data, dtype=jnp.float32)
        ts_batch = jnp.broadcast_to(times_jax, (data_jax.shape[0], len(times_jax)))
        coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts_batch, data_jax)
        return ts_batch, coeffs

    train_ts, train_coeffs = _to_jax_coeffs(train_X)
    val_ts, val_coeffs = _to_jax_coeffs(val_X)
    test_ts, test_coeffs = _to_jax_coeffs(test_X)

    train_y = jnp.array(train_y)
    val_y = jnp.array(val_y)
    test_y = jnp.array(test_y)
    train_final_index = jnp.array(train_final_index)
    val_final_index = jnp.array(val_final_index)
    test_final_index = jnp.array(test_final_index)

    return (
        times_jax,
        train_ts, train_coeffs,
        val_ts, val_coeffs,
        test_ts, test_coeffs,
        train_y, val_y, test_y,
        train_final_index, val_final_index, test_final_index,
        in_channels,
    )
