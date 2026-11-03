"""EigenWorms dataset preprocessing for Neural RDE.

Downloads the EigenWorms dataset from the UEA/UCR Time Series Classification
archive. Preprocesses by computing log-signatures on a coarse uniform grid
for use with the Neural RDE model.
"""

import collections as co
import os
import pathlib
import ssl
import urllib.request
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import signax

from . import common

here = pathlib.Path(__file__).resolve().parent
DATA_DIR = here / ".." / ".." / "data" / "EigenWorms"

DOWNLOAD_URL = (
    "https://www.timeseriesclassification.com/aeon-toolkit/EigenWorms.zip"
)


def download():
    """Download and extract the EigenWorms dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "EigenWorms_TRAIN.ts"
    if os.path.exists(train_path):
        return
    loc = DATA_DIR / "EigenWorms.zip"
    if not os.path.exists(loc):
        print("Downloading EigenWorms.zip...")
        try:
            urllib.request.urlretrieve(DOWNLOAD_URL, str(loc))
        except urllib.error.URLError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            urllib.request.urlretrieve(
                DOWNLOAD_URL, str(loc), context=ctx,
            )
    print("Extracting...")
    with zipfile.ZipFile(str(loc), "r") as f:
        f.extractall(str(DATA_DIR))


def _parse_ts(filepath):
    """Parse a .ts file into (samples, labels).

    Each sample has n_dims dimensions, each of variable length.
    Returns a list of 2-d arrays (length x dims) and a list of string labels.
    """
    with open(filepath) as f:
        text = f.read()

    samples = []
    labels = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("@data"):
            in_data = True
            continue
        if line.startswith("@") or not in_data:
            continue
        parts = line.split(":")
        label = parts[-1].strip()
        dims = []
        for part in parts[:-1]:
            vals = [float(v) for v in part.split(",") if v.strip()]
            dims.append(vals)
        samples.append(dims)
        labels.append(label)
    return samples, labels


def _pad(channel, maxlen):
    """Pad a channel to maxlen with its last value."""
    arr = np.array(channel, dtype=np.float64)
    out = np.full(maxlen, arr[-1], dtype=np.float64)
    out[: len(arr)] = arr
    return out


def _compute_logsignatures(X, times, n_intervals, points_per_interval, depth):
    """Compute log-signatures over a coarse uniform grid.

    For each interval [s_j, s_{j+1}], queries the linear interpolation of the
    path at `points_per_interval` evenly spaced points, then computes the
    log-signature of that segment.

    Args:
        X: jnp array of shape (N, T, C) — the path data.
        times: jnp array of shape (T,) — original time points.
        n_intervals: number of intervals in the coarse grid.
        points_per_interval: number of query points per interval.
        depth: log-signature truncation depth.

    Returns:
        logsigs: jnp array of shape (N, n_intervals, L) where L is the
            log-signature dimension.
        interval_times: jnp array of shape (n_intervals + 1,).
    """
    interval_times = jnp.linspace(times[0], times[-1], n_intervals + 1)

    # Build query times for all intervals: (n_intervals, points_per_interval)
    # For each interval [s_j, s_{j+1}], linspace(s_j, s_{j+1}, K)
    starts = interval_times[:-1]
    ends = interval_times[1:]
    # (n_intervals, points_per_interval)
    t_fracs = jnp.linspace(0.0, 1.0, points_per_interval)
    query_times = starts[:, None] + t_fracs[None, :] * (ends - starts)[:, None]

    # Linearly interpolate each sample at query times.
    # X: (N, T, C), times: (T,)
    # For each sample and channel, use jnp.interp.
    def interp_sample(x):
        # x: (T, C)
        def interp_channel(channel):
            # channel: (T,) -> query all intervals at once
            return jnp.interp(query_times.ravel(), times, channel).reshape(
                n_intervals, points_per_interval
            )
        # (C, n_intervals, points_per_interval) -> (n_intervals, points_per_interval, C)
        return jax.vmap(interp_channel, in_axes=1, out_axes=2)(x)

    # (N, n_intervals, points_per_interval, C)
    segments = jax.vmap(interp_sample)(X)

    # Compute log-signature for each sample and interval.
    def logsig_segment(segment):
        # segment: (points_per_interval, C)
        return signax.logsignature(segment, depth=depth, flatten=True)

    # vmap over intervals, then over samples
    def logsig_sample(sample_segments):
        # sample_segments: (n_intervals, points_per_interval, C)
        return jax.vmap(logsig_segment)(sample_segments)

    # (N, n_intervals, L)
    logsigs = jax.vmap(logsig_sample)(segments)

    return logsigs, interval_times


def _process_data(step_size, depth, points_per_interval):
    """Load, concatenate train+test, preprocess, compute log-signatures."""
    train_path = DATA_DIR / "EigenWorms_TRAIN.ts"
    test_path = DATA_DIR / "EigenWorms_TEST.ts"

    train_samples, train_labels = _parse_ts(train_path)
    test_samples, test_labels = _parse_ts(test_path)

    # Concatenate train + test (we do our own 70/15/15 split)
    all_samples = train_samples + test_samples
    all_labels = train_labels + test_labels

    # Find max length across all samples and channels
    lengths = []
    for sample in all_samples:
        lengths.append(max(len(ch) for ch in sample))
    lengths = np.array(lengths)
    maxlen = int(lengths.max())

    # Derive number of coarse intervals from step_size
    n_intervals = maxlen // step_size

    # Build padded tensor: (batch, length, channel)
    X = np.stack(
        [
            np.stack([_pad(ch, maxlen) for ch in sample], axis=-1)
            for sample in all_samples
        ]
    )

    times = np.linspace(0, 1, X.shape[1])

    # Map labels to integers using first-occurrence order
    targets = co.OrderedDict()
    counter = 0
    for yi in all_labels:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    y = np.array([targets[yi] for yi in all_labels], dtype=np.int64)

    num_classes = counter
    assert num_classes >= 2, f"Have only {num_classes} classes."

    # Normalize using training set statistics
    X = common.normalise_data(X, y)

    # Append time channel: [time, data]
    time_channel = np.broadcast_to(times, (X.shape[0], len(times)))[..., None]
    X = np.concatenate([time_channel, X], axis=-1)

    in_channels = X.shape[-1]

    # Split 70/15/15
    train_X, val_X, test_X = common.split_data(X, y)
    train_y, val_y, test_y = common.split_data(y, y)

    # Extract X0 (first observation) before converting to JAX
    train_X0 = jnp.array(train_X[:, 0, :], dtype=jnp.float32)
    val_X0 = jnp.array(val_X[:, 0, :], dtype=jnp.float32)
    test_X0 = jnp.array(test_X[:, 0, :], dtype=jnp.float32)

    # Convert to JAX for log-signature computation
    times_jax = jnp.array(times, dtype=jnp.float32)

    if points_per_interval is None:
        points_per_interval = step_size

    print(f"Computing log-signatures: {n_intervals} intervals, "
          f"{points_per_interval} points/interval, depth {depth}")

    train_logsigs, interval_times = _compute_logsignatures(
        jnp.array(train_X, dtype=jnp.float32), times_jax,
        n_intervals, points_per_interval, depth,
    )
    val_logsigs, _ = _compute_logsignatures(
        jnp.array(val_X, dtype=jnp.float32), times_jax,
        n_intervals, points_per_interval, depth,
    )
    test_logsigs, _ = _compute_logsignatures(
        jnp.array(test_X, dtype=jnp.float32), times_jax,
        n_intervals, points_per_interval, depth,
    )

    logsig_size = train_logsigs.shape[-1]

    train_y = jnp.array(train_y)
    val_y = jnp.array(val_y)
    test_y = jnp.array(test_y)

    return (
        interval_times,
        train_logsigs, val_logsigs, test_logsigs,
        train_X0, val_X0, test_X0,
        train_y, val_y, test_y,
        in_channels, logsig_size, num_classes,
    )


def get_data(step_size=281, depth=2, points_per_interval=None):
    """Return preprocessed EigenWorms dataset with log-signatures.

    Args:
        step_size: number of raw timesteps per coarse interval (default 281,
            equivalent to ~64 intervals for the 17984-length series).
        depth: log-signature truncation depth (default 2).
        points_per_interval: query points per interval for interpolation.
            Defaults to step_size.

    Returns:
        (interval_times,
         train_logsigs, val_logsigs, test_logsigs,
         train_X0, val_X0, test_X0,
         train_y, val_y, test_y,
         in_channels, logsig_size, num_classes)
    """
    download()
    return _process_data(step_size, depth, points_per_interval)
