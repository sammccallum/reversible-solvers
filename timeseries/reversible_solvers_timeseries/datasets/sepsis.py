"""PhysioNet 2019 Sepsis dataset preprocessing.

Ported from the NeuralCDE reference implementation. Key differences from the
previous inline version:
- Static features (age, gender, unit1, unit2, hospadmtime) are handled
  separately, not treated as time-varying
- NaN gaps in ICULOS are filled with NaN rows (matching reference)
- unit1/unit2 observation indicators tracked for static_intensity
- ICULOS excluded from time-varying features
"""

import csv
import math
import os
import pathlib

import jax.numpy as jnp
import numpy as np

from . import common

here = pathlib.Path(__file__).resolve().parent
DATA_DIR = here / ".." / ".." / "data" / "sepsis"


# This function requires data to be downloaded from https://physionet.org/files/challenge-2019/1.0.0/
# Note that this function is VERY important for reproducing the original NeuralCDE results.
# `os.listdir` loads data as an unordered list dependent on the underlying OS and file structure.
# This becomes a problem as Patrick's original train/val/test split is 70/15/15, where the first
# 70% depends on the order in which `os.listdir` loaded the files. We try to follow Patrick's
# dataloading as closely as possible by first unpacking the .psv into one directory which is then
# loaded by `os.listdir`.

# There is no guarantee that we are using the same train/test/split. This is not too much of
# a problem as we run every model here from scratch but should be noted for cross comparison.


def _unpack():
    """Move .psv files from training_setA/ and training_setB/ into DATA_DIR."""
    for folder in ("training_setA", "training_setB"):
        folder_path = DATA_DIR / folder
        if folder_path.exists():
            for filename in os.listdir(folder_path):
                if filename.endswith(".psv"):
                    source_path = folder_path / filename
                    dest_path = DATA_DIR / filename

                    if os.path.exists(dest_path):
                        raise RuntimeError(f"File already exists: {dest_path}")

                    os.rename(source_path, dest_path)


def _process_data(static_intensity, time_intensity):
    """Parse .psv files and run full preprocessing pipeline.

    Matches NeuralCDE reference: 34 time-varying features (no ICULOS),
    static features handled separately, NaN gap filling.
    """
    X_times = []
    X_static = []
    y = []

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".psv"):
            continue
        with open(DATA_DIR / filename) as file:
            time = []
            label = 0.0
            reader = csv.reader(file, delimiter="|")
            reader = iter(reader)
            header = next(reader)
            assert len(header) == 41, f"Expected 41 columns, got {len(header)} in {filename}"
            prev_iculos = 0
            for line in reader:
                assert len(line) == 41
                # First 34 columns are time-varying features
                # Then: Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS, SepsisLabel
                *time_values, age, gender, unit1, unit2, hospadmtime, iculos, sepsislabel = line
                iculos = int(iculos)
                if iculos > 72:  # keep at most the first three days
                    break
                # Fill missing hour gaps with NaN rows
                for _ in range(prev_iculos + 1, iculos):
                    time.append([float("nan") for _ in time_values])
                prev_iculos = iculos
                time.append([float(value) for value in time_values])
                label = max(label, float(sepsislabel))

            # Process static features
            unit1 = float(unit1)
            unit2 = float(unit2)
            unit1_obs = not math.isnan(unit1)
            unit2_obs = not math.isnan(unit2)
            if not unit1_obs:
                unit1 = 0.0
            if not unit2_obs:
                unit2 = 0.0
            hospadmtime = float(hospadmtime)
            if math.isnan(hospadmtime):
                hospadmtime = 0.0  # only happens for one record
            static = [float(age), float(gender), unit1, unit2, hospadmtime]
            if static_intensity:
                static += [float(unit1_obs), float(unit2_obs)]

            if len(time) > 2:
                X_times.append(time)
                X_static.append(static)
                y.append(label)

    print(f"Loaded {len(y)} patients (after filtering <=2 observations)")

    # Compute final indices and pad to max length
    final_indices = [len(t) - 1 for t in X_times]
    maxlen = max(final_indices) + 1
    n_time_features = len(X_times[0][0])
    for t in X_times:
        for _ in range(maxlen - len(t)):
            t.append([float("nan")] * n_time_features)

    X_times = np.array(X_times, dtype=np.float64)
    X_static = np.array(X_static, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    final_indices = np.array(final_indices, dtype=np.int64)

    # Match NeuralCDE: times = linspace(1, maxlen, maxlen) (1-indexed)
    times = np.linspace(1, X_times.shape[1], X_times.shape[1])

    cache_dir = DATA_DIR / "cache" / (
        ("intensity" if time_intensity else "nointensity")
        + ("_static" if static_intensity else "_nostatic")
    )
    result = common.preprocess_data(
        times, X_times, y, final_indices,
        append_times=True,
        append_intensity=time_intensity,
        cache_dir=cache_dir,
    )

    # Normalize static features separately (matching NeuralCDE)
    if static_intensity:
        X_static_features = X_static[:, :-2]
        X_static_features = common.normalise_data(
            X_static_features.reshape(X_static_features.shape[0], 1, -1),
            y,
        ).reshape(X_static_features.shape)
        X_static = np.concatenate(
            [X_static_features, X_static[:, -2:]], axis=1
        )
    else:
        X_static = common.normalise_data(
            X_static.reshape(X_static.shape[0], 1, -1), y
        ).reshape(X_static.shape)

    train_X_static, val_X_static, test_X_static = common.split_data(
        X_static, y
    )

    # Convert static to JAX and append to result
    train_X_static = jnp.array(train_X_static, dtype=jnp.float32)
    val_X_static = jnp.array(val_X_static, dtype=jnp.float32)
    test_X_static = jnp.array(test_X_static, dtype=jnp.float32)

    return result, train_X_static, val_X_static, test_X_static


def get_data(static_intensity=True, time_intensity=True):
    """Return preprocessed Sepsis splits.

    Returns:
        (times, train_ts, train_coeffs, val_ts, val_coeffs, test_ts, test_coeffs,
         train_y, val_y, test_y,
         train_final_index, val_final_index, test_final_index,
         in_channels,
         train_static, val_static, test_static)
    """
    # Unpack .psv files from subdirectories if not already done
    if not any(f.endswith(".psv") for f in os.listdir(DATA_DIR)):
        _unpack()
    result, train_static, val_static, test_static = _process_data(
        static_intensity, time_intensity
    )
    return (*result, train_static, val_static, test_static)
