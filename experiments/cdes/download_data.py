import os
import pathlib
import tarfile
import urllib.request

import jax.numpy as jnp
import numpy as np
import torch
import torchaudio

here = pathlib.Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def download():
    base_base_loc = here / "data"
    base_loc = base_base_loc / "SpeechCommands"
    loc = base_loc / "speech_commands.tar.gz"
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", loc
    )
    with tarfile.open(loc, "r") as f:
        f.extractall(base_loc)


def _process_data():
    base_loc = here / "data" / "SpeechCommands"
    X = torch.empty(34975, 16000, 1)
    y = torch.empty(34975, dtype=torch.long)

    batch_index = 0
    y_index = 0
    for foldername in (
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ):
        loc = base_loc / foldername
        for filename in os.listdir(loc):
            audio, _ = torchaudio.load(
                loc / filename, channels_first=False, normalize=False
            )  # for forward compatbility if they fix it
            audio = (
                audio / 2**15
            )  # Normalization argument doesn't seem to work so we do it manually.

            # A few samples are shorter than the full length; for simplicity we discard them.
            if len(audio) != 16000:
                continue

            X[batch_index] = audio
            y[batch_index] = y_index
            batch_index += 1
        y_index += 1
    assert batch_index == 34975, "batch_index is {}".format(batch_index)

    X = (
        torchaudio.transforms.MFCC(
            log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
        )(X.squeeze(-1))
        .transpose(1, 2)
        .detach()
    )
    # X is of shape (batch=34975, length=161, channels=20)

    ts = torch.linspace(0, X.size(1) - 1, X.size(1))
    np.save("ts", ts)
    np.save("Xs", X)
    np.save("ys", y)
    return jnp.array(ts), jnp.array(X), jnp.array(y)


if __name__ == "__main__":
    _process_data()
