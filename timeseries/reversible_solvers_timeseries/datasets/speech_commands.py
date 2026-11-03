"""Speech Commands v0.02 dataset preprocessing.

Ported from the NeuralCDE reference implementation, using librosa instead of
torchaudio for MFCC computation.
"""

import os
import pathlib
import tarfile
import urllib.request

import numpy as np

from . import common

here = pathlib.Path(__file__).resolve().parent
DATA_DIR = here / ".." / ".." / "data" / "SpeechCommands"
CLASSES = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")


def download():
    """Download and extract Speech Commands v0.02."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    loc = DATA_DIR / "speech_commands_v0.02.tar.gz"
    if os.path.exists(loc):
        return
    print("Downloading Speech Commands v0.02 (~2.3GB)...")
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        str(loc),
    )
    print("Extracting...")
    with tarfile.open(str(loc), "r:gz") as f:
        f.extractall(str(DATA_DIR))


def _process_data(intensity_data):
    """Load WAV files, compute MFCC, run full preprocessing pipeline."""
    import librosa
    import scipy.io.wavfile

    all_mfcc = []
    all_labels = []

    for y_index, foldername in enumerate(CLASSES):
        loc = DATA_DIR / foldername
        kept = 0
        for filename in sorted(os.listdir(loc)):
            if not filename.endswith(".wav"):
                continue
            audio_sr, audio = scipy.io.wavfile.read(str(loc / filename))
            audio = audio / 2**15

            # Discard clips that aren't exactly 16000 samples
            if len(audio) != 16000:
                continue

            audio = audio.astype(np.float32)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=16000,
                n_mfcc=20,
                n_fft=200,
                hop_length=100,
                n_mels=64,
            )
            # mfcc shape: (20, 161), transpose to (161, 20)
            all_mfcc.append(mfcc.T)
            all_labels.append(y_index)
            kept += 1
        print(f"  {foldername}: {kept} clips")

    X = np.stack(all_mfcc).astype(np.float64)  # (N, 161, 20)
    y = np.array(all_labels, dtype=np.int64)
    print(f"Total: {len(y)} clips, shape {X.shape}")

    # Match NeuralCDE: times = linspace(0, 160, 161)
    times = np.linspace(0, X.shape[1] - 1, X.shape[1])
    final_index = np.full(X.shape[0], X.shape[1] - 1, dtype=np.int64)

    return common.preprocess_data(
        times, X, y, final_index,
        append_times=True,
        append_intensity=intensity_data,
    )


def get_data(intensity_data=False):
    """Return preprocessed Speech Commands splits.

    Returns:
        (times, train_ts, train_coeffs, val_ts, val_coeffs, test_ts, test_coeffs,
         train_y, val_y, test_y,
         train_final_index, val_final_index, test_final_index,
         in_channels)
    """

    download()
    result = _process_data(intensity_data)

    return result
