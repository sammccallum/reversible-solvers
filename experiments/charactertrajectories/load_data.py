import numpy as np
from sktime.datasets import load_UCR_UEA_dataset

if __name__ == "__main__":
    X, y = load_UCR_UEA_dataset("CharacterTrajectories", return_type="pd-multiindex")
    dataset_length, max_ts_length = X.index.max()
    Xs = []
    for i in range(dataset_length + 1):
        Xi = X.loc[i]
        Xs.append(Xi.to_numpy())

        if Xi.shape[0] > max_ts_length:
            max_ts_length = Xi.shape[0]

    Xs_padded = []
    for X in Xs:
        X_padded = np.pad(
            X, ((0, max_ts_length - X.shape[0]), (0, 0)), constant_values=0
        )
        Xs_padded.append(X_padded)

    Xs_padded = np.array(Xs_padded)

    np.save("data/character_trajectories.npy", Xs_padded)
    np.save("data/labels.npy", y)
