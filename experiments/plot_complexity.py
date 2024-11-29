import matplotlib.pyplot as plt
import numpy as np


def get_checkpoints_and_runtimes(data):
    checkpoint = data[:, 0]
    runtimes = data[:, 1:]
    mean_runtime = np.mean(runtimes, axis=1)
    std_runtime = np.std(runtimes, axis=1)

    return checkpoint, runtimes, mean_runtime, std_runtime


if __name__ == "__main__":
    reversible = np.loadtxt(
        "../data/results/complexity/reversible_complexity.csv", delimiter=","
    ).reshape(1, -1)
    recursive = np.loadtxt(
        "../data/results/complexity/recursive_complexity.csv", delimiter=","
    )
    checkpoint_rev, runtimes_rev, mean_rev, std_rev = get_checkpoints_and_runtimes(
        reversible
    )

    checkpoints_rec, runtimes_rec, mean_rec, std_rec = get_checkpoints_and_runtimes(
        recursive
    )

    plt.hlines(
        y=mean_rev,
        xmin=checkpoints_rec[0],
        xmax=checkpoints_rec[-1] + 0.5,
        linestyles="--",
        color="black",
        label="Reversible algorithm",
    )
    plt.plot(
        checkpoints_rec,
        mean_rec,
        marker=".",
        color="tab:red",
        label="Binomial checkpointing",
    )
    plt.ylabel("Runtime (s)")
    plt.xlabel("Checkpoints (n)")
    plt.yscale("log")
    plt.legend()
    plt.savefig("../data/results/complexity/plot_complexity.png", dpi=300)
