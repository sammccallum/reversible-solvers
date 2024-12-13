import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.tab10.colors)


def get_checkpoints_and_runtimes(data):
    checkpoint = data[:, 0]
    runtimes = data[:, 1:]
    mean_runtime = np.mean(runtimes, axis=1)
    std_runtime = np.std(runtimes, axis=1)

    return checkpoint, runtimes, mean_runtime, std_runtime


def plot_vs_checkpoints():
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


if __name__ == "__main__":
    steps = np.linspace(100, 1000, num=10, dtype=np.int64)
    runtimes_rec = np.zeros((len(steps), 4))
    runtimes_rev = np.zeros((len(steps), 1))

    for idx, step in enumerate(steps):
        recursive_step = np.loadtxt(f"../data/recursive_{step}.csv", delimiter=",")
        reversible_step = np.loadtxt(
            f"../data/reversible_{step}.csv", delimiter=","
        ).reshape(1, -1)
        checkpoints_rec, runtimes, mean_rec, std_rec = get_checkpoints_and_runtimes(
            recursive_step
        )
        checkpoints_rev, runtimes, mean_rev, std_rev = get_checkpoints_and_runtimes(
            reversible_step
        )
        runtimes_rec[idx] = mean_rec
        runtimes_rev[idx] = mean_rev

    plt.plot(steps, runtimes_rec, marker=".", alpha=0.8)
    plt.plot(steps, runtimes_rev, color="black", marker=".", linestyle="--")

    plt.ylabel("Runtime (s)")
    plt.xlabel("Computation length (n)")
    plt.yscale("log")
    plt.legend(
        labels=[
            r"Checkpointed $n=2$",
            r"Checkpointed $n=4$",
            r"Checkpointed $n=8$",
            r"Optimal checkpointing",
            "Reversible",
        ],
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)
