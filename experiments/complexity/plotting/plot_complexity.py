import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Roman",
        "font.size": 18,
        "text.latex.preamble": r"\usepackage{lmodern, amsmath, amssymb, amsfonts}",
        "legend.fontsize": 16,
        "svg.fonttype": "none",
        "axes.prop_cycle": plt.cycler(color=plt.cm.tab10.colors),
    }
)


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

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(
        steps, runtimes_rev, color="black", marker="D", markersize=5, linestyle="--"
    )
    ax.text(
        steps[-1] + 15,
        runtimes_rev[-1],
        r"Reversible",
        color="black",
        va="center",
    )

    markers = ["p", "o", "^", "s"]
    colors = ["tab:purple", "tab:red", "tab:blue", "tab:orange"]
    labels = [r"$c\sim \sqrt{n}$", "$c=2$", r"$c=4$", r"$c=8$"]
    text_offsets = [0, 0, 0, 0]
    for i in range(4):
        ax.plot(steps, runtimes_rec[:, i], marker=markers[i], color=colors[i])
        ax.text(
            steps[-1] + 15,
            runtimes_rec[-1, i] + text_offsets[i],
            labels[i],
            color=colors[i],
            va="center",
        )

    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Time steps (n)")
    plt.yscale("log")
    ax.set_xlim(steps[0], steps[-1] + 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("complexity.pdf", format="pdf", bbox_inches="tight")
