import matplotlib.pyplot as plt


def plot_SIR(ts, ys):
    plt.plot(ts, ys[:, 0], ".-", color="tab:blue", label="S")
    plt.plot(ts, ys[:, 1], ".-", color="tab:red", label="I")
    plt.plot(ts, ys[:, 2], ".-", color="tab:green", label="R")

    plt.legend()
    plt.savefig("../../data/imgs/sir_pred.png", dpi=300)
