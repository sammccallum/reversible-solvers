import matplotlib.pyplot as plt


def plot_SIR(ts, ys):
    plt.plot(ts, ys[:, 0], ".-", color="tab:blue", label="S")
    plt.plot(ts, ys[:, 1], ".-", color="tab:red", label="I")
    plt.plot(ts, ys[:, 2], ".-", color="tab:green", label="R")

    plt.legend()
    plt.savefig("data/sir.png", dpi=300)


def plot_lotka_volterra(ts, ys):
    plt.plot(ts, ys[:, 0], ".-", color="tab:blue", label=r"x")
    plt.plot(ts, ys[:, 1], ".-", color="tab:red", label=r"y")

    plt.legend()
    plt.savefig("data/lotka_volterra.png", dpi=300)


def plot_pendulum(ts, ys):
    plt.plot(ts, ys[:, 0], ".-", color="tab:blue", label=r"x")
    plt.plot(ts, ys[:, 1], ".-", color="tab:red", label=r"y")

    plt.legend()
    plt.savefig("data/pendulum_pred.png", dpi=300)
