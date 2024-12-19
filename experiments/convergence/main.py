import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

jax.config.update("jax_enable_x64", True)
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


def vector_field(t, y, args):
    return 0.5 * y


def true_solution(t):
    return jnp.exp(0.5 * t)


def solve(t1, dt0, solver):
    term = dfx.ODETerm(vector_field)
    t0 = 0
    y0 = jnp.array([1.0])
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    y1_base = sol.ys[-1][0]
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        adjoint=dfx.ReversibleAdjoint(),
    )
    y1_rev = sol.ys[-1][0]

    return y1_base, y1_rev


def calculate_order(dts, errors):
    coefs, cov = jnp.polyfit(jnp.log(dts), jnp.log(jnp.array(errors)), 1, cov=True)
    order = coefs[0]
    std = jnp.sqrt(jnp.diag(cov))[0]
    return order, std


def plot_order(t1, solver, solver_name, ax, marker, color):
    y1_true = true_solution(t1)
    dts = jnp.logspace(0, -5, num=5, base=2)
    errors_base = []
    errors_rev = []
    for dt0 in dts:
        y1_base, y1_rev = solve(t1, dt0, solver)
        errors_base.append(jnp.abs(y1_true - y1_base))
        errors_rev.append(jnp.abs(y1_true - y1_rev))

    ax.plot(dts, errors_base, marker=marker, color=color, label=rf"{solver_name}")
    ax.plot(
        dts,
        errors_rev,
        marker=marker,
        linestyle="--",
        color=color,
        label=rf"Reversible {solver_name}",
    )


if __name__ == "__main__":
    t1 = 5
    solvers = [dfx.Euler(), dfx.Midpoint(), dfx.Bosh3(), dfx.Dopri5()]
    solver_names = ["Euler", "Midpoint", "Bosh3", "Dopri5"]
    markers = ["o", "^", "s", "p"]
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:purple"]
    custom_handles = []
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for i in range(len(solvers)):
        solver = solvers[i]
        name = solver_names[i]
        marker = markers[i]
        color = colors[i]
        plot_order(t1, solver, name, ax, marker, color)

        custom_handles.append(
            Line2D(
                [i],
                [i],
                marker=markers[i],
                label=rf"{name}",
                color=(0, 0, 0, 0),
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
            )
        )
    plt.loglog()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"Time step, $\Delta t$")
    ax.set_ylabel(r"Error, $|y_N - y(T)|$")
    plt.grid(True)
    plt.legend(handles=custom_handles, handletextpad=0.2)
    # plt.legend(bbox_to_anchor=(1, 0.85), loc="upper left")
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=300)
