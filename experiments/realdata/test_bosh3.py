import diffrax as dfx
import jax.numpy as jnp
import matplotlib.pyplot as plt

from reversible import lotka_volterra


def solve(vf, y0, t1, dt0, solver, adjoint, args, stepsize_controller):
    saveat = dfx.SaveAt(steps=True)
    term = dfx.ODETerm(vf)
    t0 = 0
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        max_steps=1000,
        stepsize_controller=stepsize_controller,
    )
    ts = sol.ts
    ys = sol.ys
    print(sol.stats["num_accepted_steps"])

    return ts, ys


args = (0.5, 0.05, 2.0, 0.05)
solver = dfx.Bosh3Simple()
ts, ys = solve(
    lotka_volterra,
    y0=jnp.array([10, 10]),
    t1=10,
    dt0=0.1,
    solver=solver,
    adjoint=dfx.RecursiveCheckpointAdjoint(),
    args=args,
    stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
)
plt.plot(ts, ys)
plt.savefig("test_lotka_volterra.png", dpi=300)
