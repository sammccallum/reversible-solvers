import jax.numpy as jnp


def SIR(t, y, args):
    beta, gamma = args
    S = y[0]
    I = y[1]
    R = y[2]
    dyS = -beta * I * S
    dyI = beta * I * S - gamma * I
    dyR = gamma * I
    return jnp.array([dyS, dyI, dyR])


def lotka_volterra(t, y, args):
    alpha, beta, gamma, delta = args
    y1 = y[0]
    y2 = y[1]
    dy1 = alpha * y1 - beta * y1 * y2
    dy2 = -gamma * y2 + delta * y1 * y2
    return jnp.array([dy1, dy2])


def lorenz(t, y, args):
    sigma, rho, beta = args
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    dy1 = sigma * (y2 - y1)
    dy2 = y1 * (rho - y3) - y2
    dy3 = y1 * y2 - beta * y3
    return jnp.array([dy1, dy2, dy3])


def pendulum(t, y, args):
    g, l = args
    y1 = y[0]  # theta
    y2 = y[1]  # dtheta/dt
    dy1 = y2
    dy2 = -(g / l) * jnp.sin(y1)
    return jnp.array([dy1, dy2])
