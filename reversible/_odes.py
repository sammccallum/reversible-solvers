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
