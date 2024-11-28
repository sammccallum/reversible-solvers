def SIR(t, y, args):
    beta, gamma = args
    S, I, R = y
    dyS = -beta * I * S
    dyI = beta * I * S - gamma * I
    dyR = gamma * I
    return (dyS, dyI, dyR)
