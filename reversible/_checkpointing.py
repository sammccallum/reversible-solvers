import math


def calculate_checkpoints(n_steps):
    return math.floor(-1.5 + math.sqrt(2 * n_steps + 0.25)) + 1
