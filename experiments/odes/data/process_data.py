import numpy as np


def calculate_stats(start_index, data):
    adjoint_data = data[start_index::7]
    runtimes = adjoint_data[:, 1] / 60
    return np.mean(runtimes), np.std(runtimes)


data = np.loadtxt("adjoint_results.csv", delimiter=",")
for i in range(7):
    print(calculate_stats(i, data))
