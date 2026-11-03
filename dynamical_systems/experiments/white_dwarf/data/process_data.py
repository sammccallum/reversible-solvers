import re

import numpy as np


def load_data(file_path):
    with open(file_path, "r") as file:
        raw_data = file.readlines()

    data_parsed = []
    for line in raw_data:
        checkpoints_match = re.search(r"checkpoints=(\d+)", line)
        adjoint_match = re.search(r"ReversibleAdjoint\(l=.*?\)", line)
        runtime_match = re.search(r"runtime: ([\d.]+)", line)
        loss_match = re.search(r"loss: ([\d.]+)", line)

        if adjoint_match:
            checkpoints = -1
        elif checkpoints_match:
            checkpoints = int(checkpoints_match.group(1))

        runtime = float(runtime_match.group(1)) / 60 if runtime_match else None
        loss = float(loss_match.group(1)) * 10000 if loss_match else None

        data_parsed.append((checkpoints, runtime, loss))

    dtype = [("checkpoints", int), ("runtime", float), ("loss", float)]
    data_np = np.array(data_parsed, dtype=dtype)

    return data_np


def calculate_mean_std(data_np):
    unique_checkpoints = np.unique(data_np["checkpoints"])

    results = []

    for checkpoint in unique_checkpoints:
        mask = data_np["checkpoints"] == checkpoint
        filtered_data = data_np[mask]

        mean_runtime = np.mean(filtered_data["runtime"])
        std_runtime = np.std(filtered_data["runtime"])
        mean_loss = np.mean(filtered_data["loss"])
        std_loss = np.std(filtered_data["loss"])

        results.append((checkpoint, mean_runtime, std_runtime, mean_loss, std_loss))

    results_dtype = [
        ("checkpoints", int),
        ("mean_runtime", float),
        ("std_runtime", float),
        ("mean_loss", float),
        ("std_loss", float),
    ]
    results_np = np.array(results, dtype=results_dtype)

    return results_np


def format_latex_table(paths):
    datas = [load_data(path) for path in paths]
    results = [calculate_mean_std(data) for data in datas]
    latex_table = ""

    for i in range(len(results[0])):
        latex_table += f" & ${results[0][i][0]}$ &" + " & ".join(
            [
                f"  ${result[i][3]:.2f} \pm {result[i][4]:.2f}$ & ${result[i][1]:.2f} \pm {result[i][2]:.2f}$"
                for result in results
            ]
        )
        latex_table += " \\\ \n"
    return latex_table


if __name__ == "__main__":
    base_path = "white_dwarf_"
    add_paths = ["euler.txt", "midpoint.txt", "Ralston3.txt", "RK4.txt"]
    paths = [base_path + path for path in add_paths]
    latex_table_str = format_latex_table(paths)
    print(latex_table_str)
