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
        loss = float(loss_match.group(1)) * 1000 if loss_match else None

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


def format_latex_table(data):
    column_names = [
        "Method",
        "\shortstack{Runtime\\\(min)}",
        "\shortstack{Memory\\\(checkpoints)}",
    ]
    latex_table = "\\begin{table}\n\\centering\n\\begin{tabular}{l|c c}\n"

    header_row = " & ".join(column_names) + " \\\\ \\hline\n"
    latex_table += header_row

    for row in data:
        if row[0] == -1:
            adjoint = "Reversible"
            checkpoints = 2
        else:
            adjoint = "Recursive"
            checkpoints = row[0]

        row_values = " & ".join(
            [
                f"{adjoint}",  # Method
                f"${row[1]:.2f} \pm {row[2]:.2f}$",  # Runtime
                f"${checkpoints}$",
            ]
        )
        latex_table += row_values + " \\\ \n"

    latex_table += "\end{tabular}\n\end{table}"

    return latex_table


if __name__ == "__main__":
    data = load_data("white_dwarf_RK3.txt")
    results = calculate_mean_std(data)
    latex_table_str = format_latex_table(results)
    print(latex_table_str)
