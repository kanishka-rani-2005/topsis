import pandas as pd
import numpy as np


class TopsisError(Exception):
    pass


def _to_float_list(text: str, label: str):
    parts = [x.strip() for x in text.split(",")]
    if len(parts) == 1:
        raise TopsisError(f"{label} must be comma-separated")
    try:
        return [float(x) for x in parts]
    except:
        raise TopsisError(f"{label} must contain numeric values only")


def _to_impact_list(text: str):
    impacts = [x.strip() for x in text.split(",")]
    if len(impacts) == 1:
        raise TopsisError("Impacts must be comma-separated")
    for i in impacts:
        if i not in ["+", "-"]:
            raise TopsisError("Impacts must be either + or -")
    return impacts


def run_topsis(input_file, weights, impacts, output_file):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        raise TopsisError("Input file not found")

    if df.shape[1] < 3:
        raise TopsisError("Input file must contain 3 or more columns")

    data = df.iloc[:, 1:].copy()

    # Convert criteria columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data.isnull().values.any():
        raise TopsisError("From 2nd to last columns must contain numeric values only")

    w = _to_float_list(weights, "Weights")
    imp = _to_impact_list(impacts)

    if len(w) != data.shape[1]:
        raise TopsisError("Number of weights must match number of criteria columns")

    if len(imp) != data.shape[1]:
        raise TopsisError("Number of impacts must match number of criteria columns")

    # Step 1: Normalize
    norm = data / np.sqrt((data ** 2).sum())

    # Step 2: Weighting
    weighted = norm * w

    # Step 3: Ideal best/worst
    ideal_best = []
    ideal_worst = []

    for j in range(len(imp)):
        if imp[j] == "+":
            ideal_best.append(weighted.iloc[:, j].max())
            ideal_worst.append(weighted.iloc[:, j].min())
        else:
            ideal_best.append(weighted.iloc[:, j].min())
            ideal_worst.append(weighted.iloc[:, j].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Step 4: Distances
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Score
    score = d_worst / (d_best + d_worst)

    # Step 6: Rank
    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    df.to_csv(output_file, index=False)
    return output_file