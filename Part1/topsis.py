import sys
import pandas as pd
import numpy as np


def exit_with_msg(message):
    print(f"Error: {message}")
    sys.exit(1)


def parse_list(text, name="value"):
    if "," not in text:
        exit_with_msg(f"{name} must be separated by ',' (comma)")
    return [x.strip() for x in text.split(",")]


def validate_numeric_criteria(df_criteria):
    for col in df_criteria.columns:
        df_criteria[col] = pd.to_numeric(df_criteria[col], errors="coerce")

    if df_criteria.isnull().values.any():
        exit_with_msg("From 2nd to last columns must contain numeric values only")

    return df_criteria


def normalize_matrix(matrix):
    denom = np.sqrt((matrix ** 2).sum(axis=0))
    return matrix / denom


def topsis_process(input_path, weights_str, impacts_str, output_path):
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        exit_with_msg("Input file not found")
    except Exception as e:
        exit_with_msg(f"Unable to read input file: {e}")

    if df.shape[1] < 3:
        exit_with_msg("Input file must contain three or more columns")

    obj_col = df.columns[0]
    criteria_df = df.iloc[:, 1:].copy()

    # Check numeric
    criteria_df = validate_numeric_criteria(criteria_df)

    # Parse weights & impacts 
    weights_list = parse_list(weights_str, "Weights")
    impacts_list = parse_list(impacts_str, "Impacts")

    # Validate counts 
    if len(weights_list) != len(impacts_list):
        exit_with_msg("Number of weights must be equal to number of impacts")

    if len(weights_list) != criteria_df.shape[1]:
        exit_with_msg("Number of weights, impacts and criteria columns must be same")

    # Validate weights numeric 
    try:
        weights = np.array([float(w) for w in weights_list])
    except:
        exit_with_msg("Weights must be numeric values")

    # Validate impacts 
    impacts = np.array(impacts_list)
    if not np.all(np.isin(impacts, ["+", "-"])):
        exit_with_msg("Impacts must be either + or -")

    # TOPSIS
    mat = criteria_df.to_numpy(dtype=float)

    # Step-1: Normalize
    norm_mat = normalize_matrix(mat)

    # Step-2: Apply weights
    weighted_mat = norm_mat * weights

    # Step-3: Ideal Best/Worst
    ideal_best = np.zeros(weighted_mat.shape[1])
    ideal_worst = np.zeros(weighted_mat.shape[1])

    for j in range(weighted_mat.shape[1]):
        if impacts[j] == "+":
            ideal_best[j] = weighted_mat[:, j].max()
            ideal_worst[j] = weighted_mat[:, j].min()
        else:
            ideal_best[j] = weighted_mat[:, j].min()
            ideal_worst[j] = weighted_mat[:, j].max()

    # Step-4: Distances
    dist_pos = np.sqrt(((weighted_mat - ideal_best) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((weighted_mat - ideal_worst) ** 2).sum(axis=1))

    # Step-5: Score
    score = dist_neg / (dist_pos + dist_neg)

    # Step-6: Rank (dense ranking)
    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    # Save output
    df.to_csv(output_path, index=False)
    print(f" TOPSIS completed! Result saved in: {output_path}")


if __name__ == "__main__":
    # Assignment requires: python topsis.py input.csv "1,1,1,2" "+,+,-,+" output.csv
    if len(sys.argv) != 5:
        exit_with_msg("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <OutputResultFileName>")

    input_file = sys.argv[1]
    weights_arg = sys.argv[2]
    impacts_arg = sys.argv[3]
    output_file = sys.argv[4]

    topsis_process(input_file, weights_arg, impacts_arg, output_file)