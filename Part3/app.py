import os
import re
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

import smtplib
from email.message import EmailMessage


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def topsis_calculate(input_file, weights_str, impacts_str, output_file):
    df = pd.read_csv(input_file)

    if df.shape[1] < 3:
        raise Exception("Input file must contain 3 or more columns")

    data = df.iloc[:, 1:].copy()

    # Convert to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data.isnull().values.any():
        raise Exception("From 2nd to last columns must contain numeric values only")

    # Split weights and impacts
    if "," not in weights_str or "," not in impacts_str:
        raise Exception("Weights and Impacts must be separated by ',' (comma)")

    weights = [w.strip() for w in weights_str.split(",")]
    impacts = [i.strip() for i in impacts_str.split(",")]

    if len(weights) != len(impacts):
        raise Exception("Number of weights must be equal to number of impacts")

    if len(weights) != data.shape[1]:
        raise Exception("Number of weights and impacts must match criteria columns")

    # Validate weights numeric
    try:
        weights = np.array([float(x) for x in weights])
    except:
        raise Exception("Weights must be numeric values")

    # Validate impacts +/-
    for i in impacts:
        if i not in ["+", "-"]:
            raise Exception("Impacts must be either + or -")

    impacts = np.array(impacts)

    # Step 1: Normalize
    norm_data = data / np.sqrt((data ** 2).sum())

    # Step 2: Multiply weights
    weighted_data = norm_data * weights

    # Step 3: Ideal Best/Worst
    ideal_best = np.zeros(weighted_data.shape[1])
    ideal_worst = np.zeros(weighted_data.shape[1])

    for j in range(weighted_data.shape[1]):
        if impacts[j] == "+":
            ideal_best[j] = weighted_data.iloc[:, j].max()
            ideal_worst[j] = weighted_data.iloc[:, j].min()
        else:
            ideal_best[j] = weighted_data.iloc[:, j].min()
            ideal_worst[j] = weighted_data.iloc[:, j].max()

    # Step 4: Distances
    dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Score
    score = dist_worst / (dist_best + dist_worst)

    # Step 6: Rank
    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    df.to_csv(output_file, index=False)


def send_email(receiver_email, attachment_path):
    # ✅ Use your Gmail App Password (NOT normal password)
    SENDER_EMAIL = "rsingh7_be23@thapar.edu"
    APP_PASSWORD = "qeha atat lroi zdzn"

    msg = EmailMessage()
    msg["Subject"] = "TOPSIS Result File"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg.set_content("Hello,\n\nYour TOPSIS result file is attached.\n\nThank you!")

    with open(attachment_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)

    msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        file = request.files.get("file")
        weights = request.form.get("weights", "").strip()
        impacts = request.form.get("impacts", "").strip()
        email = request.form.get("email", "").strip()

        # validations
        if file is None or file.filename == "":
            return render_template("index.html", error="Please upload a CSV file")

        if not validate_email(email):
            return render_template("index.html", error="Invalid Email format")

        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Only CSV files are allowed")

        # Save input file
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Output file name
        output_filename = "topsis_result.csv"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        # Run topsis
        topsis_calculate(input_path, weights, impacts, output_path)

        # Send output file via email
        send_email(email, output_path)

        return render_template("index.html", message="✅ Result generated and sent to your email successfully!")

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)