import os
import re
import threading
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ---------------------------
# Email Validation
# ---------------------------
def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


# ---------------------------
# TOPSIS Calculation
# ---------------------------
def topsis_calculate(input_file, weights_str, impacts_str, output_file):
    df = pd.read_csv(input_file)

    if df.shape[1] < 3:
        raise Exception("Input file must contain 3 or more columns")

    data = df.iloc[:, 1:].copy()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data.isnull().values.any():
        raise Exception("All criteria columns must contain numeric values.")

    weights = [w.strip() for w in weights_str.split(",")]
    impacts = [i.strip() for i in impacts_str.split(",")]

    if len(weights) != len(impacts):
        raise Exception("Weights and impacts count mismatch.")

    if len(weights) != data.shape[1]:
        raise Exception("Weights/impacts must match criteria columns.")

    try:
        weights = np.array([float(x) for x in weights])
    except:
        raise Exception("Weights must be numeric.")

    for i in impacts:
        if i not in ["+", "-"]:
            raise Exception("Impacts must be '+' or '-'.")

    impacts = np.array(impacts)

    # Normalize
    norm_data = data / np.sqrt((data ** 2).sum())

    # Weight
    weighted_data = norm_data * weights

    # Ideal best & worst
    ideal_best = []
    ideal_worst = []

    for j in range(weighted_data.shape[1]):
        if impacts[j] == "+":
            ideal_best.append(weighted_data.iloc[:, j].max())
            ideal_worst.append(weighted_data.iloc[:, j].min())
        else:
            ideal_best.append(weighted_data.iloc[:, j].min())
            ideal_worst.append(weighted_data.iloc[:, j].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Distances
    dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)

    df["Topsis Score"] = score
    df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

    df.to_csv(output_file, index=False)


# ---------------------------
# Email Sending (Async Safe)
# ---------------------------
def send_email(receiver_email, attachment_path):
    try:
        SENDER_EMAIL = os.getenv("EMAIL")
        APP_PASSWORD = os.getenv("EMAIL_PASSWORD")

        if not SENDER_EMAIL or not APP_PASSWORD:
            print("Email credentials not configured.")
            return

        domain = SENDER_EMAIL.split("@")[-1].lower()

        msg = EmailMessage()
        msg["Subject"] = "TOPSIS Result File"
        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email
        msg.set_content("Your TOPSIS result file is attached.")

        with open(attachment_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="octet-stream",
                filename=os.path.basename(attachment_path)
            )

        # ----------------------------
        # SMTP Selection Based on Domain
        # ----------------------------

        if domain == "gmail.com":
            smtp_server = "smtp.gmail.com"
            port = 465
            use_ssl = True

        elif domain in ["outlook.com", "hotmail.com", "live.com", "office365.com", "thapar.edu"]:
            smtp_server = "smtp.office365.com"
            port = 587
            use_ssl = False

        else:
            # Default fallback (try Gmail style SSL)
            smtp_server = "smtp.gmail.com"
            port = 465
            use_ssl = True

        # ----------------------------
        # Connect and Send
        # ----------------------------

        if use_ssl:
            with smtplib.SMTP_SSL(smtp_server, port, timeout=20) as smtp:
                smtp.login(SENDER_EMAIL, APP_PASSWORD)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, port, timeout=20) as smtp:
                smtp.starttls()
                smtp.login(SENDER_EMAIL, APP_PASSWORD)
                smtp.send_message(msg)

        print("Email sent successfully via", smtp_server)

    except Exception as e:
        print("Email failed:", str(e))


def send_email_async(email, path):
    thread = threading.Thread(target=send_email, args=(email, path))
    thread.start()


# ---------------------------
# Routes
# ---------------------------
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

        if not file or file.filename == "":
            return render_template("index.html", error="Please upload a CSV file.")

        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Only CSV files allowed.")

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        output_path = os.path.join(RESULT_FOLDER, "Result.csv")

        topsis_calculate(input_path, weights, impacts, output_path)

        # Optional Email (Non-blocking)
        if email and validate_email(email):
            send_email_async(email, output_path)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
