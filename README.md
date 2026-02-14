# TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

This project implements the **TOPSIS** multi-criteria decision-making technique to rank alternatives based on multiple conflicting criteria.

The assignment is implemented as:

- A **command-line tool** (Part-I)
- A **Python package uploaded on PyPI** (Part-II)
- A **Flask web service** that emails the output file (Part-III)

---

## 1. Methodology

The TOPSIS method ranks alternatives based on their distance from an **ideal best** and an **ideal worst** solution.

### Workflow

Input CSV File
↓
Validation of Inputs
↓
Normalization of Decision Matrix
↓
Weighted Normalization
↓
Ideal Best & Ideal Worst
↓
Distance Calculation
↓
TOPSIS Score Computation
↓
Ranking of Alternatives

### Explanation of Steps

 **Input Validation**

- Checks file existence
- Ensures all criteria columns are numeric
- Validates weights & impacts format

 **Normalization**

- Removes scale differences among criteria values

 **Weight Application**

- Assigns importance to each criterion using user-defined weights

 **Ideal Solutions**

- Positive impact (+): Higher value is preferred
- Negative impact (-): Lower value is preferred

 **Distance Calculation**

- Calculates Euclidean distance from ideal best and ideal worst

 **Scoring & Ranking**

- Higher TOPSIS score → Better rank

---

## 2. Description

- **Input Type:** CSV File
- **Decision Criteria:** Multiple numerical columns
- **Weights:** User-defined (comma separated)
- **Impacts:** `+` or `-` for each criterion
- **Output:** Ranked alternatives based on TOPSIS score

---

## 3. Input / Output Format

### Input File Format (Example)

| Fund Name | P1   | P2   | P3  | P4    |
| --------- | ---- | ---- | --- | ----- |
| M1        | 0.67 | 0.45 | 6.5 | 12.56 |
| M2        | 0.60 | 0.38 | 6.3 | 14.47 |
| M3        | 0.82 | 0.67 | 3.8 | 17.10 |

### Output File Format (Example)

| Fund Name | P1   | P2   | P3  | P4    | TOPSIS Score | Rank |
| --------- | ---- | ---- | --- | ----- | ------------ | ---- |
| M3        | 0.82 | 0.67 | 3.8 | 17.10 | 1.0000       | 1    |
| M5        | 0.68 | 0.55 | 5.2 | 15.25 | 0.5415       | 2    |
| M4        | 0.75 | 0.50 | 4.5 | 13.80 | 0.4711       | 3    |

---

## 4. Result Analysis

- Alternatives with higher TOPSIS score are ranked better.
- This method balances benefit (+) and cost (-) type criteria.
- Ranking helps decision-makers select the most suitable alternative.

---

# PART-I (Command Line Program)

### Run Command

python topsis.py `<InputDataFile>` `<Weights>` `<Impacts>` `<OutputResultFileName>`

PART-II (PyPI Package)
PyPI Package Name

 Topsis-Kanishka-102317165

Install from PyPI

pip install Topsis-Kanishka-102317165

Run using CLI command
topsis data.csv "1,1,1,2" "+,+,-,+" output.csv

PyPI Link

https://pypi.org/project/Topsis-Kanishka-102317165

 PART-III (Web Service)

This project also provides a web service where the user can:

Upload the input CSV file

Provide weights and impacts

Enter email ID

Receive the output CSV via email

Run Flask Server
python app.py

Then open:

link given
