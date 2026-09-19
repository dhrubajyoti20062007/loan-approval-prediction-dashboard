# Loan Approval Analytics & Prediction System

An AI-powered data analytics project that studies what drives loan approval decisions. It cleans the data, explores it, calculates KPIs, writes business-intelligence insights, trains a simple prediction model and shows everything in an interactive Streamlit executive dashboard.

The whole solution (backend + frontend) is in **one file: `Project.py`**.

## Problem statement

Which applicant and loan characteristics drive approval or rejection, what risks and opportunities does this reveal, and can the decision be predicted from application data?

## Features

- **Data cleaning**: fixes text formatting, number formats, duplicates, impossible values and blanks, checks outliers, and logs every step
- **Exploratory data analysis**: summary statistics, distributions, correlation matrix, approval rate by segment, chi-square significance tests
- **KPIs**: approval rate, average loan and income, loan-to-income ratio, approved loan volume, credit-history split and more
- **Charts**: bar, line, pie, histogram, box plot, scatter plot and heat-maps
- **Business intelligence**: executive overview, KPI summary, trend analysis, risk analysis, opportunity analysis and prioritised recommendations, all generated from the data
- **Prediction model**: Logistic Regression vs Random Forest, cross-validation, confusion matrix, permutation feature importance, and a form to predict a single applicant
- **Dashboard**: 9 tabs with sidebar filters (property area, employment, education), CSV upload, and downloads (cleaned data, manual-review candidate list)

## Dataset

| Item | Details |
|---|---|
| File | `train.csv` (provided inside `archive.zip`) |
| Size | 3,192 rows x 14 columns |
| Target | `Loan_Status` (Approved / Rejected) |
| Columns | Loan_ID, Gender, Married, Dependents, Education, Employment_Status, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Term, Credit_History, Property_Area, Age, Loan_Status |
| Source | *Add the link or citation of where the dataset was downloaded from.* |

Notes about the data:

- Classes are perfectly balanced (50% approved), so the approval rate reflects how the sample was built, not a real-world rate.
- About 0.8% of cells are blank (Gender, Married, Education, Employment_Status, Applicant_Income, Loan_Amount).
- Currency, income period and the unit of `Loan_Term` are not documented. `Loan_Term` is assumed to be in months.
- The data contains decisions only, with no repayment outcomes.

## Key findings

- Credit history decides almost everything: about 96.7% of applicants with a good history are approved versus about 0.1% without one.
- Self-employed applicants are approved less often (45.2% vs 51.9% for salaried), even among applicants with good credit.
- Income, loan amount, age, loan term, gender and marital status show no meaningful effect.
- The model reaches about 97.8% test accuracy, but only because of `Credit_History`. Without it the ROC-AUC falls to about 0.55 (0.50 = random guessing). Use it as decision support, not as a final decision-maker.

## Project structure

```
.
├── Project.py            # Complete backend + Streamlit frontend (single file)
├── requirements.txt      # Python libraries with versions
├── README.md             # This file
├── Project_Report.docx   # Full project report
└── train.csv             # Dataset (place it here, see below)
```

## Installation

1. Install **Python 3.9 to 3.12**.
2. Put `train.csv` (or `archive.zip`) in the same folder as `Project.py`.
3. Open a terminal in that folder and create a virtual environment (recommended):

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

4. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

## How to run

```bash
streamlit run Project.py
```

Your browser opens the dashboard at `http://localhost:8501`. (`python Project.py` also works: it starts Streamlit for you.)

If the dataset is not in the project folder, upload a CSV with the same columns from the sidebar.

## How the code is organised

`Project.py` is written top to bottom in seven commented sections:

1. Imports and settings
2. Data loading and cleaning
3. KPIs and statistics
4. Charts
5. Machine-learning model
6. Business-intelligence text
7. Streamlit user interface

## Model summary

- Split: 80% training / 20% testing (stratified), plus 5-fold cross-validation
- Models compared: Logistic Regression and Random Forest; the best by ROC-AUC is used
- Features: applicant and loan details plus a loan-to-income ratio (`Loan_ID` is not used)
- Metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix

## Limitations

- No dates, so "trends" are read across ordered variables (loan term, age group, income band) rather than over time.
- No repayment data, so credit risk itself cannot be measured.
- Results come from one sampled dataset and should be validated before real-world use.

## Report

See `Project_Report.docx` for the full write-up: methodology, cleaning, EDA, KPIs, charts, model, dashboard mockups, results, business insights and conclusion.
