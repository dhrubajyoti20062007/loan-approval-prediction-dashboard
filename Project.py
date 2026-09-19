"""
=====================================================================
 Loan Approval Analytics & Prediction System
 An AI-powered Data Analytics project (single-file: backend + UI)
=====================================================================

WHAT THIS FILE DOES
  1. Loads the loan dataset (train.csv, or archive.zip that contains it)
  2. Cleans it (missing values, duplicates, wrong / impossible values)
  3. Runs Exploratory Data Analysis (EDA) and computes business KPIs
  4. Builds charts (bar, line, pie, histogram, box, heat-map, ...)
  5. Trains a simple machine-learning model that predicts loan approval
  6. Writes business insights: overview, trends, risks, opportunities,
     and recommended actions
  7. Shows everything in an interactive Streamlit executive dashboard

HOW TO RUN
  pip install -r requirements.txt
  streamlit run Project.py

FILE LAYOUT (top to bottom)
  Section 1  - Imports & settings
  Section 2  - Data loading & cleaning
  Section 3  - KPIs & statistics
  Section 4  - Charts
  Section 5  - Machine-learning model
  Section 6  - Business-intelligence text (insights)
  Section 7  - Streamlit user interface
"""

# ---------------------------------------------------------------------
# SECTION 1: IMPORTS & SETTINGS
# ---------------------------------------------------------------------
import os
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # draw charts in memory (no pop-up windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---- Project settings -------------------------------------------------
PROJECT_TITLE = "Loan Approval Analytics & Prediction System"
DATA_FILE_CANDIDATES = ["train.csv", "archive.zip"]  # files we look for

ID_COL = "Loan_ID"
TARGET = "Loan_Status"  # "Approved" / "Rejected"
POSITIVE_LABEL = "Approved"

NUMERIC_COLS = ["Applicant_Income", "Coapplicant_Income", "Loan_Amount", "Age"]
DISCRETE_COLS = ["Dependents", "Loan_Term", "Credit_History"]  # whole-number codes
CATEGORICAL_COLS = ["Gender", "Married", "Education", "Employment_Status", "Property_Area"]
REQUIRED_COLS = [ID_COL, TARGET] + NUMERIC_COLS + DISCRETE_COLS + CATEGORICAL_COLS

# Columns that the machine-learning model is allowed to use
MODEL_NUMERIC = NUMERIC_COLS + DISCRETE_COLS + ["Loan_to_Income_Ratio"]
MODEL_CATEGORICAL = CATEGORICAL_COLS

# Columns used to compare approval rates between groups
SEGMENT_COLS = [
    "Credit_History", "Employment_Status", "Education", "Property_Area", "Gender",
    "Married", "Dependents", "Loan_Term", "Age_Group", "Income_Band", "Loan_to_Income_Band",
]

# Fixed display order for the "band" columns we create
AGE_ORDER = ["18-25", "26-35", "36-45", "46+"]
INCOME_ORDER = ["Low", "Lower-Mid", "Upper-Mid", "High"]
LTI_ORDER = ["Low", "Moderate", "High", "Very High"]
BAND_ORDERS = {"Age_Group": AGE_ORDER, "Income_Band": INCOME_ORDER, "Loan_to_Income_Band": LTI_ORDER}

# Friendly names for coded values
VALUE_LABELS = {"Credit_History": {0: "No / poor history", 1: "Good history"}}

# Colours used in every chart
C_APPROVED = "#1B9E77"
C_REJECTED = "#D95F02"
C_NEUTRAL = "#4C72B0"
STATUS_PALETTE = {"Approved": C_APPROVED, "Rejected": C_REJECTED}


# ---------------------------------------------------------------------
# SECTION 2: DATA LOADING & CLEANING
# ---------------------------------------------------------------------
def find_default_data():
    """Look for train.csv (or archive.zip) next to this file or in the current folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in DATA_FILE_CANDIDATES:
        for folder in (here, os.getcwd()):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
    return None


def load_data(source):
    """Read the dataset from a file path or an uploaded file and check the columns."""
    raw = pd.read_csv(source)  # also works for a .zip that holds a single CSV
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {missing}")
    return raw


def add_features(df):
    """Create helpful new columns. Also reused when predicting for a single applicant."""
    df = df.copy()
    df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
    df["Loan_to_Income_Ratio"] = df["Loan_Amount"] / df["Total_Income"].replace(0, np.nan)
    df["Has_Coapplicant"] = np.where(df["Coapplicant_Income"] > 0, "Yes", "No")
    return df


MISSING_TOKENS = ["", " ", "?", "NA", "N/A", "na", "n/a", "NaN", "nan", "None", "null", "NULL", "-"]


def clean_data(raw):
    """
    Clean the raw dataset.
    Returns (clean_dataframe, report_dictionary). The report describes every step
    so it can be shown in the dashboard and in the project report.
    """
    df = raw.copy()
    steps = []  # a log of every cleaning action

    def log(step, count, action):
        steps.append({"Step": step, "Cells / rows affected": int(count), "Action taken": action})

    missing_before = raw.isna().sum()
    missing_before = missing_before[missing_before > 0].sort_values(ascending=False)

    # 1) Text columns: remove extra spaces, unify capitalisation, turn "?" / "NA" into blanks
    text_cols = [c for c in df.columns if c not in NUMERIC_COLS + DISCRETE_COLS]
    na_before = int(df[text_cols].isna().sum().sum())
    for col in text_cols:
        s = df[col].astype("string").str.strip()
        s = s.mask(s.isin(MISSING_TOKENS))
        s = s.str.upper() if col == ID_COL else s.str.title()
        df[col] = s.astype(object).where(s.notna(), np.nan)
    hidden_missing = int(df[text_cols].isna().sum().sum()) - na_before
    log("Standardise text", hidden_missing,
        "Trimmed spaces, unified capitalisation and converted placeholders such as '?' or 'NA' to blanks")

    # 2) Numeric columns: force them to numbers (bad text becomes blank)
    numeric_all = NUMERIC_COLS + DISCRETE_COLS
    na_before = int(df[numeric_all].isna().sum().sum())
    for col in numeric_all:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    bad_numbers = int(df[numeric_all].isna().sum().sum()) - na_before
    log("Fix number formats", bad_numbers, "Converted non-numeric entries in numeric columns to blanks")

    # 3) Duplicates: identical rows, then repeated loan IDs
    full_dups = int(df.duplicated().sum())
    df = df.drop_duplicates()
    id_dups = int(df.duplicated(subset=ID_COL).sum())
    df = df.drop_duplicates(subset=ID_COL, keep="first")
    log("Remove duplicates", full_dups + id_dups, "Dropped identical rows and repeated Loan_ID values")

    # 4) Impossible values -> blanks (they are filled in step 6)
    rules = {
        "Applicant_Income": lambda s: s <= 0,
        "Coapplicant_Income": lambda s: s < 0,
        "Loan_Amount": lambda s: s <= 0,
        "Age": lambda s: (s < 18) | (s > 100),
        "Dependents": lambda s: s < 0,
        "Loan_Term": lambda s: s <= 0,
        "Credit_History": lambda s: (~s.isin([0, 1])) & s.notna(),
    }
    invalid_total = 0
    for col, is_bad in rules.items():
        mask = is_bad(df[col])
        invalid_total += int(mask.sum())
        df.loc[mask, col] = np.nan
    log("Fix impossible values", invalid_total,
        "Set negative / zero incomes, loan amounts, ages outside 18-100 and invalid credit codes to blank")

    # 5) The target must be known - rows without a decision cannot be used
    valid_target = df[TARGET].isin(["Approved", "Rejected"])
    dropped = int((~valid_target).sum())
    df = df[valid_target].copy()
    log("Check target", dropped, "Removed rows whose Loan_Status is missing or not Approved / Rejected")

    # 6) Fill missing values: median for numbers, most frequent value for categories
    filled_numeric = 0
    for col in NUMERIC_COLS:
        n = int(df[col].isna().sum())
        if n and df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())
            filled_numeric += n
    log("Fill missing numbers", filled_numeric, "Replaced blanks in income / amount / age with the column median")

    filled_category = 0
    for col in DISCRETE_COLS + CATEGORICAL_COLS:
        n = int(df[col].isna().sum())
        if n and df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            filled_category += n
    log("Fill missing categories", filled_category,
        "Replaced blanks in category / code columns with the most frequent value")

    for col in DISCRETE_COLS:
        df[col] = df[col].astype(int)

    # 7) Tidy precision (values had 10+ decimals)
    for col in ["Applicant_Income", "Coapplicant_Income", "Loan_Amount"]:
        df[col] = df[col].round(2)
    df["Age"] = df["Age"].round(1)
    log("Round values", len(df), "Rounded money columns to 2 decimals and Age to 1 decimal")

    # 8) Outlier check (IQR rule). We only REPORT them - large incomes/loans are plausible.
    outlier_rows = []
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < low) | (df[col] > high)).sum())
        outlier_rows.append({
            "Column": col, "Lower bound": round(low, 1), "Upper bound": round(high, 1),
            "Outliers": count, "Share of rows (%)": round(count / len(df) * 100, 2),
        })
    outliers = pd.DataFrame(outlier_rows)
    log("Outlier check", outliers["Outliers"].sum(),
        "Flagged with the 1.5 x IQR rule but kept, because high incomes / loans are realistic")

    # 9) New columns for analysis
    df = add_features(df)
    df["Approved"] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    df["Age_Group"] = pd.cut(np.floor(df["Age"]), bins=[0, 25, 35, 45, 200], labels=AGE_ORDER).astype(str)
    df["Income_Band"] = pd.qcut(df["Total_Income"], 4, labels=INCOME_ORDER).astype(str)
    df["Loan_to_Income_Band"] = pd.qcut(df["Loan_to_Income_Ratio"], 4, labels=LTI_ORDER).astype(str)
    log("Create features", 6,
        "Added Total_Income, Loan_to_Income_Ratio, Has_Coapplicant, Age_Group, Income_Band, Loan_to_Income_Band")

    df = df.reset_index(drop=True)
    report = {
        "raw_rows": len(raw),
        "raw_cols": raw.shape[1],
        "clean_rows": len(df),
        "clean_cols": df.shape[1],
        "missing_before": missing_before,
        "missing_cells_before": int(raw.isna().sum().sum()),
        "missing_pct_before": float(raw.isna().sum().sum() / raw.size * 100),
        "missing_cells_after": int(df.isna().sum().sum()),
        "steps": pd.DataFrame(steps),
        "outliers": outliers,
    }
    return df, report


# ---------------------------------------------------------------------
# SECTION 3: KPIs & STATISTICS
# ---------------------------------------------------------------------
def money(x):
    """Format a number with thousands separators (currency is not stated in the data)."""
    return "n/a" if pd.isna(x) else f"{x:,.0f}"


def pct(x, decimals=1):
    """Format a number as a percentage string."""
    return "n/a" if pd.isna(x) else f"{x:.{decimals}f}%"


def rate(df, mask):
    """Approval rate (%) for the rows selected by `mask`."""
    subset = df[mask]
    return subset["Approved"].mean() * 100 if len(subset) else np.nan


def compute_kpis(df):
    """The most important numbers for an executive summary."""
    approved = df[df["Approved"] == 1]
    return {
        "Total applications": len(df),
        "Approved": int(df["Approved"].sum()),
        "Rejected": int((1 - df["Approved"]).sum()),
        "Approval rate (%)": df["Approved"].mean() * 100,
        "Average loan amount": df["Loan_Amount"].mean(),
        "Average applicant income": df["Applicant_Income"].mean(),
        "Average household income": df["Total_Income"].mean(),
        "Average loan-to-income ratio": df["Loan_to_Income_Ratio"].mean(),
        "Approved loan volume": approved["Loan_Amount"].sum(),
        "Share with good credit history (%)": (df["Credit_History"] == 1).mean() * 100,
        "Approval rate - good credit (%)": rate(df, df["Credit_History"] == 1),
        "Approval rate - poor credit (%)": rate(df, df["Credit_History"] == 0),
        "Average applicant age": df["Age"].mean(),
    }


def approval_by(df, col, order=None):
    """Approval rate for every group in `col` (e.g. every Property_Area)."""
    g = df.groupby(col)["Approved"].agg(["count", "sum", "mean"]).reset_index()
    g.columns = [col, "Applications", "Approved", "Approval_Rate"]
    g["Approval_Rate"] = g["Approval_Rate"] * 100
    order = order or BAND_ORDERS.get(col)
    if order:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
    g = g.sort_values(col).reset_index(drop=True)
    g[col] = g[col].astype(object)
    return g


def association_tests(df):
    """
    Chi-square test: is approval statistically related to each column?
    p-value < 0.05 means the difference between groups is unlikely to be chance.
    """
    rows = []
    for col in SEGMENT_COLS:
        if col not in df.columns:
            continue
        table = pd.crosstab(df[col], df["Approved"])
        seg = approval_by(df, col)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_value, _, _ = chi2_contingency(table)
        rows.append({
            "Feature": col,
            "Chi-square": round(chi2, 2),
            "p-value": p_value,
            "Significant (p<0.05)": "Yes" if p_value < 0.05 else "No",
            "Approval gap (points)": round(seg["Approval_Rate"].max() - seg["Approval_Rate"].min(), 1),
        })
    result = pd.DataFrame(rows)
    return result.sort_values("Approval gap (points)", ascending=False).reset_index(drop=True) if len(result) else result


def review_candidates(df):
    """
    Applicants with NO good credit history but above-median household income and
    below-median loan-to-income ratio. They are rejected today, yet look financially
    strong, so they are good candidates for a manual review pilot.
    """
    if df.empty:
        return df
    mask = (
        (df["Credit_History"] == 0)
        & (df["Total_Income"] >= df["Total_Income"].median())
        & (df["Loan_to_Income_Ratio"] <= df["Loan_to_Income_Ratio"].median())
    )
    cols = [ID_COL, "Employment_Status", "Property_Area", "Total_Income", "Loan_Amount",
            "Loan_to_Income_Ratio", "Loan_Term", TARGET]
    return df.loc[mask, cols].sort_values("Loan_to_Income_Ratio").reset_index(drop=True)


# ---------------------------------------------------------------------
# SECTION 4: CHARTS  (every function returns a matplotlib figure)
# Each chart accepts an optional `ax` so it can be placed inside a bigger figure.
# ---------------------------------------------------------------------
def _get_axes(ax, size=(6, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = ax.figure
    return fig, ax


def _tick_labels(col, values):
    labels = VALUE_LABELS.get(col)
    out = []
    for v in values:
        if labels:
            out.append(labels.get(int(v), str(v)))
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return out


def chart_status_pie(df, ax=None):
    """PIE chart: share of approved vs rejected applications."""
    fig, ax = _get_axes(ax, (4.5, 4))
    counts = df[TARGET].value_counts().reindex(["Approved", "Rejected"]).fillna(0)
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=[C_APPROVED, C_REJECTED], wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Loan decisions")
    return fig


def chart_segment_bar(df, col, title=None, ax=None, order=None):
    """BAR chart: approval rate for each group of `col`, with the overall average as a dashed line."""
    fig, ax = _get_axes(ax)
    seg = approval_by(df, col, order)
    labels = _tick_labels(col, seg[col])
    bars = ax.bar(labels, seg["Approval_Rate"], color=C_NEUTRAL, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
    overall = df["Approved"].mean() * 100
    ax.axhline(overall, color="grey", linestyle="--", linewidth=1, label=f"Overall {overall:.1f}%")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, min(118, max(70, seg["Approval_Rate"].max() * 1.3)))
    ax.set_ylabel("Approval rate (%)")
    ax.set_xlabel(col.replace("_", " "))
    ax.set_title(title or f"Approval rate by {col.replace('_', ' ').lower()}")
    if len(labels) > 3:
        ax.tick_params(axis="x", labelsize=8)
    return fig


def chart_trend_line(df, col, title=None, ax=None, order=None):
    """LINE chart: how the approval rate moves across an ordered variable (e.g. loan term)."""
    fig, ax = _get_axes(ax)
    seg = approval_by(df, col, order)
    labels = _tick_labels(col, seg[col])
    ax.plot(labels, seg["Approval_Rate"], marker="o", color=C_NEUTRAL, linewidth=2)
    for x, y in zip(labels, seg["Approval_Rate"]):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    overall = df["Approved"].mean() * 100
    ax.axhline(overall, color="grey", linestyle="--", linewidth=1, label=f"Overall {overall:.1f}%")
    ax.set_ylim(30, 70)  # zoomed in so small differences are visible - read the labels!
    ax.set_ylabel("Approval rate (%)")
    ax.set_xlabel(col.replace("_", " "))
    ax.set_title(title or f"Approval rate by {col.replace('_', ' ').lower()}")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def chart_histogram(df, col, title=None, ax=None):
    """HISTOGRAM: distribution of a number, split by decision."""
    fig, ax = _get_axes(ax)
    sns.histplot(data=df, x=col, hue=TARGET, bins=30, element="step", alpha=0.35,
                 palette=STATUS_PALETTE, ax=ax)
    ax.set_xlabel(col.replace("_", " "))
    ax.set_title(title or f"Distribution of {col.replace('_', ' ').lower()}")
    return fig


def chart_box(df, col, title=None, ax=None):
    """BOX plot: compare a number between approved and rejected applications."""
    fig, ax = _get_axes(ax)
    sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=STATUS_PALETTE, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(col.replace("_", " "))
    ax.set_title(title or f"{col.replace('_', ' ')} by decision")
    return fig


def chart_scatter(df, ax=None):
    """SCATTER plot: applicant income vs loan amount, coloured by decision."""
    fig, ax = _get_axes(ax)
    sample = df.sample(min(len(df), 1200), random_state=1)
    sns.scatterplot(data=sample, x="Applicant_Income", y="Loan_Amount", hue=TARGET,
                    palette=STATUS_PALETTE, alpha=0.5, s=18, ax=ax)
    ax.set_title("Applicant income vs loan amount")
    ax.set_xlabel("Applicant income")
    ax.set_ylabel("Loan amount")
    return fig


def chart_correlation(df, ax=None):
    """HEAT-MAP: correlation between the numeric columns and approval."""
    fig, ax = _get_axes(ax, (6, 5))
    cols = NUMERIC_COLS + ["Loan_Term", "Dependents", "Credit_History", "Loan_to_Income_Ratio", "Approved"]
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, annot_kws={"size": 7}, cbar=False, ax=ax)
    ax.set_title("Correlation matrix")
    ax.tick_params(axis="both", labelsize=8)
    return fig


def chart_missing(missing_before, total_rows, ax=None):
    """BAR chart: how many values were missing in each column before cleaning."""
    fig, ax = _get_axes(ax)
    if len(missing_before) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
        return fig
    bars = ax.barh(missing_before.index[::-1], missing_before.values[::-1], color=C_REJECTED)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values before cleaning")
    return fig


def chart_confusion(cm, ax=None):
    """HEAT-MAP: model predictions vs real decisions on unseen test data."""
    fig, ax = _get_axes(ax, (4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test data)")
    return fig


def chart_importance(importance, title="Feature importance", ax=None):
    """BAR chart: which inputs matter most to the model."""
    fig, ax = _get_axes(ax, (6, 4))
    top = importance.head(10)[::-1]
    ax.barh(top.index, top.values * 100, color=C_NEUTRAL)
    ax.set_xlabel("Importance (%)")
    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------
# SECTION 5: MACHINE-LEARNING MODEL
# ---------------------------------------------------------------------
def build_pipeline(model, numeric_cols, categorical_cols):
    """Preparation (fill blanks, scale numbers, one-hot encode categories) + a model."""
    numeric_steps = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    category_steps = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([("num", numeric_steps, numeric_cols), ("cat", category_steps, categorical_cols)])
    return Pipeline([("prep", prep), ("model", model)])


def train_models(df, drop_features=()):
    """
    Train two models, compare them on a hidden test set and keep the best one.
    `drop_features` lets us test the model WITHOUT some columns (e.g. Credit_History).
    """
    num_cols = [c for c in MODEL_NUMERIC if c not in drop_features]
    cat_cols = [c for c in MODEL_CATEGORICAL if c not in drop_features]
    features = num_cols + cat_cols
    X, y = df[features], df["Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
        fitted[name] = (pipe, pred)

    results = pd.DataFrame(rows).sort_values(["ROC-AUC", "F1-score"], ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best_pipe, best_pred = fitted[best_name]

    # 5-fold cross-validation gives a more stable accuracy estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clone(best_pipe), X, y, cv=cv, scoring="accuracy")

    # Feature importance = how much the test ROC-AUC drops when one column is shuffled.
    # (Permutation importance works the same way for every model and for every original column.)
    perm = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc",
                                  n_repeats=10, random_state=42)
    importance = pd.Series(perm.importances_mean, index=features).clip(lower=0)
    total = importance.sum()
    importance = (importance / total if total > 0 else importance + 1 / len(importance))
    importance = importance.sort_values(ascending=False)

    return {
        "results": results,
        "best_name": best_name,
        "pipeline": best_pipe,
        "features": features,
        "confusion": confusion_matrix(y_test, best_pred),
        "importance": importance,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dropped": list(drop_features),
    }


def predict_applicant(model_info, applicant):
    """Return the model's probability (0-1) that a single applicant is approved."""
    row = add_features(pd.DataFrame([applicant]))
    return float(model_info["pipeline"].predict_proba(row[model_info["features"]])[0, 1])


# ---------------------------------------------------------------------
# SECTION 6: BUSINESS-INTELLIGENCE TEXT
# Every sentence is built from the (filtered) data, so the numbers always match the charts.
# ---------------------------------------------------------------------
def _spread_sentence(df, col, label):
    seg = approval_by(df, col)
    if len(seg) < 2:
        return None
    hi = seg.loc[seg["Approval_Rate"].idxmax()]
    lo = seg.loc[seg["Approval_Rate"].idxmin()]
    hi_name, lo_name = _tick_labels(col, [hi[col]])[0], _tick_labels(col, [lo[col]])[0]
    gap = hi["Approval_Rate"] - lo["Approval_Rate"]
    return (f"{label}: approval ranges from {pct(lo['Approval_Rate'])} ({lo_name}) to "
            f"{pct(hi['Approval_Rate'])} ({hi_name}), a gap of {gap:.1f} points.")


def generate_insights(df, kpis, assoc, report, model_full, model_no_credit):
    """Build the Business Intelligence text: overview, drivers, trends, risks, opportunities, actions."""
    out = {}
    n = len(df)
    good = rate(df, df["Credit_History"] == 1)
    bad = rate(df, df["Credit_History"] == 0)
    share_bad = (df["Credit_History"] == 0).mean() * 100
    n_bad = int((df["Credit_History"] == 0).sum())
    corr = df[NUMERIC_COLS + ["Loan_to_Income_Ratio"]].corrwith(df["Approved"]).abs().max()
    best = model_full["results"].iloc[0]
    no_cred_auc = model_no_credit["results"].iloc[0]["ROC-AUC"]
    top_features = model_full["importance"].head(3)

    # ---- Executive overview
    out["overview"] = [
        f"The portfolio holds {n:,} loan applications. {pct(kpis['Approval rate (%)'])} were approved, with an "
        f"average requested loan of {money(kpis['Average loan amount'])} and an average applicant income of "
        f"{money(kpis['Average applicant income'])} (currency units as recorded in the dataset).",
        f"Credit history is by far the strongest driver of decisions: {pct(good)} of applicants with a good "
        f"history were approved, against only {pct(bad)} of those without one.",
        f"{pct(share_bad)} of applicants ({n_bad:,}) have no good credit history, so almost half of all demand is "
        f"declined on one single criterion.",
        f"Income, loan amount, age and loan-to-income ratio show almost no relationship with approval "
        f"(largest absolute correlation: {corr:.2f}).",
        f"A {model_full['best_name']} model predicts decisions with {pct(best['Accuracy'] * 100)} accuracy "
        f"(ROC-AUC {best['ROC-AUC']:.3f}). Without Credit_History the same approach falls to ROC-AUC "
        f"{no_cred_auc:.2f}, close to random guessing (0.50).",
    ]

    # ---- Key drivers
    significant = assoc[assoc["Significant (p<0.05)"] == "Yes"]["Feature"].tolist() if len(assoc) else []
    out["drivers"] = [
        "Model importance ranking (permutation importance): " + ", ".join(f"{k} ({v * 100:.1f}%)" for k, v in top_features.items()) + ".",
        "Statistically significant differences in approval (chi-square, p<0.05): "
        + (", ".join(significant) if significant else "none found") + ".",
    ]

    # ---- Trend analysis (the data has no dates, so we look across ordered variables)
    trends = ["The dataset has no date column, so trends are read across ordered attributes instead of over time."]
    for col, label in [("Loan_Term", "Loan term (months)"), ("Age_Group", "Age group"),
                       ("Income_Band", "Household income band"), ("Dependents", "Number of dependents"),
                       ("Loan_to_Income_Band", "Loan-to-income band")]:
        s = _spread_sentence(df, col, label)
        if s:
            trends.append(s)
    trends.append(f"All of these gaps are small next to the {good - bad:.1f}-point gap created by credit "
                  f"history, so none of them is a strong lever on its own. Treat only the factors marked "
                  f"significant in the chi-square table as real patterns; the rest may be noise.")
    out["trends"] = trends

    # ---- Risks
    high_lti = df[(df["Approved"] == 1) & (df["Loan_to_Income_Ratio"] > 3)]
    approved_n = max(int(df["Approved"].sum()), 1)
    q1 = rate(df, df["Loan_to_Income_Band"] == "Low")
    q4 = rate(df, df["Loan_to_Income_Band"] == "Very High")
    emp = approval_by(df, "Employment_Status")
    weakest = emp.loc[emp["Approval_Rate"].idxmin()] if len(emp) else None

    # Employment gap among applicants who have a GOOD credit history (removes the credit effect)
    good_df = df[df["Credit_History"] == 1]
    emp_good = approval_by(good_df, "Employment_Status") if len(good_df) else pd.DataFrame()
    emp_good = emp_good[emp_good["Applications"] >= 30] if len(emp_good) else emp_good  # ignore tiny groups
    emp_gap = None
    if len(emp_good) > 1:
        lo_g = emp_good.loc[emp_good["Approval_Rate"].idxmin()]
        if weakest is not None:  # prefer the group with the lowest overall approval rate
            same_group = emp_good[emp_good["Employment_Status"] == weakest["Employment_Status"]]
            if len(same_group):
                lo_g = same_group.iloc[0]
        hi_g = emp_good.loc[emp_good["Approval_Rate"].idxmax()]
        extra = int(round(lo_g["Applications"] * (hi_g["Approval_Rate"] - lo_g["Approval_Rate"]) / 100))
        emp_gap = {"low": lo_g["Employment_Status"], "low_rate": lo_g["Approval_Rate"],
                   "high": hi_g["Employment_Status"], "high_rate": hi_g["Approval_Rate"], "extra": extra}
    gender_row = assoc[assoc["Feature"] == "Gender"] if len(assoc) else assoc
    risks = [
        f"Single-point dependency: {pct(share_bad)} of applicants are screened out almost entirely by credit "
        f"history ({pct(bad)} approved). Any error in the credit data would flip a large share of decisions.",
        f"Leverage is not controlled: applicants in the highest loan-to-income band are approved {pct(q4)} of the "
        f"time versus {pct(q1)} in the lowest band, and {len(high_lti) / approved_n * 100:.1f}% of approved loans "
        f"exceed 3x household income (assuming incomes are yearly; the dataset does not state the period).",
    ]
    if weakest is not None and len(emp) > 1:
        text = (f"Segment disadvantage: '{weakest['Employment_Status']}' applicants have the lowest approval "
                f"rate ({pct(weakest['Approval_Rate'])}).")
        if emp_gap:
            text += (f" The gap remains among applicants with a good credit history ({emp_gap['low']}: "
                     f"{pct(emp_gap['low_rate'])} vs {emp_gap['high']}: {pct(emp_gap['high_rate'])}), so credit "
                     f"history does not explain it. The data cannot show the cause.")
        risks.append(text)
    if len(gender_row):
        p_val = float(gender_row.iloc[0]["p-value"])
        verdict = ("no statistically significant gender gap was found" if p_val >= 0.05
                   else "a statistically significant gender gap exists and needs review")
        risks.append(f"Fairness monitoring: {verdict} (p={p_val:.2f}). Gender and marital status should stay "
                     f"outside decision rules and be re-checked regularly.")
    risks.append(f"Data quality: {report['missing_pct_before']:.1f}% of cells "
                 f"({report['missing_cells_before']:,}) were blank before cleaning, and the data records "
                 f"decisions, not repayment, so real default risk cannot be measured yet.")
    out["risks"] = risks

    # ---- Opportunities
    cand = review_candidates(df)
    cand_volume = cand["Loan_Amount"].sum() if len(cand) else 0
    rural = df[df["Property_Area"] == "Rural"]
    co_yes = rate(df, df["Has_Coapplicant"] == "Yes")
    co_no = rate(df, df["Has_Coapplicant"] == "No")
    opps = [
        f"Untapped segment: {len(cand):,} applicants with no good credit history still have above-median household "
        f"income and below-median loan-to-income ratio. They represent {money(cand_volume)} in requested loans that "
        f"are currently declined and could be reviewed manually.",
        f"Rural reach: rural applicants are only {len(rural) / n * 100:.1f}% of applications and are approved "
        f"{pct(rate(df, df['Property_Area'] == 'Rural'))} of the time, so there is room for targeted outreach.",
        f"Co-applicants: approval is {pct(co_yes)} with a co-applicant income versus {pct(co_no)} without, so "
        f"encouraging co-applicants is unlikely to raise approvals by itself.",
    ]
    if emp_gap:
        opps.append(f"Policy review: with a good credit history, {emp_gap['low']} applicants are approved "
                    f"{pct(emp_gap['low_rate'])} of the time versus {pct(emp_gap['high_rate'])} for "
                    f"{emp_gap['high']} applicants. Matching the higher rate would mean about {emp_gap['extra']} "
                    f"more approvals from creditworthy applicants.")
    out["opportunities"] = opps

    # ---- Recommendations
    out["recommendations"] = [
        {"Priority": "High",
         "Action": "Pilot manual / alternative-data review for the no-credit-history applicants who have strong income.",
         "Why (from the data)": f"{len(cand):,} such applicants ({money(cand_volume)} requested) are declined today.",
         "KPI to track": "Approval rate and repayment of the pilot group"},
        {"Priority": "High",
         "Action": "Use the prediction model only as a triage / pre-screening aid, with human sign-off on decisions.",
         "Why (from the data)": f"Accuracy is {pct(best['Accuracy'] * 100)}, but it relies almost entirely on Credit_History.",
         "KPI to track": "Model vs. human decision agreement"},
        {"Priority": "High",
         "Action": "Add a loan-to-income guardrail (for example a review flag above 3x household income).",
         "Why (from the data)": f"{len(high_lti) / approved_n * 100:.1f}% of approved loans already exceed 3x income.",
         "KPI to track": "Share of approved loans above the threshold"},
        {"Priority": "Medium",
         "Action": "Show a pre-application eligibility check so applicants learn about credit requirements early.",
         "Why (from the data)": f"{pct(share_bad)} of applications have almost no chance of approval.",
         "KPI to track": "Share of applications with poor credit history"},
        {"Priority": "Medium",
         "Action": "Review the approval rules for non-salaried applicants who have a good credit history, and offer "
                   "document-preparation support.",
         "Why (from the data)": (f"With good credit, {emp_gap['low']} applicants are approved "
                                 f"{pct(emp_gap['low_rate'])} vs {pct(emp_gap['high_rate'])} for {emp_gap['high']}."
                                 if emp_gap else "Some employment groups show lower approval rates."),
         "KPI to track": "Approval gap between employment types (good-credit applicants)"},
        {"Priority": "Medium",
         "Action": "Track repayment outcomes and review approval rates by gender and marital status every quarter.",
         "Why (from the data)": "The data has decisions only, and fairness needs continuous monitoring.",
         "KPI to track": "Default rate; approval gap by protected group"},
        {"Priority": "Low",
         "Action": "Make key fields mandatory in the application form.",
         "Why (from the data)": f"{report['missing_pct_before']:.1f}% of cells were blank before cleaning.",
         "KPI to track": "Percentage of blank cells"},
    ]
    return out


# ---------------------------------------------------------------------
# SECTION 7: STREAMLIT USER INTERFACE
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_data(raw):
    """Cached wrapper so cleaning only runs once per dataset."""
    return clean_data(raw)


@st.cache_data(show_spinner="Training models...")
def get_models(df, drop_features=()):
    """Cached wrapper so training only runs once per dataset."""
    return train_models(df, drop_features)


def prob_text(p):
    """Show a probability as text without pretending to be 100% or 0% sure."""
    if p > 0.999:
        return ">99.9%"
    if p < 0.001:
        return "<0.1%"
    return f"{p * 100:.1f}%"


def show_fig(fig):
    """Display a matplotlib figure in Streamlit and free its memory."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_bullets(items):
    for item in items:
        st.markdown(f"- {item}")


def kpi_cards(kpis):
    """Top row of headline numbers."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Applications", f"{kpis['Total applications']:,}")
    c2.metric("Approval rate", pct(kpis["Approval rate (%)"]))
    c3.metric("Avg loan amount", money(kpis["Average loan amount"]))
    c4.metric("Avg applicant income", money(kpis["Average applicant income"]))
    c5.metric("Good credit history", pct(kpis["Share with good credit history (%)"]))


def sidebar_filters(df):
    """Sidebar filters that apply to the dashboard tabs."""
    st.sidebar.header("Filters")
    picks = {}
    for col, label in [("Property_Area", "Property area"), ("Employment_Status", "Employment status"),
                       ("Education", "Education")]:
        options = sorted(df[col].unique())
        picks[col] = st.sidebar.multiselect(label, options, default=options)
    mask = pd.Series(True, index=df.index)
    for col, chosen in picks.items():
        mask &= df[col].isin(chosen)
    return df[mask]


def tab_executive(fdf, kpis, insights):
    st.subheader("Executive Overview")
    kpi_cards(kpis)
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(chart_status_pie(fdf))
    with c2:
        show_fig(chart_segment_bar(fdf, "Credit_History", "Approval rate by credit history"))
    c3, c4 = st.columns(2)
    with c3:
        show_fig(chart_histogram(fdf, "Loan_Amount", "Loan amount distribution"))
    with c4:
        show_fig(chart_segment_bar(fdf, "Employment_Status"))
    st.markdown("**Key messages**")
    show_bullets(insights["overview"])


def tab_kpis(fdf, kpis, assoc):
    st.subheader("KPI Summary")
    kpi_cards(kpis)
    table = pd.DataFrame({
        "KPI": list(kpis.keys()),
        "Value": [f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" for v in kpis.values()],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("**Approval rate by segment**")
    col = st.selectbox("Choose a segment", SEGMENT_COLS, index=1)
    seg = approval_by(fdf, col)
    seg["Approval_Rate"] = seg["Approval_Rate"].round(1)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(seg, use_container_width=True, hide_index=True)
    with c2:
        show_fig(chart_segment_bar(fdf, col))
    st.markdown("**Which factors really matter? (chi-square test)**")
    st.dataframe(assoc, use_container_width=True, hide_index=True)


def tab_trends(fdf, insights):
    st.subheader("Trend Analysis")
    show_bullets(insights["trends"])
    c1, c2 = st.columns(2)
    with c1:
        show_fig(chart_trend_line(fdf, "Loan_Term", "Approval rate by loan term (months)"))
    with c2:
        show_fig(chart_trend_line(fdf, "Age_Group", "Approval rate by age group"))
    c3, c4 = st.columns(2)
    with c3:
        show_fig(chart_trend_line(fdf, "Income_Band", "Approval rate by household income band"))
    with c4:
        show_fig(chart_trend_line(fdf, "Dependents", "Approval rate by dependents"))
    st.caption("Line charts are zoomed to 30-70% so small differences are visible. Read the value labels.")


def tab_risks(fdf, insights):
    st.subheader("Risk Analysis")
    show_bullets(insights["risks"])
    c1, c2 = st.columns(2)
    with c1:
        show_fig(chart_segment_bar(fdf, "Loan_to_Income_Band", "Approval rate by loan-to-income band"))
    with c2:
        show_fig(chart_box(fdf, "Loan_to_Income_Ratio", "Loan-to-income ratio by decision"))


def tab_opportunities(fdf, insights):
    st.subheader("Opportunity Analysis")
    show_bullets(insights["opportunities"])
    c1, c2 = st.columns(2)
    with c1:
        show_fig(chart_segment_bar(fdf, "Property_Area"))
    with c2:
        show_fig(chart_scatter(fdf))
    cand = review_candidates(fdf)
    st.markdown(f"**Manual-review candidates ({len(cand):,} applicants)** - no good credit history, "
                f"above-median income, below-median loan-to-income ratio")
    if len(cand):
        st.dataframe(cand.head(100).round(2), use_container_width=True, hide_index=True)
        st.download_button("Download candidate list (CSV)", cand.to_csv(index=False).encode("utf-8"),
                           file_name="review_candidates.csv", mime="text/csv")


def tab_recommendations(insights):
    st.subheader("Actionable Recommendations")
    st.dataframe(pd.DataFrame(insights["recommendations"]), use_container_width=True, hide_index=True)


def tab_cleaning(clean, report):
    st.subheader("Data Cleaning Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw -> clean)", f"{report['raw_rows']:,} -> {report['clean_rows']:,}")
    c2.metric("Columns (raw -> clean)", f"{report['raw_cols']} -> {report['clean_cols']}")
    c3.metric("Blank cells before", f"{report['missing_cells_before']:,}")
    c4.metric("Blank cells after", f"{report['missing_cells_after']:,}")
    c5, c6 = st.columns(2)
    with c5:
        show_fig(chart_missing(report["missing_before"], report["raw_rows"]))
    with c6:
        st.markdown("**Outlier check (kept, not removed)**")
        st.dataframe(report["outliers"], use_container_width=True, hide_index=True)
    st.markdown("**Every cleaning step**")
    st.dataframe(report["steps"], use_container_width=True, hide_index=True)
    st.download_button("Download cleaned data (CSV)", clean.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_loan_data.csv", mime="text/csv")


def tab_eda(fdf):
    st.subheader("Exploratory Data Analysis")
    st.markdown("**Data preview**")
    st.dataframe(fdf.head(20), use_container_width=True, hide_index=True)
    st.markdown("**Summary statistics**")
    st.dataframe(fdf[NUMERIC_COLS + ["Loan_Term", "Loan_to_Income_Ratio"]].describe().round(2),
                 use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        col = st.selectbox("Histogram of", NUMERIC_COLS + ["Total_Income", "Loan_to_Income_Ratio"], index=2)
        show_fig(chart_histogram(fdf, col))
    with c2:
        show_fig(chart_correlation(fdf))
    c3, c4 = st.columns(2)
    with c3:
        show_fig(chart_box(fdf, "Applicant_Income"))
    with c4:
        show_fig(chart_box(fdf, "Loan_Amount"))


def tab_model(clean, model_full, model_no_credit):
    st.subheader("Prediction Model")
    best = model_full["results"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", model_full["best_name"])
    c2.metric("Test accuracy", pct(best["Accuracy"] * 100))
    c3.metric("ROC-AUC", f"{best['ROC-AUC']:.3f}")
    c4.metric("5-fold CV accuracy", f"{model_full['cv_mean'] * 100:.1f}% (+/- {model_full['cv_std'] * 100:.1f})")
    st.caption(f"Trained on {model_full['n_train']:,} applications, tested on {model_full['n_test']:,} unseen ones.")
    st.dataframe(model_full["results"].round(3), use_container_width=True, hide_index=True)

    c5, c6 = st.columns(2)
    with c5:
        show_fig(chart_confusion(model_full["confusion"]))
    with c6:
        show_fig(chart_importance(model_full["importance"]))

    st.warning(
        f"Important: the model depends almost entirely on Credit_History. Without it, the best ROC-AUC drops to "
        f"{model_no_credit['results'].iloc[0]['ROC-AUC']:.2f} (0.50 = random guessing). "
        "Use predictions as decision support, not as a final decision."
    )
    st.markdown("**Model without Credit_History (for comparison)**")
    st.dataframe(model_no_credit["results"].round(3), use_container_width=True, hide_index=True)

    # ---- Single-applicant prediction form
    st.markdown("---")
    st.markdown("### Try it: predict one applicant")
    with st.form("applicant_form"):
        a, b, c = st.columns(3)
        gender = a.selectbox("Gender", sorted(clean["Gender"].unique()))
        married = b.selectbox("Married", sorted(clean["Married"].unique()))
        education = c.selectbox("Education", sorted(clean["Education"].unique()))
        employment = a.selectbox("Employment status", sorted(clean["Employment_Status"].unique()))
        area = b.selectbox("Property area", sorted(clean["Property_Area"].unique()))
        dependents = c.selectbox("Dependents", sorted(clean["Dependents"].unique()))
        income = a.number_input("Applicant income", min_value=1000.0, value=float(clean["Applicant_Income"].median()), step=1000.0)
        co_income = b.number_input("Co-applicant income", min_value=0.0, value=float(clean["Coapplicant_Income"].median()), step=1000.0)
        amount = c.number_input("Loan amount", min_value=1000.0, value=float(clean["Loan_Amount"].median()), step=5000.0)
        term = a.selectbox("Loan term (months)", sorted(clean["Loan_Term"].unique()))
        age = b.number_input("Age", min_value=18.0, max_value=100.0, value=float(round(clean["Age"].median())), step=1.0)
        history = c.selectbox("Credit history", [1, 0], format_func=lambda v: VALUE_LABELS["Credit_History"][v])
        submitted = st.form_submit_button("Predict")
    if submitted:
        applicant = {
            "Gender": gender, "Married": married, "Education": education, "Employment_Status": employment,
            "Property_Area": area, "Dependents": int(dependents), "Applicant_Income": income,
            "Coapplicant_Income": co_income, "Loan_Amount": amount, "Loan_Term": int(term),
            "Age": age, "Credit_History": int(history),
        }
        probability = predict_applicant(model_full, applicant)
        st.progress(min(max(probability, 0.0), 1.0))
        if probability >= 0.5:
            st.success(f"Likely APPROVED (estimated probability {prob_text(probability)})")
        else:
            st.error(f"Likely REJECTED (estimated approval probability {prob_text(probability)})")


def main():
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    st.title(PROJECT_TITLE)
    st.caption("AI-powered data analytics: cleaning, EDA, KPIs, business intelligence and prediction")

    # ---- Load data
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload a loan CSV (optional)", type=["csv"])
    default_path = find_default_data()
    if uploaded is None and default_path is None:
        st.error("Dataset not found. Put train.csv (or archive.zip) in the same folder as Project.py, "
                 "or upload a CSV from the sidebar.")
        st.stop()
    try:
        raw = load_data(uploaded if uploaded is not None else default_path)
    except Exception as error:
        st.error(f"Could not read the dataset: {error}")
        st.stop()

    # ---- Clean, filter, model
    clean, report = get_clean_data(raw)
    fdf = sidebar_filters(clean)
    st.sidebar.caption(f"Showing {len(fdf):,} of {len(clean):,} applications")
    if len(fdf) < 50:
        st.warning("Too few applications match the filters. Please widen the filters.")
        st.stop()

    model_full = get_models(clean)
    model_no_credit = get_models(clean, ("Credit_History",))
    kpis = compute_kpis(fdf)
    assoc = association_tests(fdf)
    insights = generate_insights(fdf, kpis, assoc, report, model_full, model_no_credit)

    # ---- Tabs
    tabs = st.tabs(["Executive Dashboard", "KPI Summary", "Trend Analysis", "Risk Analysis",
                    "Opportunities", "Recommendations", "Data Cleaning", "EDA", "Prediction Model"])
    with tabs[0]:
        tab_executive(fdf, kpis, insights)
    with tabs[1]:
        tab_kpis(fdf, kpis, assoc)
    with tabs[2]:
        tab_trends(fdf, insights)
    with tabs[3]:
        tab_risks(fdf, insights)
    with tabs[4]:
        tab_opportunities(fdf, insights)
    with tabs[5]:
        tab_recommendations(insights)
    with tabs[6]:
        tab_cleaning(clean, report)
    with tabs[7]:
        tab_eda(fdf)
    with tabs[8]:
        tab_model(clean, model_full, model_no_credit)


def running_inside_streamlit():
    """True when the file is started with `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if running_inside_streamlit():
        main()
    else:
        # Beginner-friendly: `python Project.py` also works by launching Streamlit for you
        print("Starting the dashboard... (you can also run:  streamlit run Project.py)")
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
