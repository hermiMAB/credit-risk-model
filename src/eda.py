"""
eda.py
======
Reusable, modular EDA functions for Task 1 (Understanding Credit Risk).

Design goal: the notebook should only ever call these functions and add
markdown commentary — it should not contain raw plotting/aggregation code.
That keeps the notebook readable and lets the same functions be reused
(e.g. in a Streamlit app, a report script, or unit tests).

Typical usage in the notebook:

    from pathlib import Path
    import sys
    sys.path.append(str(Path.cwd().parent / "scripts"))  # wherever this file lives
    import eda

    df = eda.load_data(Path.cwd().parent / "data" / "data.csv")
    eda.overview(df)
    eda.summary_statistics(df)
    eda.plot_numerical_distributions(df, NUMERIC_COLS)
    eda.plot_categorical_distributions(df, CATEGORICAL_COLS)
    eda.correlation_heatmap(df, NUMERIC_COLS)
    eda.missing_value_report(df)
    eda.plot_outlier_boxplots(df, NUMERIC_COLS)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# ==========================================
# 0. LOADING & TYPE PREPARATION
# ==========================================

def load_data(
    path: Path,
    time_cols: Iterable[str] = ("TransactionStartTime",),
    id_cols: Iterable[str] = (
        "TransactionId", "BatchId", "AccountId", "SubscriptionId",
        "CustomerId", "ProviderId", "ProductId", "CountryCode",
    ),
) -> pd.DataFrame:
    """Load the raw CSV and apply the minimal, consistent dtype fixes.

    Parameters
    ----------
    path: path to the raw csv
    time_cols: columns to parse as datetime
    id_cols: columns to force to string so they are treated as categorical
        identifiers rather than numbers (e.g. CountryCode).
    """
    df = pd.read_csv(path)

    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    existing_id_cols = [c for c in id_cols if c in df.columns]
    df[existing_id_cols] = df[existing_id_cols].astype(str)

    return df


# ==========================================
# 1. OVERVIEW
# ==========================================

def overview(df: pd.DataFrame) -> None:
    """Print shape, dtypes, and non-null counts (Task item 1)."""
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    df.info()


def summary_statistics(df: pd.DataFrame, numeric_cols: Optional[list] = None) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns (Task item 2)."""
    cols = numeric_cols if numeric_cols else df.select_dtypes(include=np.number).columns.tolist()
    return df[cols].describe().T


def skew_kurtosis_table(df: pd.DataFrame, numeric_cols: Optional[list] = None) -> pd.DataFrame:
    """Return a table of skewness and kurtosis for numeric columns (Task item 2)."""
    cols = numeric_cols if numeric_cols else df.select_dtypes(include=np.number).columns.tolist()
    return pd.DataFrame({
        "Skewness": df[cols].skew(numeric_only=True),
        "Kurtosis": df[cols].kurt(numeric_only=True),
    }).round(2)


# ==========================================
# 2. DISTRIBUTIONS (Task items 3 & 4)
# ==========================================

def plot_numerical_distributions(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    bins: int = 50,
    log_scale_cols: Optional[Iterable[str]] = None,
) -> None:
    """Histogram + KDE for each numeric column, one subplot per column.

    log_scale_cols: columns that are heavy-tailed (e.g. Amount, Value) and
    should be shown with a log-scaled x-axis alongside the raw version.
    """
    log_scale_cols = set(log_scale_cols or [])
    n = len(list(numeric_cols))
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col], bins=bins, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
        if col in log_scale_cols and (df[col] > 0).all():
            ax.set_xscale("log")

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: Iterable[str],
    top_n: int = 15,
) -> None:
    """Count plot for each categorical column, sorted by frequency."""
    for col in categorical_cols:
        order = df[col].value_counts().iloc[:top_n].index
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, data=df, order=order, color="orchid")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Frequency Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count of Transactions")
        plt.tight_layout()
        plt.show()


# ==========================================
# 3. CORRELATION (Task item 5)
# ==========================================

def correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Plot a correlation heatmap for numeric columns and return the matrix.

    method='spearman' is a better default when the target is a rare binary
    event and features are heavy-tailed (rank correlation is robust to both).
    """
    corr = df[list(numeric_cols)].corr(method=method)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"{method.title()} Correlation Matrix (Numerical Features)")
    plt.tight_layout()
    plt.show()
    return corr


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Bar plot of a binary/categorical target's class balance; returns the rate table."""
    plt.figure(figsize=(5, 4))
    sns.countplot(x=target_col, hue=target_col, data=df, palette="Set2", legend=False)
    plt.title(f"{target_col}: Class Distribution")
    plt.tight_layout()
    plt.show()
    return df[target_col].value_counts(normalize=True).rename("proportion")


def scatter_by_target(
    df: pd.DataFrame,
    x: str,
    y: str,
    target_col: str,
    symlog: bool = True,
) -> None:
    """Scatter of two numeric features colored by a (typically rare) target class."""
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=x, y=y, hue=target_col, data=df, palette="Set1", alpha=0.6)
    if symlog:
        plt.xscale("symlog")
        plt.yscale("symlog")
    plt.title(f"{x} vs. {y} (colored by {target_col})")
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. MISSING VALUES (Task item 6)
# ==========================================

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of missing counts and percentages per column."""
    counts = df.isnull().sum()
    pct = (counts / len(df)) * 100
    report = pd.DataFrame({"missing_count": counts, "missing_pct": pct.round(2)})
    return report[report["missing_count"] >= 0].sort_values("missing_count", ascending=False)


# ==========================================
# 5. OUTLIERS (Task item 7)
# ==========================================

def plot_outlier_boxplots(df: pd.DataFrame, numeric_cols: Iterable[str]) -> None:
    """Boxplot for each numeric column to visualize outliers."""
    n = len(list(numeric_cols))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(y=df[col], ax=ax, color="lightseagreen")
        ax.set_title(col)

    plt.tight_layout()
    plt.show()


def iqr_outlier_summary(df: pd.DataFrame, numeric_cols: Iterable[str], factor: float = 1.5) -> pd.DataFrame:
    """Count how many values fall outside the IQR bounds per numeric column."""
    rows = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "Feature": col, "Lower_Bound": lower, "Upper_Bound": upper,
            "Outlier_Count": n_outliers, "Outlier_Pct": round(100 * n_outliers / len(df), 2),
        })
    return pd.DataFrame(rows).sort_values("Outlier_Pct", ascending=False)


# ==========================================
# 6. ORCHESTRATOR
# ==========================================

def run_full_eda(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
    log_scale_cols: Optional[Iterable[str]] = None,
) -> dict:
    """Run every step in sequence and return the tables worth keeping.

    Plots are shown inline; only the tabular outputs are returned so the
    notebook can inspect or export them (e.g. missing-value report).
    """
    overview(df)
    stats = summary_statistics(df, numeric_cols)
    plot_numerical_distributions(df, numeric_cols, log_scale_cols=log_scale_cols)
    plot_categorical_distributions(df, categorical_cols)
    corr = correlation_heatmap(df, numeric_cols)
    missing = missing_value_report(df)
    plot_outlier_boxplots(df, numeric_cols)
    outliers = iqr_outlier_summary(df, numeric_cols)

    return {
        "summary_statistics": stats,
        "correlation": corr,
        "missing_report": missing,
        "outlier_summary": outliers,
    }
