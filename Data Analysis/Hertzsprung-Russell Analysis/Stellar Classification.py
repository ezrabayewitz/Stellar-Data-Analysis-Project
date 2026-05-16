"""
Astronomy — Stellar Phase Classification
=========================================
Classifies stars as Giants or Main Sequence based on derived luminosity,
then analyzes the distribution across spectral types using statistical
tests and visualizations.

    1. Derive luminosity and assign stellar phase
    2. Compute phase distribution by spectral type
    3. Statistical tests (Chi-squared, ANOVA)
    4. Visualizations (bar chart, pie chart)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH = "/Users/ezrabayewitz/Downloads/portfolio/cleaned_astronomy_data.csv"

EXCLUDED_SPECTRAL_TYPES  = ["Z"]
GIANT_LUMINOSITY_THRESHOLD = 10   # solar luminosities — stars above this are Giants
SUN_ABSOLUTE_MAGNITUDE     = 4.83

PIE_COLORS = ["red", "orange"]


# ── Step 1 · Derive Luminosity & Assign Phase ──────────────────────────────────

def load_and_prepare(path: str, exclude_types: list) -> pd.DataFrame:
    """
    Load the cleaned CSV, drop excluded spectral types, derive luminosity
    from apparent magnitude, and assign each star a phase label.
    """
    df = pd.read_csv(path)
    df = df[~df["spectral_type"].isin(exclude_types)]

    df["luminosity"] = 10 ** ((SUN_ABSOLUTE_MAGNITUDE - df["magnitude"]) / 2.5)
    df["phase"] = np.where(
        df["luminosity"] > GIANT_LUMINOSITY_THRESHOLD,
        "Giant",
        "Main Sequence",
    )

    print(f"[load] {len(df):,} rows loaded ({', '.join(exclude_types)} excluded)")
    return df


# ── Step 2 · Phase Distribution ───────────────────────────────────────────────

def compute_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by spectral type and phase, compute counts and percentages.
    Also prints overall dataset-wide phase percentages.
    """
    grouped = df.groupby(["spectral_type", "phase"]).size().unstack(fill_value=0)
    grouped["Total"]           = grouped.sum(axis=1)
    grouped["% Giants"]        = (grouped["Giant"]        / grouped["Total"]) * 100
    grouped["% Main Sequence"] = (grouped["Main Sequence"] / grouped["Total"]) * 100

    total_stars      = len(df)
    pct_giants       = (df["phase"] == "Giant").sum()        / total_stars * 100
    pct_main_seq     = (df["phase"] == "Main Sequence").sum() / total_stars * 100

    print(f"[distribution] Overall Giants:        {pct_giants:.1f}%")
    print(f"[distribution] Overall Main Sequence: {pct_main_seq:.1f}%")

    return grouped


# ── Step 3 · Statistical Tests ────────────────────────────────────────────────

def run_chi_squared(grouped: pd.DataFrame) -> None:
    """Chi-squared test on Giant vs Main Sequence counts across spectral types."""
    chi2, p, dof, _ = chi2_contingency(grouped[["Giant", "Main Sequence"]].values)
    print(f"[chi-squared] statistic={chi2:.4f}, p-value={p:.4f}, dof={dof}")


def run_anova(grouped: pd.DataFrame) -> None:
    """One-way ANOVA comparing % Giants vs % Main Sequence across spectral types."""
    result = f_oneway(grouped["% Giants"], grouped["% Main Sequence"])
    print(f"[ANOVA] F-statistic={result.statistic:.4f}, p-value={result.pvalue:.4f}")


# ── Step 4 · Visualizations ───────────────────────────────────────────────────

def plot_bar_chart(grouped: pd.DataFrame) -> None:
    """Bar chart showing percentage of Giants per spectral type."""
    grouped[["% Giants"]].plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.title("Percentage of Giants by Spectral Type")
    plt.tight_layout()
    plt.show()


def plot_pie_chart(df: pd.DataFrame) -> None:
    """Pie chart showing overall Giant vs Main Sequence split."""
    total       = len(df)
    pct_giants  = (df["phase"] == "Giant").sum() / total * 100
    sizes       = [pct_giants, 100 - pct_giants]
    labels      = ["Giants", "Main Sequence"]

    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=PIE_COLORS)
    plt.title("Overall Distribution of Star Classes")
    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df      = load_and_prepare(INPUT_PATH, EXCLUDED_SPECTRAL_TYPES)
    grouped = compute_distribution(df)

    print("[step 3] Running statistical tests...")
    run_chi_squared(grouped)
    run_anova(grouped)

    print("[step 4] Generating visualizations...")
    plot_bar_chart(grouped)
    plot_pie_chart(df)


if __name__ == "__main__":
    main()
