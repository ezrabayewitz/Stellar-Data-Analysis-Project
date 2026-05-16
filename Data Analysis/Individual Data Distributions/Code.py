"""
Astronomy Exploratory Data Analysis (EDA)
==========================================
Visualizes the cleaned astronomy dataset through the following steps:
    1. Distributions of individual columns (histograms, bar charts, scatter)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH = "/Users/ezrabayewitz/Downloads/cleaned_astronomy_data.csv"

NUMERICAL_COLUMNS   = ["magnitude", "distance_light_years", "discovery_year", "apparent_size_arcminutes"]
CATEGORICAL_COLUMNS = ["spectral_type", "planetary_system"]


# ── Step 1 · Individual Distributions ─────────────────────────────────────────

def plot_histograms(df: pd.DataFrame) -> None:
    """Plot a histogram for each numerical column."""
    for col in NUMERICAL_COLUMNS:
        plt.figure(figsize=(8, 6))
        plt.hist(df[col], bins=20, color="blue", edgecolor="black")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


def plot_bar_charts(df: pd.DataFrame) -> None:
    """Plot a frequency bar chart for each categorical column."""
    for col in CATEGORICAL_COLUMNS:
        plt.figure(figsize=(8, 6))
        df[col].value_counts().plot(kind="bar", color="lightgreen", edgecolor="black")
        plt.title(f"Frequency of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_sky_positions(df: pd.DataFrame) -> None:
    """Scatter plot of observed Right Ascension vs. Declination."""
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["observed_position_ra"],
        df["observed_position_dec"],
        alpha=0.5,
        color="orange",
    )
    plt.title("Position of Stars in the Sky")
    plt.xlabel("Right Ascension")
    plt.ylabel("Declination")
    plt.tight_layout()
    plt.show()



# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"[load] {len(df):,} rows loaded")

    print("[step 1] plotting individual distributions...")
    plot_histograms(df)
    plot_bar_charts(df)
    plot_sky_positions(df)


if __name__ == "__main__":
    main()
