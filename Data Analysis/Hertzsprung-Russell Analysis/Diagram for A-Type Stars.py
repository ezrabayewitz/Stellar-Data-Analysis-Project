"""
Astronomy — Hertzsprung-Russell Diagram for A-Type Stars
=========================================================
Derives luminosity and assigns random temperatures for A-type stars,
classifies them into stellar phases, and plots an H-R diagram.

    1. Load and filter data
    2. Derive temperature and luminosity
    3. Classify stellar phase
    4. Plot H-R diagram
    5. Print phase distribution statistics
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH = "/Users/ezrabayewitz/Downloads/cleaned_astronomy_data.csv"

EXCLUDED_SPECTRAL_TYPES  = ["Z"]
TARGET_SPECTRAL_TYPE     = "A"
RANDOM_SEED              = 42

A_TYPE_TEMP_MIN = 7500   # Kelvin
A_TYPE_TEMP_MAX = 10000  # Kelvin

SUN_ABSOLUTE_MAGNITUDE       = 4.83
SUPERGIANT_LUMINOSITY_THRESHOLD = 1000  # solar luminosities
GIANT_LUMINOSITY_THRESHOLD      = 10    # solar luminosities

PHASE_SIZES = {
    "Main Sequence": 50,
    "Giant":         150,
    "Supergiant":    200,
}


# ── Step 1 · Load & Filter ────────────────────────────────────────────────────

def load_data(path: str, exclude_types: list, target_type: str) -> pd.DataFrame:
    """Load the cleaned CSV, drop excluded spectral types, and filter to target type."""
    df = pd.read_csv(path)
    df = df[~df["spectral_type"].isin(exclude_types)]

    target = df[df["spectral_type"] == target_type].copy()
    print(f"[load] {len(target):,} {target_type}-type stars found out of {len(df):,} total")
    return target


# ── Step 2 · Derive Temperature & Luminosity ──────────────────────────────────

def derive_features(stars: pd.DataFrame) -> pd.DataFrame:
    """
    Assign random temperatures within the A-type range and
    derive luminosity from absolute magnitude.
    """
    np.random.seed(RANDOM_SEED)
    stars["temperature"] = np.random.uniform(
        A_TYPE_TEMP_MIN, A_TYPE_TEMP_MAX, size=len(stars)
    )
    stars["luminosity"] = 10 ** ((SUN_ABSOLUTE_MAGNITUDE - stars["magnitude"]) / 2.5)
    return stars


# ── Step 3 · Classify Stellar Phase ──────────────────────────────────────────

def classify_phase(row: pd.Series) -> str:
    """Return the stellar phase based on luminosity thresholds."""
    if row["luminosity"] > SUPERGIANT_LUMINOSITY_THRESHOLD:
        return "Supergiant"
    elif row["luminosity"] > GIANT_LUMINOSITY_THRESHOLD:
        return "Giant"
    return "Main Sequence"


def assign_phases(stars: pd.DataFrame) -> pd.DataFrame:
    """Apply phase classification and map to plot sizes."""
    stars["phase"] = stars.apply(classify_phase, axis=1)
    stars["size"]  = stars["phase"].map(PHASE_SIZES)
    return stars


# ── Step 4 · Plot H-R Diagram ─────────────────────────────────────────────────

def plot_hr_diagram(stars: pd.DataFrame, spectral_type: str) -> None:
    """Scatter plot of temperature vs luminosity (H-R diagram style)."""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        stars["temperature"],
        stars["luminosity"],
        c="yellow",
        s=stars["size"],
        alpha=0.7,
        edgecolor="black",
    )
    plt.gca().invert_xaxis()
    plt.yscale("log")
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel("Luminosity (Solar Luminosities)", fontsize=14)
    plt.title(f"Hertzsprung-Russell Diagram for {spectral_type}-Type Stars", fontsize=16)
    plt.tight_layout()
    plt.show()


# ── Step 5 · Print Phase Statistics ──────────────────────────────────────────

def print_phase_stats(stars: pd.DataFrame) -> None:
    """Print the count and percentage of each stellar phase."""
    total      = len(stars)
    giants     = (stars["phase"] == "Giant").sum()
    supergiants = (stars["phase"] == "Supergiant").sum()

    print(f"[stats] Total {TARGET_SPECTRAL_TYPE}-type stars: {total}")
    print(f"[stats] Giants:      {giants:,}  ({giants / total * 100:.2f}%)")
    print(f"[stats] Supergiants: {supergiants:,}  ({supergiants / total * 100:.2f}%)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    stars = load_data(INPUT_PATH, EXCLUDED_SPECTRAL_TYPES, TARGET_SPECTRAL_TYPE)
    stars = derive_features(stars)
    stars = assign_phases(stars)

    print("[step 4] Plotting H-R diagram...")
    plot_hr_diagram(stars, TARGET_SPECTRAL_TYPE)

    print("[step 5] Phase statistics:")
    print_phase_stats(stars)


if __name__ == "__main__":
    main()
