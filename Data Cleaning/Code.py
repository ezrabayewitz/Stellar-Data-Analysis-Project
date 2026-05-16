"""
Astronomy Data Cleaning Pipeline
=================================
Cleans and preprocesses raw astronomy CSV data through the following steps:
    1. Import data
    2. Inspect data
    3. Remove duplicates
    4. Standardize column names
    5. Fix data types
    6. Handle outliers
    7. Fill missing values
    8. Scale apparent size
    9. Export cleaned data
"""

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH  = "/Users/ezrabayewitz/Downloads/portfolio/astronomy_data.csv"
OUTPUT_PATH = "/Users/ezrabayewitz/Downloads/portfolio/cleaned_astronomy_data.csv"

OUTLIER_APPARENT_SIZE_MAX = 15       # arcminutes
OUTLIER_DISTANCE_MAX      = 20_000   # light-years

SCALED_SIZE_MIN = 0.01               # arcseconds (realistic lower bound)
SCALED_SIZE_MAX = 1.0                # arcseconds (realistic upper bound)


# ── Step 1 · Import ────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Read CSV file into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[load] {len(df):,} rows loaded from {path}")
    return df


# ── Step 2 · Inspect ───────────────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick overview of the DataFrame."""
    print("\n── Head ──────────────────────────────")
    print(df.head())
    print("\n── Info ──────────────────────────────")
    print(df.info())
    print("\n── Describe ──────────────────────────")
    print(df.describe())


# ── Step 3 · Remove Duplicates ─────────────────────────────────────────────────

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[duplicates] removed {before - len(df):,} duplicate rows")
    return df


# ── Step 4 · Standardize Column Names ─────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, and replace spaces with underscores."""
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    print(f"[columns] standardized: {list(df.columns)}")
    return df


# ── Step 5 · Fix Data Types ────────────────────────────────────────────────────

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce 'distance_light_years' to numeric (non-parseable → NaN)."""
    df["distance_light_years"] = pd.to_numeric(
        df["distance_light_years"], errors="coerce"
    )
    print(f"[dtypes]\n{df.dtypes}")
    return df


# ── Step 6 · Handle Outliers ───────────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with implausible apparent size or distance values."""
    n_size = df[df["apparent_size_arcminutes"] > OUTLIER_APPARENT_SIZE_MAX].shape[0]
    print(f"[outliers] apparent_size_arcminutes > {OUTLIER_APPARENT_SIZE_MAX}: {n_size} rows")

    df = df[df["apparent_size_arcminutes"] < OUTLIER_APPARENT_SIZE_MAX]
    df = df[df["distance_light_years"] < OUTLIER_DISTANCE_MAX]

    print(f"[outliers] {len(df):,} rows remaining after removal")
    return df


# ── Step 7 · Fill Missing Values ───────────────────────────────────────────────

def _build_spectral_magnitude_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with min/max magnitude per spectral type."""
    return df.groupby("spectral_type")["magnitude"].agg(["min", "max"])


def _fill_magnitude_row(row: pd.Series, ranges: pd.DataFrame) -> float:
    """Return a random magnitude within the spectral type's observed range."""
    if pd.isnull(row["magnitude"]):
        lo = ranges.loc[row["spectral_type"], "min"]
        hi = ranges.loc[row["spectral_type"], "max"]
        return np.random.uniform(lo, hi)
    return row["magnitude"]


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - magnitude          → random value within the spectral-type's min/max range
      - distance_light_years → column mean
      - planetary_system   → 'unknown'
    """
    # Magnitude
    missing_mag = df["magnitude"].isna().sum()
    print(f"[missing] magnitude: {missing_mag} NaNs")
    spectral_ranges = _build_spectral_magnitude_ranges(df)
    df["magnitude"] = df.apply(
        _fill_magnitude_row, axis=1, ranges=spectral_ranges
    )

    # Distance
    missing_dist = df["distance_light_years"].isna().sum()
    print(f"[missing] distance_light_years: {missing_dist} NaNs → filled with mean")
    df["distance_light_years"].fillna(df["distance_light_years"].mean(), inplace=True)

    # Planetary system
    missing_ps = df["planetary_system"].isna().sum()
    print(f"[missing] planetary_system: {missing_ps} NaNs → filled with 'unknown'")
    df["planetary_system"].fillna("unknown", inplace=True)

    return df


# ── Step 8 · Scale Apparent Size ───────────────────────────────────────────────

def scale_apparent_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max scale 'apparent_size_arcminutes' into a realistic arcsecond range
    and store the result in 'scaled_apparent_size'.
    """
    col = df["apparent_size_arcminutes"]
    df["scaled_apparent_size"] = (
        (col - col.min()) / (col.max() - col.min())
    ) * (SCALED_SIZE_MAX - SCALED_SIZE_MIN) + SCALED_SIZE_MIN

    print("[scale] scaled_apparent_size stats:")
    print(df["scaled_apparent_size"].describe())
    return df


# ── Step 9 · Export ────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, path: str) -> None:
    """Write the cleaned DataFrame to a CSV file."""
    df.to_csv(path, index=False)
    print(f"\n[save] cleaned data saved to {path}  ({len(df):,} rows)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_data(INPUT_PATH)
    inspect_data(df)
    df = drop_duplicates(df)
    df = standardize_columns(df)
    df = fix_dtypes(df)
    df = remove_outliers(df)
    df = fill_missing_values(df)
    df = scale_apparent_size(df)
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
