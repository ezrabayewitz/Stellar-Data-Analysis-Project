"""
Astronomy — Distance vs Apparent Size Clustering
==================================================
Applies K-Means clustering to explore the relationship between
star distance and apparent size, broken down by spectral type.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH = "/Users/ezrabayewitz/Downloads/cleaned_astronomy_data.csv"

EXCLUDED_SPECTRAL_TYPES = ["Z"]

KMEANS_N_CLUSTERS   = 4
RANDOM_STATE        = 42
DISTANCE_THRESHOLD  = 5000  # light-years
SIZE_THRESHOLD      = 0.5   # scaled arcseconds

CLUSTER_LABELS = {
    0: "Close & Small",
    1: "Far & Big",
    2: "Far & Small",
    3: "Close & Big",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(path: str, exclude_types: list) -> pd.DataFrame:
    """Load the cleaned CSV and drop any excluded spectral types."""
    df = pd.read_csv(path)
    df = df[~df["spectral_type"].isin(exclude_types)]
    print(f"[load] {len(df):,} rows loaded ({', '.join(exclude_types)} excluded)")
    return df


def assign_rule_based_labels(stars: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable cluster labels based on
    distance and apparent size thresholds.
    """
    stars = stars.copy()
    close = stars["distance_light_years"] < DISTANCE_THRESHOLD
    big   = stars["scaled_apparent_size"] >= SIZE_THRESHOLD

    stars.loc[ close &  big,  "cluster_label"] = "Close & Big"
    stars.loc[ close & ~big,  "cluster_label"] = "Close & Small"
    stars.loc[~close &  big,  "cluster_label"] = "Far & Big"
    stars.loc[~close & ~big,  "cluster_label"] = "Far & Small"
    return stars


def print_cluster_distribution(stars: pd.DataFrame, spectral_type: str) -> None:
    """Print the percentage breakdown of cluster labels for a spectral type."""
    counts      = stars["cluster_label"].value_counts()
    percentages = (counts / len(stars)) * 100
    print(f"Cluster distribution — {spectral_type} type stars:")
    print(percentages.to_string())
    print()


# ── Step 1 · K-Means Clustering ───────────────────────────────────────────────

def run_kmeans(df: pd.DataFrame) -> None:
    """
    For each spectral type:
      - Normalize distance and apparent size
      - Fit K-Means with KMEANS_N_CLUSTERS clusters
      - Override cluster IDs with rule-based labels
      - Print cluster distribution and plot results
    """
    for spectral_type in df["spectral_type"].unique():
        stars = df[df["spectral_type"] == spectral_type].copy()

        # Normalize features
        features_scaled = StandardScaler().fit_transform(
            stars[["distance_light_years", "scaled_apparent_size"]]
        )

        # Fit K-Means
        kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE)
        stars["cluster"] = kmeans.fit_predict(features_scaled)

        # Apply rule-based labels
        stars = assign_rule_based_labels(stars)
        print_cluster_distribution(stars, spectral_type)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=stars,
            x="distance_light_years",
            y="scaled_apparent_size",
            hue="cluster_label",
            palette="viridis",
            s=100,
            legend="full",
        )
        plt.title(f"Distance vs Apparent Size — {spectral_type} Type Stars (K-Means)")
        plt.xlabel("Distance (Light Years)")
        plt.ylabel("Apparent Size (Arcseconds)")
        plt.xticks(rotation=45)
        plt.legend(title="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_data(INPUT_PATH, EXCLUDED_SPECTRAL_TYPES)

    print("[step 1] Running K-Means clustering...")
    run_kmeans(df)


if __name__ == "__main__":
    main()
