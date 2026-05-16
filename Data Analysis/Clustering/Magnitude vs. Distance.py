"""
Astronomy Clustering Analysis
==============================
Applies unsupervised clustering to the cleaned astronomy dataset
to explore relationships between star distance and magnitude,
broken down by spectral type.

    1. K-Means Clustering  — 4 interpretable clusters per spectral type
    2. Gaussian Mixture Model (GMM) — probabilistic clustering per spectral type
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH = "/Users/ezrabayewitz/Downloads/portfolio/cleaned_astronomy_data.csv"

EXCLUDED_SPECTRAL_TYPES = ["Z"]

KMEANS_N_CLUSTERS  = 4
GMM_N_COMPONENTS   = 13
RANDOM_STATE       = 42
DISTANCE_THRESHOLD = 5000   # light-years
MAGNITUDE_THRESHOLD = 8

CLUSTER_LABELS = {
    0: "Close & Dim",
    1: "Far & Bright",
    2: "Far & Dim",
    3: "Close & Bright",
}

CLUSTER_PALETTE = {
    "Close & Dim":    "#1f77b4",
    "Far & Bright":   "#ff7f0e",
    "Far & Dim":      "#2ca02c",
    "Close & Bright": "#d62728",
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
    Override K-Means cluster IDs with human-readable labels
    based on distance and magnitude thresholds.
    """
    stars = stars.copy()
    close = stars["distance_light_years"] < DISTANCE_THRESHOLD
    bright = stars["magnitude"] < MAGNITUDE_THRESHOLD

    stars.loc[ close &  bright, "cluster_label"] = "Close & Bright"
    stars.loc[ close & ~bright, "cluster_label"] = "Close & Dim"
    stars.loc[~close &  bright, "cluster_label"] = "Far & Bright"
    stars.loc[~close & ~bright, "cluster_label"] = "Far & Dim"
    return stars


def print_cluster_distribution(stars: pd.DataFrame, spectral_type: str) -> None:
    """Print the percentage breakdown of cluster labels for a spectral type."""
    counts = stars["cluster_label"].value_counts()
    percentages = (counts / len(stars)) * 100
    print(f"Cluster distribution — {spectral_type} type stars:")
    print(percentages.to_string())
    print()


# ── Step 1 · K-Means Clustering ───────────────────────────────────────────────

def run_kmeans(df: pd.DataFrame) -> None:
    """
    For each spectral type:
      - Normalize distance and magnitude
      - Fit K-Means with KMEANS_N_CLUSTERS clusters
      - Override cluster IDs with rule-based labels
      - Print cluster distribution and plot results
    """
    for spectral_type in df["spectral_type"].unique():
        stars = df[df["spectral_type"] == spectral_type].copy()

        # Normalize features
        features_scaled = StandardScaler().fit_transform(
            stars[["distance_light_years", "magnitude"]]
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
            y="magnitude",
            hue="cluster_label",
            palette=CLUSTER_PALETTE,
            s=100,
            legend="full",
        )
        plt.title(f"Magnitude vs Distance — {spectral_type} Type Stars (K-Means)")
        plt.xlabel("Distance (Light Years)")
        plt.ylabel("Magnitude")
        plt.xticks(rotation=45)
        plt.gca().invert_yaxis()
        plt.legend(title="Cluster", loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ── Step 2 · Gaussian Mixture Model Clustering ────────────────────────────────

def run_gmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each spectral type:
      - Fit a Gaussian Mixture Model with GMM_N_COMPONENTS components
      - Store cluster assignments back on the main DataFrame
      - Plot results
    Returns the DataFrame with a new 'gmm_cluster' column.
    """
    df = df.copy()

    for spectral_type, group in df.groupby("spectral_type"):
        features = group[["distance_light_years", "magnitude"]].values

        gmm = GaussianMixture(n_components=GMM_N_COMPONENTS, random_state=RANDOM_STATE)
        clusters = gmm.fit_predict(features)
        df.loc[group.index, "gmm_cluster"] = clusters

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=group,
            x="distance_light_years",
            y="magnitude",
            hue=clusters,
            palette="coolwarm",
            s=100,
            legend="full",
        )
        plt.title(f"Magnitude vs Distance — {spectral_type} Type Stars (GMM)")
        plt.xlabel("Distance (Light Years)")
        plt.ylabel("Magnitude")
        plt.gca().invert_yaxis()
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.show()

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_data(INPUT_PATH, EXCLUDED_SPECTRAL_TYPES)

    print("[step 1] Running K-Means clustering...")
    run_kmeans(df)

    print("[step 2] Running Gaussian Mixture Model clustering...")
    df = run_gmm(df)


if __name__ == "__main__":
    main()
