
# Refactored as a function for agent use
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_supplier_risk(df):
    """
    Takes a DataFrame with supplier and lead time/delay columns, returns DataFrame with risk labels and silhouette score.
    """
    df = df.copy()
    df["Transit Delay Scenario"] = df["Transit Delay Scenario"].astype(str).str.strip().str.lower()
    delay_map = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "high (port delay +7d)": 4
    }
    df["Transit_Risk"] = df["Transit Delay Scenario"].map(delay_map)
    df = df.dropna(subset=["Transit_Risk"])
    df["Total_Time"] = (
        df["Lead Time Supplier→Plant (Days)"] +
        df["PO Processing (Days)"] +
        df["GR Processing (Days)"]
    )
    features = ["Total_Time", "Transit_Risk"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use fewer clusters if not enough samples
    n_samples = len(df)
    n_clusters = min(3, max(2, n_samples - 1))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    cluster_mean = df.groupby("Cluster")["Total_Time"].mean()
    sorted_clusters = cluster_mean.sort_values().index

    risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
    risk_map = {sorted_clusters[i]: risk_labels[i] for i in range(n_clusters)}
    df["Supplier_Risk"] = df["Cluster"].map(risk_map)

    # Silhouette score needs at least 2 clusters and 2 samples per cluster
    try:
        score = silhouette_score(X_scaled, df["Cluster"])
    except Exception:
        score = 0.0

    return df, score