
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
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    cluster_mean = df.groupby("Cluster")["Total_Time"].mean()
    sorted_clusters = cluster_mean.sort_values().index
    risk_map = {
        sorted_clusters[0]: "Low Risk",
        sorted_clusters[1]: "Medium Risk",
        sorted_clusters[2]: "High Risk"
    }
    df["Supplier_Risk"] = df["Cluster"].map(risk_map)
    score = silhouette_score(X_scaled, df["Cluster"])
    return df, score