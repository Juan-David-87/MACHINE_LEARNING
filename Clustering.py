import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def getDataSet():
    df = pd.read_csv("real_drug_dataset.csv")
    
    features = ["Dosage_mg", "Improvement_Score"]
    df_model = df[features].dropna()
    
    return df_model

def AppClusteringKmeans(k=3):
    df = getDataSet().copy()
    X = df.values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    df["Cluster"] = labels

    # Summary
    summary = df["Cluster"].value_counts().to_dict()

    # Centroids
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    centers_list = centers_original.tolist()

    # Sample
    sampled = []
    for cluster_id in range(k):
        group = df[df["Cluster"] == cluster_id]
        sample = group.sample(min(5, len(group)), random_state=42)
        sampled.extend(sample.to_dict(orient="records"))

    jitter = np.random.normal(0, 100, size=len(df))

    # Plot
    plt.figure(figsize=(8, 6))
    
    plt.scatter(
        df["Dosage_mg"] + jitter, 
        df["Improvement_Score"], 
        c=df["Cluster"], 
        cmap='viridis', 
        alpha=0.6
    )

    plt.scatter(
        centers_original[:, 0], 
        centers_original[:, 1], 
        c='red', 
        marker='X', 
        s=200, 
        label='Centroids'
    )

    plt.xlabel("Dosage (mg)")
    plt.ylabel("Improvement Score")
    plt.title("Patient Clustering: Dosage vs Improvement")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    
    graph = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    plt.close()

    return {
        "results": sampled,
        "summary": summary,
        "centers": centers_list,
        "graph": graph
    }