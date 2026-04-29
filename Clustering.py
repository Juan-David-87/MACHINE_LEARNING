import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def getDataSet():
    # Load dataset
    df = pd.read_csv("real_drug_dataset.csv")
    
    # Use exact column names from your CSV
    features = ["Dosis_mg", "Puntuación de mejora"]
    df_model = df[features].dropna()
    
    return df_model

def AppClusteringKmeans(k=3):
    df = getDataSet()
    X = df.values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model training
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    df["Cluster"] = labels

    # Cluster Summary (Count per cluster)
    summary = df["Cluster"].value_counts().to_dict()

    # Centroids (Inverse transformed to original scale for accurate plotting)
    centers_scaled = model.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    centers_list = centers_original.tolist()

    # Sample records for the results table (5 per cluster to keep UI clean)
    sampled = []
    for cluster_id in range(k):
        group = df[df["Cluster"] == cluster_id]
        sample = group.sample(min(5, len(group)), random_state=42)
        sampled.extend(sample.to_dict(orient="records"))

    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Data points
    plt.scatter(
        df["Dosis_mg"], 
        df["Puntuación de mejora"], 
        c=df["Cluster"], 
        cmap='viridis', 
        alpha=0.6
    )

    # Centroids
    plt.scatter(
        centers_original[:, 0], 
        centers_original[:, 1], 
        c='red', 
        marker='X', 
        s=200, 
        label='Centroids'
    )

    plt.xlabel("Dose (mg)")
    plt.ylabel("Improvement Score")
    plt.title("Patient Clustering: Dose vs Improvement")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save to Base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graph = base64.b64encode(image_png).decode("utf-8")
    plt.close()

    return {
        "results": sampled,
        "summary": summary,
        "centers": centers_list,
        "graph": graph
    }