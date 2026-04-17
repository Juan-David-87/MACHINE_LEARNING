import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("US_Crude_Oil_Imports.csv")
df = df.sort_values(by="year")
df = df.groupby("year", as_index=False)["quantity"].sum()
df = df.sort_values("year").reset_index(drop=True)

threshold = df["quantity"].median()
df["High_Imports"] = (df["quantity"].shift(-1) > threshold).astype(int)

X = df[["quantity"]]
y = df["High_Imports"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearDiscriminantAnalysis()
model.fit(X_scaled, y)

def getThreshold():
    return threshold

def predictOilCategoryLDA(year):
    row = df[df["year"] == year]
    if row.empty:
        raise ValueError(f"Year {year} not found in dataset.")
    
    quantity = row["quantity"].values[0]
    quantity_scaled = scaler.transform([[quantity]])
    
    prob = model.predict_proba(quantity_scaled)[0][1]
    category = model.predict(quantity_scaled)[0]
    
    return int(category), float(prob)

def generatePlot(year_value=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    q_min = df["quantity"].min()
    q_max = df["quantity"].max()
    margin = (q_max - q_min) * 0.1

    quantity_range = np.linspace(q_min - margin, q_max + margin, 500)
    q_scaled_range = scaler.transform(quantity_range.reshape(-1, 1))
    y_prob = model.predict_proba(q_scaled_range)[:, 1]
    decision_boundary_q = quantity_range[np.argmin(np.abs(y_prob - 0.5))]

    ax.axvspan(q_min - margin, decision_boundary_q,
               alpha=0.08, color="steelblue", label="Region: LOW Imports")
    
    ax.axvspan(decision_boundary_q, q_max + margin,
               alpha=0.08, color="tomato", label="Region: HIGH Imports")

    ax.plot(quantity_range, y_prob,
            color="steelblue", linewidth=2.5, label="P(HIGH | quantity)")

    ax.axvline(decision_boundary_q, color="gray", linestyle="--", linewidth=1.5,
               label=f"Decision boundary ≈ {decision_boundary_q:,.0f}")

    # Umbral horizontal 0.5
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=1.5,
               label="Threshold P = 0.5")

    colors = df["High_Imports"].map({0: "steelblue", 1: "tomato"})
    ax.scatter(df["quantity"], 
               model.predict_proba(scaler.transform(df[["quantity"]]))[:, 1],
               c=colors, s=60, zorder=4, edgecolors="white", linewidths=0.5,
               alpha=0.85, label="Observed data points")

    if year_value is not None:
        row = df[df["year"] == year_value]
        if not row.empty:
            qty = row["quantity"].values[0]
            qty_scaled = scaler.transform([[qty]])
            prob = model.predict_proba(qty_scaled)[0][1]
            category = model.predict(qty_scaled)[0]
            color_pred = "tomato" if category == 1 else "steelblue"

            ax.scatter(qty, prob, s=180, zorder=6, color=color_pred,
                       edgecolors="black", linewidths=1.5,
                       label=f"Year {year_value} (P={prob:.3f})")
            ax.annotate(
                f"  {year_value}\n  qty={qty:,.0f}\n  P={prob:.3f}",
                xy=(qty, prob), fontsize=8.5,
                xytext=(10, -30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
            )

    ax.set_xlabel("Oil Import Quantity", fontsize=11)
    ax.set_ylabel("P(HIGH Imports) — LDA posterior probability", fontsize=11)
    ax.set_title("Linear Discriminant Analysis (LDA) — Decision Boundary in Feature Space",
                 fontsize=13)
    ax.set_xlim(q_min - margin, q_max + margin)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="center right")

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()