import pandas as pd
import matplotlib.pyplot as plt
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

df["High_Imports"] = (df["quantity"] > threshold).astype(int)

X = df[["quantity"]]
y = df["High_Imports"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearDiscriminantAnalysis()
model.fit(X_scaled, y)

def getThreshold():
    return threshold

def predictOilCategoryLDA(year):
    quantity = np.interp(year, df["year"], df["quantity"])

    quantity_scaled = scaler.transform([[quantity]])
    prob = model.predict_proba(quantity_scaled)[0][1]
    category = model.predict(quantity_scaled)[0]

    return int(category), float(prob)

def generatePlot(year_value=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    year_min = df["year"].min()
    year_max = df["year"].max()

    years_range = np.linspace(year_min, year_max, 300)

    quantities = np.interp(years_range, df["year"], df["quantity"])

    q_scaled = scaler.transform(quantities.reshape(-1, 1))
    y_prob = model.predict_proba(q_scaled)[:, 1]

    ax.plot(years_range, y_prob, linewidth=2.5, label="LDA Probability Curve")

    ax.axhline(0.5, linestyle="--", label="Threshold (0.5)")

    if year_value is not None:
        quantity = np.interp(year_value, df["year"], df["quantity"])
        q_scaled_point = scaler.transform([[quantity]])
        prob = model.predict_proba(q_scaled_point)[0][1]

        ax.scatter(year_value, prob, s=120, zorder=5,
                   edgecolors="black", label=f"Prediction ({year_value})")
        ax.text(year_value, prob + 0.05,
                f"{round(prob, 3)}", ha="center", fontsize=9)

    ax.set_xlabel("Year")
    ax.set_ylabel("Probability of High Imports")
    ax.set_title("Linear Discriminant Analysis (LDA) - Oil Imports")
    ax.set_xlim(year_min - 1, year_max + 1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)
    ax.legend()

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()