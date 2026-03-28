import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression

#charge dataset
df = pd.read_csv("US_Crude_Oil_Imports.csv")

#removes empty rows
df = df.dropna()

threshold = df["quantity"].mean()
df["High_Imports"] = (df["quantity"] > threshold).astype(int)

X = df[["year"]]
y = df["High_Imports"]

model = LogisticRegression()
model.fit(X, y)

def predictOilCategory(year):
    result = model.predict([[year]])
    return int(result[0])

def generatePlot(year_value=None):
    fig, ax = plt.subplots()

    df_sorted = df.sort_values(by="year")

    X_plot = df_sorted["year"]
    Y_plot = df_sorted["High_Imports"]

    #coordinates for points
    ax.scatter(X_plot, Y_plot, alpha=0.5)

    #logistics curve
    X_range = pd.DataFrame({
        "year": sorted(df["year"])
    })

    y_prob = model.predict_proba(X_range)[:,1]

    ax.plot(X_range["year"], y_prob)

    # punto usuario
    if year_value is not None:
        input_df = pd.DataFrame([[year_value]], columns=["year"])
        prob = model.predict_proba(input_df)[0][1]

        ax.scatter(year_value, prob, color='red', s=100)
        ax.text(year_value, prob, f"{round(prob,2)}")

    ax.set_xlabel("Year")
    ax.set_ylabel("Probability of High Imports")
    ax.set_title("Logistic Regression - Oil Imports")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()