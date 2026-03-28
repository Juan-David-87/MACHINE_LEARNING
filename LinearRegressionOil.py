import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

df = pd.read_csv("US_Crude_Oil_Imports.csv")

x = df[["year"]] #independent Variable
y = df["quantity"] #dependent Variable

model = LinearRegression()
model.fit(x, y)

def predictOil(year):
    input_df = pd.DataFrame([[year]], columns=["year"])
    result = model.predict(input_df)
    return float(result[0])

def generatePlot(year_value=None):
    fig, ax = plt.subplots()

    df_sorted = df.sort_values(by="year")

    X = df_sorted["year"].values 
    Y = df_sorted["quantity"].values

    ax.scatter(X, Y, alpha=0.5)

    X_model = df_sorted[["year"]]
    y_line = model.predict(X_model).flatten()
    ax.plot(X, y_line, color='red')

    if year_value is not None:
        input_df = pd.DataFrame([[year_value]], columns=["year"])
        predicted = model.predict(input_df).flatten()[0]

        ax.scatter(year_value, predicted, color='green', s=100)
        ax.text(year_value, predicted, f"({year_value}, {round(predicted,2)})")

    ax.set_xlabel("Year")
    ax.set_ylabel("Oil Quantity")
    ax.set_title("Linear Regression - Oil Imports")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()