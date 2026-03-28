import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

df = pd.read_csv("US_Crude_Oil_Imports.csv")

#  removes empty rows
df = df.dropna()

# variables
x = df[["year"]]   # independent variable
y = df[["quantity"]]  #dependent variable


model = LinearRegression()
model.fit(x, y)

def predictOil(year):
    result = model.predict([[year]])
    return float(result[0][0])

def generatePlot(year_value=None):
    fig, ax = plt.subplots()

    df_sorted = df.sort_values(by="year")

    X = df_sorted[["year"]]
    Y = df_sorted["quantity"]

    #real coordinates for points
    ax.scatter(X.values.flatten(), Y.values.flatten(), alpha=0.5)

    y_line = model.predict(X)
    ax.plot(X.values.flatten(), y_line.flatten())

    if year_value is not None:
        input_df = pd.DataFrame([[year_value]], columns=["year"])
        predicted = model.predict(input_df)
        predicted_value = predicted[0][0]

        ax.scatter(year_value, predicted_value, color='red', s=100)
        ax.text(year_value, predicted_value, f"({year_value}, {round(predicted_value,2)})")

    ax.set_xlabel("Year")
    ax.set_ylabel("Oil Quantity")
    ax.set_title("Linear Regression - Oil Imports")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()