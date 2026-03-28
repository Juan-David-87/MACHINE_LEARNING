import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

df = pd.read_csv("world_happiness_report.csv")

#removes unnecessary columns for predictio
df = df.drop(columns=["Unnamed: 0"])

# removes empty rows
df = df.dropna()

#automatically fills missing rows without deleting data
#df = df.fillna(df.mean())

x = df[["Economy (GDP per Capita)"]]   #independent variable
y = df[["Happiness Score"]]  #dependent variable

model = LinearRegression()
model.fit(x , y) 

def predictHappiness(gdp):
    result = model.predict([[gdp]])
    return float(result[0][0])
def generatePlot(gdp_value=None):
    fig, ax = plt.subplots()

    df_sorted = df.sort_values(by="Economy (GDP per Capita)")

    X = df_sorted[["Economy (GDP per Capita)"]]
    Y = df_sorted["Happiness Score"]

    ax.scatter(X, Y, alpha=0.5)

    y_line = model.predict(X)
    ax.plot(X, y_line)

    if gdp_value is not None:
        import pandas as pd

        input_df = pd.DataFrame([[gdp_value]], columns=["Economy (GDP per Capita)"])
        predicted = model.predict(input_df)

        predicted_value = predicted[0][0]

        ax.scatter(gdp_value, predicted_value, color='red', s=100)
        ax.text(gdp_value, predicted_value, f"({gdp_value}, {round(predicted_value,2)})")

    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Happiness Score")
    ax.set_title("Linear Regression with Prediction")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()