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