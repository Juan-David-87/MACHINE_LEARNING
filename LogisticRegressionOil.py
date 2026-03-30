import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("US_Crude_Oil_Imports.csv")

threshold = df["quantity"].mean()
df["High_Imports"] = (df["quantity"] > threshold).astype(int)

X = df[["year"]]
y = df["High_Imports"]

#this function works to scale the years and make the curve look better.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

def getThreshold():
    return threshold

def predictOilCategory(year):
    year_scaled = scaler.transform([[year]])
    prob = model.predict_proba(year_scaled)[0][1]
    category = model.predict(year_scaled)[0]
    return int(category), float(prob)

def generatePlot(year_value=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #sort the data
    df_sorted = df.sort_values(by="year")
    
    X_plot = df_sorted["year"].values
    Y_plot = df_sorted["High_Imports"].values
    
    ax.scatter(X_plot, Y_plot, alpha=0.5, color='blue', label='Real Data (0=Dwarf, 1=Higt)')
    
    #create a range of years to better see a smooth curve
    year_min = df["year"].min() - 15
    year_max = df["year"].max() + 15
    X_smooth = np.linspace(year_min, year_max, 300).reshape(-1, 1)
    
    #scale de years for prediction
    X_smooth_scaled = scaler.transform(X_smooth)
    
    #calculete probabilitys for a smooth curve
    y_prob_smooth = model.predict_proba(X_smooth_scaled)[:, 1]
    
    #draw a smooth sigmoidea curve
    ax.plot(X_smooth, y_prob_smooth, color='red', linewidth=2.5, label='Curve for Logistical Regression')
    
    #umbral line dession
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='umbral (0.5)')
    
    #view the prediction
    if year_value is not None:
        year_scaled = scaler.transform([[year_value]])
        prob = model.predict_proba(year_scaled)[0][1]
        category = predictOilCategory(year_value)
        
        #color category
        point_color = 'green' if category == 1 else 'orange'
        
        ax.scatter(year_value, prob, color=point_color, s=120, 
                  edgecolors='darkgreen' if category == 1 else 'darkorange', 
                  linewidth=2, label=f'Prediction {year_value}')
        ax.text(year_value, prob + 0.05, f"({year_value}, {round(prob,3)})", 
               ha='center', fontsize=9, fontweight='bold')
    
    #here we configure labels and titles
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Probability of High Imports", fontsize=12)
    ax.set_title("Logistic Regression - Oil Imports Classification", fontsize=14, fontweight='bold')
    
    #ciionfiure limits
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(year_min, year_max)
    
    #agrage grid
    ax.grid(True, alpha=0.3)
    
    # agrage legend
    ax.legend(loc='best')
    
    #save the graph
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()