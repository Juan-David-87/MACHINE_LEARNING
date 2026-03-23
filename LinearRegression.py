import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}
# Create DataFrame
df = pd.DataFrame(data)

# Define input (X) and output (y)
x = df[["Study Hours"]]
y = df[["Final Grade"]]

# Create and train the model
model = LinearRegression()
model.fit(x,y)

# Prediction function
def calculateGrade(hours):
    result = model.predict([[hours]])[0][0]
    
    # Limit the result between 0 and 5
    result = max(0, min(5, result))

    # Round the result to 2 decimal places 
    result = round(result, 2)
    return result

