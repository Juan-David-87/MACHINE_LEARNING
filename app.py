from flask import Flask, render_template, request
from LinearRegressionHappiness import predictHappiness
app = Flask(__name__)

@app.route('/')
def home():
    return "Machine Learning Use Cases in Flask"

@app.route("/Menu")
def firstPage(): 
    return render_template("Menu.html") # este recarga el home toca hacer que este en home y vaya al menu 

@app.route("/FirstCase")
def secondPage(): 
    return render_template("UseCaseJuanda.html")

@app.route("/SecondCase")
def thirdPage(): 
    return render_template("UseCaseElkin.html")

@app.route("/ThirdCase")
def fourthPage(): 
    return render_template("UseCaseYira.html")

@app.route("/FourthCase")
def fifthPage(): 
    return render_template("UseCaseJuan.html")

@app.route("/LinearRegression", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        gdp = float(request.form["gdp"])
        result = predictHappiness(gdp)

    return render_template("HappinessLinearRegression.html", result=result)