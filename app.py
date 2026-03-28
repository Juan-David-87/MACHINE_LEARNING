from flask import Flask, render_template, request
from LinearRegressionOil import predictHappiness, generatePlot
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("Menu.html")

@app.route("/Menu")
def firstPage(): 
    return render_template("Menu.html")

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

@app.route("/BasicConcepts")
def basicConcepts():
    return render_template("BasicConcepts.html")

@app.route("/Application")
def application():
    return render_template("Application.html")

@app.route("/OilLinearRegression", methods=["GET", "POST"])
def index():
    result = None
    plot = None

    if request.method == "POST":
        gdp = float(request.form["gdp"])
        result = predictHappiness(gdp)

        # now de graphic depent to valour
        plot = generatePlot(gdp)

    else:
        #graphic base
        plot = generatePlot()

    return render_template("LinearRegressionOil.html", result=result, plot=plot)