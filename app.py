from flask import Flask, render_template, request
from LinearRegressionOil import predictOil, generatePlot as generatePlotLinear
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
@app.route("/OilLinearRegression", methods=["GET", "POST"])
def LinearRegression():
    result = None
    plot = None

    if request.method == "POST":
        year = int(request.form["year"])
        result = predictOil(year)

        plot = generatePlotLinear(year)

    else:
        
        plot = generatePlotLinear()

    return render_template("OilLinearRegression.html", result=result, plot=plot)