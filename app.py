
import subprocess
import sys

from flask import Flask, render_template, request
import LinearRegression
import LogisticRegression 

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Flask"

@app.route('/FirstPage/<name>')
def firstPage(name):
    return render_template('UseCaseYira.html', name=name)

@app.route("/LinearRegression/", methods=['GET', 'POST'])
def calculateGrade(): 
    calculateGrade = None
    if request.method == 'POST':
        hours = float(request.form['hours'])
        calculateGrade = LinearRegression.calculateGrade(hours)
    return render_template('linearRegressionGrade.html', result=calculateGrade)

@app.route("/LogisticRegression/", methods=['GET', 'POST'])
def logisticRegression(): 
    result = None

    if request.method == 'POST':
        try:
            proceso = subprocess.run(
                [sys.executable, "LogisticRegression.py"],
                capture_output=True,
                text=True
            )

            result = proceso.stdout

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('LogisticRegression.html', result=result)

