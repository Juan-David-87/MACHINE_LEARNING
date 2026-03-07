from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Flask"

@app.route('/FirstPage/<name>')
def firstPage(name):
    return render_template('index.html', name=name)
