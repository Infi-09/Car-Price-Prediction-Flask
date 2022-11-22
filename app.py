import pickle
import pandas as pd
import numpy as np 
from flask import Flask, render_template, request, redirect, url_for

filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

app  = Flask(__name__)

@app.route("/")
def input():
    return render_template("index.html")

@app.route("/result/<float:price>")
def result(price):
    price = "$ "+str(price)
    return render_template("result.html", amount=price)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":

        wheelbase = float(request.form["wheelbase"])
        carlength = float(request.form["carlength"])
        curbweight = float(request.form["curbweight"])
        boreratio = float(request.form["boreratio"])
        enginesize = float(request.form["enginesize"])
        horsepower = float(request.form["horsepower"])

        data = np.array([wheelbase, carlength, curbweight, boreratio, enginesize, horsepower])
        data = data.reshape(1, 6) 
        output = model.predict(data)

        return redirect(url_for("result", price=output))

if __name__ == "__main__":
    app.run(debug=True)