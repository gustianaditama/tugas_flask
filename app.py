from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("view.html")
    elif request.method == 'POST':
        features = dict(request.form).values()
        model = joblib.load("model/linreg_model.pkl")
        result = model.predict(features)
        return render_template('view.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)