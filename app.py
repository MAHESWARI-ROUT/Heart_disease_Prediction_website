from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)
model = pickle.load(open("model/heart_model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs and convert to proper format
        features = [
            float(request.form['age']),
            float(request.form['sex']),         
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),          
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),        
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        prediction = model.predict([features])[0]

        if prediction == 1:
            outcome = "⚠️ Patient is likely to have heart disease."
            tips = """
            🩺 Recommendations:
            • Consult a cardiologist as soon as possible.
            • Get diagnostic tests like ECG, Echo, or Angiography.
            • Adopt a heart-healthy lifestyle (diet, exercise, quit smoking).
            • Consider medication management for blood pressure or cholesterol.
            """
        else:
            outcome = "✅ Patient is unlikely to have heart disease."
            tips = """
            ✅ Prevention Tips:
            • Maintain a healthy diet (low salt, low fat).
            • Exercise regularly (30 min/day).
            • Monitor blood pressure, cholesterol & blood sugar.
            • Avoid smoking and manage stress.
            """

        return render_template('result.html', prediction_text=outcome, tips=tips)
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"
@app.route('/metrics')
def metrics():
    try:
        with open("model/metrics.txt", "r") as f:
            content = f.read()
        return render_template("metrics.html", metrics_text=content)
    except Exception as e:
        return f"Error loading metrics: {e}"


if __name__ == '__main__':
    app.run(debug=True)
