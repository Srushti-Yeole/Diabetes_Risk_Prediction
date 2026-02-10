from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        data = {
            "Pregnancies": float(request.form["pregnancies"]),
            "Glucose": float(request.form["glucose"]),
            "BloodPressure": float(request.form["bloodpressure"]),
            "SkinThickness": float(request.form["skinthickness"]),
            "Insulin": float(request.form["insulin"]),
            "BMI": float(request.form["bmi"]),
            "DiabetesPedigreeFunction": float(request.form["pedigree"]),
            "Age": float(request.form["age"]),
        }

        # Validate inputs
        for field, value in data.items():
            if value < 0:
                return render_template("index.html", error=f"{field} cannot be negative")
            if field != "Pregnancies" and value == 0:
                return render_template("index.html", error=f"{field} cannot be zero")

        # Create DataFrame
        df = pd.DataFrame([data])

        # Add engineered features
        df['Glucose_BMI'] = df['Glucose'] * df['BMI']
        df['Age_BMI'] = df['Age'] * df['BMI']
        df['Glucose_Age'] = df['Glucose'] * df['Age']

        # Make prediction using pipeline (includes scaling)
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[0]

        # Get confidence score
        confidence = prediction_proba[1] if prediction[0] == 1 else prediction_proba[0]
        confidence_text = f"{confidence*100:.1f}%"

        # Prepare result text
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        # Add risk level
        risk_level = ""
        if prediction[0] == 1:
            if confidence > 0.8:
                risk_level = "High Risk"
            else:
                risk_level = "Moderate Risk"
        else:
            if confidence > 0.8:
                risk_level = "Low Risk"
            else:
                risk_level = "Moderate-Low Risk"

        return render_template("index.html",
                             result=result,
                             confidence=confidence_text,
                             risk_level=risk_level)

    except ValueError:
        return render_template("index.html", error="Please enter valid numerical values for all fields")
    except Exception as e:
        return render_template("index.html", error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)