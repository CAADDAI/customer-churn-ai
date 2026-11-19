import pandas as pd
import joblib

from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Form Page
@app.route("/form")
def form():
    return render_template(
        "index.html",
        prediction_text=None
    )


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        features = [
            request.form["A1"],
            float(request.form["A2"]),
            float(request.form["A3"]),
            request.form["A4"],
            request.form["A5"],
            request.form["A6"],
            request.form["A7"],
            float(request.form["A8"]),
            request.form["A9"],
            request.form["A10"],
            float(request.form["A11"]),
            request.form["A12"],
            request.form["A13"],
            float(request.form["A14"]),
            float(request.form["A15"])
        ]

        data = pd.DataFrame(
            [features],
            columns=[
                "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9",
                "10", "11", "12", "13", "14"
            ]
        )

        print("\nINPUT DATA:")
        print(data)

        transformed_data = preprocessor.transform(data)

        prediction = model.predict(transformed_data)[0]

        print("\nRAW PREDICTION:", prediction)

        if prediction == 1 or prediction == 1.0:
            result = "Approved ✅"
        else:
            result = "Rejected ❌"

        return render_template(
            "index.html",
            prediction_text=f"Credit Card Status: {result}"
        )

    except Exception as e:

        print("\nERROR OCCURRED:")
        print(str(e))

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5005,
        debug=True
    )