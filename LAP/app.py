from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

from src.logger import logger


# ==================================
# DASHBOARD STATISTICS
# ==================================

total_predictions = 0
approved_loans = 0
rejected_loans = 0


# ==================================
# FLASK APP
# ==================================

application = Flask(__name__)
app = application


# ==================================
# HOME PAGE
# ==================================

@app.route("/")
def index():
    return render_template("index.html")


# ==================================
# DASHBOARD PAGE
# ==================================

@app.route("/dashboard")
def dashboard():

    model_accuracy = 99.20

    approval_rate = 0

    if total_predictions > 0:
        approval_rate = round(
            (approved_loans / total_predictions) * 100,
            2
        )

    return render_template(
        "dashboard.html",
        model_accuracy=model_accuracy,
        total_predictions=total_predictions,
        approved_loans=approved_loans,
        rejected_loans=rejected_loans,
        approval_rate=approval_rate
    )


# ==================================
# PREDICTION PAGE
# ==================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    global total_predictions
    global approved_loans
    global rejected_loans

    if request.method == "GET":
        return render_template("home.html")

    try:

        logger.info("Receiving user inputs")

        data = CustomData(
            Age=float(request.form.get("Age")),
            Experience=float(request.form.get("Experience")),
            Income=float(request.form.get("Income")),
            Family=float(request.form.get("Family")),
            CCAvg=float(request.form.get("CCAvg")),
            Education=float(request.form.get("Education")),
            Mortgage=float(request.form.get("Mortgage")),
            Securities_Account=float(
                request.form.get("Securities_Account")
            ),
            CD_Account=float(
                request.form.get("CD_Account")
            ),
            Online=float(
                request.form.get("Online")
            ),
            CreditCard=float(
                request.form.get("CreditCard")
            )
        )

        logger.info("Creating DataFrame")

        pred_df = data.get_data_as_dataframe()

        logger.info("Running prediction pipeline")

        predict_pipeline = PredictPipeline()

        result, confidence = predict_pipeline.predict(
            pred_df
        )

        # ==================================
        # UPDATE DASHBOARD COUNTERS
        # ==================================

        total_predictions += 1

        if result[0] == 1:
            approved_loans += 1
        else:
            rejected_loans += 1

        # ==================================
        # PREDICTION RESULT
        # ==================================

        prediction = (
            "✅ Loan Approved"
            if result[0] == 1
            else "❌ Loan Rejected"
        )

        reasons = []
        recommendations = []
        ai_message = ""

        # ==================================
        # APPROVED LOAN
        # ==================================

        if result[0] == 1:

            ai_message = f"""
The applicant demonstrates a strong financial profile.

Annual Income: {data.Income}

Professional Experience: {data.Experience} years

The machine learning model has identified
strong repayment potential based on the
customer's financial indicators and banking
profile.

Risk Level: Low

Recommendation:
Proceed with loan approval.

The applicant may also qualify for premium
banking products and higher-value loan
facilities in the future.
"""

            if data.Income >= 100:
                reasons.append(
                    "Strong annual income."
                )

            if data.Experience >= 5:
                reasons.append(
                    "Good professional experience."
                )

            if data.CCAvg <= 3:
                reasons.append(
                    "Healthy credit card spending pattern."
                )

            if data.CD_Account == 1:
                reasons.append(
                    "Maintains a Certificate Deposit account."
                )

            recommendations.extend([
                "Maintain your current financial discipline.",
                "Continue responsible credit usage.",
                "Explore premium banking services."
            ])

        # ==================================
        # REJECTED LOAN
        # ==================================

        else:

            ai_message = f"""
The applicant currently presents a higher
lending risk profile.

Annual Income: {data.Income}

Professional Experience: {data.Experience} years

The machine learning model has identified
financial characteristics that may reduce
repayment confidence.

Risk Level: Moderate to High

Recommendation:
Improve financial standing before
submitting another application.

Suggested focus areas include increasing
income, reducing liabilities, and building
stronger banking relationships.
"""

            if data.Income < 100:

                reasons.append(
                    "Income level may be insufficient."
                )

                recommendations.append(
                    "Increase annual income."
                )

            if data.Experience < 5:

                reasons.append(
                    "Limited work experience."
                )

                recommendations.append(
                    "Gain additional professional experience."
                )

            if data.Mortgage > 250:

                reasons.append(
                    "High mortgage obligations."
                )

                recommendations.append(
                    "Reduce outstanding debt."
                )

            if data.CD_Account == 0:

                reasons.append(
                    "No Certificate Deposit account."
                )

                recommendations.append(
                    "Consider opening a CD account."
                )

        logger.info(
            f"Prediction Result: {prediction}"
        )

        logger.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        logger.info(
            f"Total Predictions: {total_predictions}"
        )

        logger.info(
            f"Approved Loans: {approved_loans}"
        )

        logger.info(
            f"Rejected Loans: {rejected_loans}"
        )

        return render_template(
            "home.html",
            results=prediction,
            confidence=round(confidence, 2),
            reasons=reasons,
            recommendations=recommendations,
            ai_message=ai_message
        )

    except Exception as e:

        logger.error(
            f"Prediction Error : {str(e)}"
        )

        return render_template(
            "home.html",
            results="❌ Prediction Error",
            confidence=0,
            reasons=[],
            recommendations=[],
            ai_message=""
        )


# ==================================
# MAIN
# ==================================

if __name__ == "__main__":

    logger.info(
        "Starting Loan Approval Flask Application"
    )

    app.run(
        host="0.0.0.0",
        port=5002,
        debug=True
    )