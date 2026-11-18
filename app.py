from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)

from src.pipeline.ai_recommendation import (
    AIRecommendation
)

import plotly.express as px

import pandas as pd
import os

from datetime import datetime

app = Flask(__name__)


# ==========================================
# LANDING PAGE
# ==========================================

@app.route("/")
def landing():

    return render_template(
        "landing.html"
        
    )
# ==========================================
# PREDICTION PAGE
# ==========================================

@app.route("/predict-page")
def predict_page():

    return render_template(
        "home.html"
    )

# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = CustomData(

            CreditScore=float(
                request.form.get("CreditScore")
            ),

            Geography=request.form.get(
                "Geography"
            ),

            Gender=request.form.get(
                "Gender"
            ),

            Age=float(
                request.form.get("Age")
            ),

            Tenure=float(
                request.form.get("Tenure")
            ),

            Balance=float(
                request.form.get("Balance")
            ),

            NumOfProducts=float(
                request.form.get("NumOfProducts")
            ),

            HasCrCard=int(
                request.form.get("HasCrCard")
            ),

            IsActiveMember=int(
                request.form.get("IsActiveMember")
            ),

            EstimatedSalary=float(
                request.form.get("EstimatedSalary")
            )

        )

        pred_df = (
            data.get_data_as_dataframe()
        )

        predictor = PredictPipeline()

        prediction, confidence = (
            predictor.predict(
                pred_df
            )
        )

        if prediction == 1:

            result = (
                "Customer Likely To Churn"
            )

            status = (
                "High Risk"
            )

            color = "danger"

        else:

            result = (
                "Customer Likely To Stay"
            )

            status = (
                "Low Risk"
            )

            color = "success"

        # ====================================
        # AI RECOMMENDATION ENGINE
        # ====================================

        recommendations = (
            AIRecommendation.generate(

                prediction=prediction,

                age=float(
                    request.form.get("Age")
                ),

                balance=float(
                    request.form.get("Balance")
                ),

                tenure=float(
                    request.form.get("Tenure")
                ),

                is_active_member=int(
                    request.form.get(
                        "IsActiveMember"
                    )
                ),

                num_products=float(
                    request.form.get(
                        "NumOfProducts"
                    )
                )

            )
        )

        # ====================================
        # SAVE TO EXCEL DATABASE
        # ====================================

        excel_path = (
            "artifacts/dashboard_data.xlsx"
        )

        if os.path.exists(
            excel_path
        ):

            dashboard_df = pd.read_excel(
                excel_path
            )

        else:

            dashboard_df = pd.DataFrame(
                columns=[
                    "Timestamp",
                    "CreditScore",
                    "Geography",
                    "Gender",
                    "Age",
                    "Tenure",
                    "Balance",
                    "NumOfProducts",
                    "HasCrCard",
                    "IsActiveMember",
                    "EstimatedSalary",
                    "Prediction",
                    "Status",
                    "Confidence"
                ]
            )

        new_record = {

            "Timestamp":
            datetime.now(),

            "CreditScore":
            request.form.get(
                "CreditScore"
            ),

            "Geography":
            request.form.get(
                "Geography"
            ),

            "Gender":
            request.form.get(
                "Gender"
            ),

            "Age":
            request.form.get(
                "Age"
            ),

            "Tenure":
            request.form.get(
                "Tenure"
            ),

            "Balance":
            request.form.get(
                "Balance"
            ),

            "NumOfProducts":
            request.form.get(
                "NumOfProducts"
            ),

            "HasCrCard":
            request.form.get(
                "HasCrCard"
            ),

            "IsActiveMember":
            request.form.get(
                "IsActiveMember"
            ),

            "EstimatedSalary":
            request.form.get(
                "EstimatedSalary"
            ),

            "Prediction":
            prediction,

            "Status":
            status,

            "Confidence":
            confidence

        }

        dashboard_df.loc[
            len(dashboard_df)
        ] = new_record

        dashboard_df.to_excel(
            excel_path,
            index=False
        )

        return render_template(

            "index.html",

            prediction=result,

            status=status,

            confidence=confidence,

            color=color,

            recommendations=recommendations

        )

    except Exception as e:

        return f"Prediction Error: {str(e)}"
    # ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    excel_path = "artifacts/dashboard_data.xlsx"

    if os.path.exists(excel_path):

        df = pd.read_excel(excel_path)

    else:

        df = pd.DataFrame()

    total_predictions = len(df)

    high_risk = len(
        df[df["Prediction"] == 1]
    ) if total_predictions > 0 else 0

    low_risk = len(
        df[df["Prediction"] == 0]
    ) if total_predictions > 0 else 0

    average_confidence = round(
        df["Confidence"].mean(),
        2
    ) if total_predictions > 0 else 0

    # ==============================
    # PIE CHART
    # ==============================

    chart_data = pd.DataFrame({

        "Category": [
            "High Risk",
            "Low Risk"
        ],

        "Count": [
            high_risk,
            low_risk
        ]

    })

    pie_chart = px.pie(

        chart_data,

        values="Count",

        names="Category",

        color="Category",

        color_discrete_map={

            "High Risk": "#e74c3c",

            "Low Risk": "#27ae60"

        },

        height=280

    )
    pie_chart.update_layout(

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )

    )

    pie_chart_html = pie_chart.to_html(
        full_html=False
    )
    # ==============================
    # CONFIDENCE TREND
    # ==============================
    if total_predictions > 0:

        trend_chart = px.line(

            df,

            x="Timestamp",

            y="Confidence",

            markers=True,

            height=480

        )

        trend_chart_html = (
            trend_chart.to_html(
                full_html=False
            )
        )

    else:

        trend_chart_html = ""

    # ==============================
    # RECENT PREDICTIONS
    # ==============================

    recent_predictions = []

    if total_predictions > 0:
            
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"]
            )
            df["Timestamp"] =(
            df["Timestamp"].dt.strftime(
                "%d-%b-%Y %H:%M" 
            )
            )
            recent_predictions = (
                df.tail(10)
                .iloc[::-1]
                .to_dict("records")
            )
    return render_template(

        "dashboard.html",

        total_predictions=total_predictions,

        high_risk=high_risk,

        low_risk=low_risk,

        average_confidence=average_confidence,

        recent_predictions=recent_predictions,

        pie_chart_html=pie_chart_html,

        trend_chart_html=trend_chart_html

    )

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5006,

        debug=True

    )


  