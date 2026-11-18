📊Customer Churn Analytics Platform
An end-to-end Machine Learning web application that predicts customer churn, generates AI-powered retention recommendations, stores prediction history, and provides interactive business analytics dashboards.
🚀Project Overview
 Customer churn is one of the biggest challenges faced by financial institutions. Retaining existing customers is significantly cheaper than acquiring new ones.

This project uses Machine Learning to predict whether a bank customer is likely to churn and provides intelligent recommendations to improve customer retention.

The application includes:

Customer churn prediction
Risk classification
AI-generated customer retention recommendations
Interactive analytics dashboard
Prediction history tracking
Churn distribution visualization
Confidence trend monitoring
Data persistence using Excel

🎯Objectives
The primary objectives of this project are:

Predict customer churn using Machine Learning
Improve customer retention through actionable insights
Visualize customer risk trends
Track prediction performance over time
Demonstrate deployment of an end-to-end ML solution

🏗️System Architecture
 User Input
      │
      ▼
Flask Web Application
      │
      ▼
Data Preprocessing Pipeline
      │
      ▼
Random Forest Model
      │
      ▼
Prediction + Confidence Score
      │
      ▼
AI Recommendation Engine
      │
      ▼
Excel Database Storage
      │
      ▼
Analytics Dashboard

🧠Machine Learning Model

Algorithm Used
Random Forest Classifier

Why Random Forest?
Handles non-linear relationships
Robust against overfitting
Works well on tabular data
Produces probability scores
High predictive performance

📂 Features
✅ Customer Prediction

Predict whether a customer is likely to churn based on:

Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Credit Card Ownership
Active Membership Status
Estimated Salary

✅ Risk Classification
Customers are automatically grouped into:

🔴 High Risk
Customers likely to churn.

🟢 Low Risk
Customers likely to remain with the bank.

✅ Confidence Score
Each prediction includes a confidence score.

Example:

Prediction:
Customer Likely To Stay

Confidence:
83.37%

✅ AI Recommendation Engine
The system generates tailored customer retention recommendations.

Examples:

Loyalty Programs
Personalized Offers
Product Cross-Selling
Customer Engagement Campaigns
Retention Strategies

✅ Interactive Dashboard
The dashboard provides:

KPI Metrics
Total Predictions
High Risk Customers
Low Risk Customers
Average Confidence

Customer Risk Distribution
Interactive pie chart displaying:

High Risk Customers
vs
Low Risk Customers

Confidence Trend Analysis
Interactive chart showing:

Prediction Confidence Over Time

Recent Predictions
Tracks historical predictions including:

Date
Country
Risk Status
Confidence
Older predictions can be viewed using an expandable section.

📈 Dashboard Preview
Dashboard Includes
KPI Cards
Churn Distribution Pie Chart
Confidence Trend Chart
Prediction History Table
Responsive Layout

🛠️ Technologies Used
Backend
Python
Flask
Machine Learning
Scikit-Learn
Pandas
NumPy

Visualization
Plotly
Frontend
HTML5
CSS3
Jinja2
Data Storage
Excel (.xlsx)

📁 Project Structure
Customer_Churn_AI/
│
├── artifacts/
│   ├── dashboard_data.xlsx
│
├── notebooks/
│
├── src/
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   ├── ai_recommendation.py
│   │
│   └── components/
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── landing.html
│   ├── home.html
│   ├── index.html
│   └── dashboard.html
│
├── app.py
│
├── requirements.txt
│
└── README.md

⚙️ Installation
Clone Repository
git clone https://github.com/CADDAI/customer-churn-ai.git
cd customer-churn-ai

Install Dependencies
pip install -r requirements.txt

▶️ Running the Application
Start Flask server:

python app.py
Open browser:

http://127.0.0.1:5006

📊 Example Workflow
Step 1
Enter customer details.

Step 2
Submit prediction request.

Step 3
Model predicts:

Likely To Stay
or

Likely To Churn
Step 4
Receive AI recommendations.

Step 5
Prediction is stored automatically.

Step 6
View analytics dashboard.

📌 Future Improvements
Potential enhancements:

User Authentication
SQL Database Integration
PDF Report Downloads
Cloud Deployment (Azure/AWS)
REST API Endpoint
Customer Segmentation
Model Monitoring
Real-Time Data Streaming

🎓 Learning Outcomes
This project demonstrates:

Machine Learning Model Development
Data Preprocessing
Model Deployment
Flask Web Development
Interactive Dashboards
Data Visualization
Business Intelligence Reporting
AI-Powered Decision Support

👨‍💻 Author
Collins Aboagye Addai
Machine Learning,Data Science Engineer & AI Engineer 
MSc Big Data Candidate | University of Stirling

GitHub: https://github.com/CAADDAI
LinkedIn: https://linkedin.com/in/caaddai
Email:    addaiaboagyecollins@gmail.com

📜 License
This project is licensed under the MIT License.

⭐ If you found this project useful, please consider giving it a star.


