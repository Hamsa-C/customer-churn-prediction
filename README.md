==============================================================
CUSTOMER CHURN PREDICTION
==============================================================

A machine learning project that predicts whether a customer is
likely to leave (churn) a subscription-based business, so that
the company can take action to retain them.


--------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------
Customer churn means a customer stops using a company's service.
Keeping an existing customer is usually cheaper than acquiring a
new one, so predicting churn early helps reduce revenue loss.

This project:
  - Cleans and prepares raw customer data
  - Explores patterns linked to churn
  - Trains and compares classification models
  - Evaluates them with metrics suited to imbalanced data
  - Identifies the main factors that drive churn


--------------------------------------------------------------
2. DATASET
--------------------------------------------------------------
Dataset : Telco Customer Churn (IBM sample data, available on Kaggle)
Rows    : ~7,000 customers
Target  : Churn (Yes / No)

Main feature groups:
  - Demographics   : gender, senior citizen, partner, dependents
  - Account info   : tenure, contract type, payment method,
                     paperless billing, monthly and total charges
  - Services       : phone, internet, online security, tech support,
                     streaming TV and movies, etc.

Place the CSV file in the project folder (or the data/ folder)
before running the notebook.


--------------------------------------------------------------
3. PROJECT STRUCTURE
--------------------------------------------------------------
customer-churn-prediction/
|
|-- data/
|   |-- WA_Fn-UseC_-Telco-Customer-Churn.csv   Dataset
|
|-- models/
|   |-- churn_model_gb.pkl                     Saved trained model
|   |-- scaler.pkl                             Saved feature scaler
|
|-- notebooks/
|   |-- churn_prediction.ipynb                 Main analysis and model code
|   |-- churn_walkthrough.ipynb                Step-by-step walkthrough
|
|-- outputs/                                   Generated plots and results
|-- .gitignore
|-- requirements.txt                           Python dependencies
|-- README.md                                  This file

--------------------------------------------------------------
4. TECHNOLOGIES USED
--------------------------------------------------------------
  - Python 3.8+
  - pandas, numpy
  - scikit-learn
  - matplotlib, seaborn
  - Jupyter Notebook
  - joblib (saving the model)


--------------------------------------------------------------
5. INSTALLATION
--------------------------------------------------------------
Step 1: Clone the repository
    git clone https://github.com/Hamsa-C/Customer-churn-prediction.git
    cd Customer-churn-prediction

Step 2: Create a virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate          (Windows: venv\Scripts\activate)

Step 3: Install dependencies
    pip install -r requirements.txt

    If there is no requirements.txt, run:
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib


--------------------------------------------------------------
6. HOW TO RUN
--------------------------------------------------------------
    jupyter notebook

Open the notebook and run the cells from top to bottom.
You can also open it in Google Colab and upload the CSV file.


--------------------------------------------------------------
7. METHODOLOGY
--------------------------------------------------------------
1. Data loading and inspection
2. Data cleaning
     - Drop the customer ID column
     - Convert TotalCharges to numeric and handle missing values
     - Encode Churn as 1 (Yes) and 0 (No)
3. Exploratory data analysis (EDA)
     - Churn distribution
     - Churn vs contract type, tenure, monthly charges, services
4. Feature engineering
     - One-hot encoding of categorical variables
     - Scaling numeric features (for Logistic Regression)
5. Train/test split (80/20, stratified)
6. Model training
     - Logistic Regression
     - Random Forest (class_weight = balanced)
7. Model evaluation and comparison
8. Feature importance analysis
9. Saving the final model


--------------------------------------------------------------
8. EVALUATION METRICS
--------------------------------------------------------------
Because only about 26% of customers churn, accuracy alone can be
misleading. The models are evaluated using:
  - Precision
  - Recall (important: catching customers who will leave)
  - F1-score
  - ROC-AUC
  - Confusion matrix


--------------------------------------------------------------
9. KEY INSIGHTS
--------------------------------------------------------------
  1. Overall churn rate: about 26.5% of customers leave, so the
     data is imbalanced (roughly 1 in 4 customers churn).

  2. Contract type is the strongest factor: month-to-month
     customers churn at 42.7%, versus 11.3% on one-year and only
     2.8% on two-year contracts.

  3. New customers leave first: churned customers had an average
     tenure of 18.0 months, compared with 37.6 months for
     customers who stayed.

  4. Higher bills increase churn: churned customers paid about
     $74.4 per month on average, versus $61.3 for retained
     customers.

  5. Fiber optic customers churn the most among internet users:
     41.9%, compared with 19.0% for DSL.

  6. Online security matters: customers without it churn at 41.8%,
     versus 14.6% for those who have it.

  7. Tech support matters too: customers without it churn at
     41.6%, versus 15.2% for those who have it.

  8. Payment method is a warning sign: electronic check users churn
     at 45.3%, far above automatic bank transfer (16.7%) and
     automatic credit card (15.2%).

  9. Customer profile: senior citizens churn at 41.7% (vs 23.6%
     for non-seniors), and customers without dependents churn at
     31.3% (vs 15.5% for those with dependents).

Recommendation: move new customers to longer contracts and
automatic payments, and promote online security and tech support
add-ons to reduce churn.


--------------------------------------------------------------
10. RESULTS
--------------------------------------------------------------
Model                 Accuracy   Precision   Recall   F1    ROC-AUC
Logistic Regression      0.81	     0.68	      0.54	 0.60	   0.85
Random Forest            0.78	     0.62	      0.47	 0.54	   0.82

--------------------------------------------------------------
11. USING THE SAVED MODEL
--------------------------------------------------------------
    import joblib
    model = joblib.load("churn_model.pkl")
    prediction = model.predict(new_customer_data)
    probability = model.predict_proba(new_customer_data)[:, 1]


--------------------------------------------------------------
12. FUTURE IMPROVEMENTS
--------------------------------------------------------------
  - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
  - Try XGBoost or Gradient Boosting
  - Handle imbalance with SMOTE
  - Build a Streamlit or Flask web app for live predictions
  - Add customer segmentation (K-Means)
