# Startup-success-prediction
End-to-end Machine Learning project that predicts startup success using startup funding data, feature engineering, Random Forest Classification, and a Streamlit web application.

## Overview

Startup Success Prediction is an end-to-end Machine Learning project that predicts whether a startup is likely to be successful based on funding information, industry, location, investment patterns, and other business attributes.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model training, model evaluation, and deployment through a Streamlit web application.

---

## Problem Statement

Investors and entrepreneurs often need to evaluate the potential success of startups before making investment decisions.

The objective of this project is to build a machine learning model that can classify startups as successful or unsuccessful using historical startup funding data.

---

## Dataset Information

The dataset contains startup funding records with the following features:

* Startup
* Industry
* SubVertical
* City
* Investors
* Investment Type
* Investment Amount (USD)
* Date

Additional engineered features:

* Year
* Month
* Investor Count
* Success (Target Variable)

---

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Encoding
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Streamlit Deployment

---

## Machine Learning Models Used

### Logistic Regression

Accuracy: 88.64%

### Random Forest Classifier

Accuracy: 90.45%

Random Forest was selected as the final model due to better predictive performance.

---

## Feature Importance

The most influential feature identified by the model was:

* InvestmentAmount_USD

Other important features included:

* SubVertical
* Month
* City
* Industry
* Year
* Investor Count

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit

---

## Project Structure

startup-success-prediction/

├── app.py

├── requirements.txt

├── dataset/

├── model/

│ └── startup_model.pkl

├── notebook/

│ └── startup_success_prediction.ipynb

└── README.md

---

## How to Run

Clone the repository:

git clone <repository-url>

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

---

## Results

* Logistic Regression Accuracy: 88.64%
* Random Forest Accuracy: 90.45%
* Successfully deployed using Streamlit
* Interactive prediction interface for startup success classification

---

## Future Improvements

* Hyperparameter tuning
* Additional startup features
* Real-time startup data integration
* Cloud deployment
* Enhanced dashboard and analytics

---

## Screenshots

### Streamlit Application

<img width="1617" height="963" alt="image" src="https://github.com/user-attachments/assets/f832fa80-bc5c-48cb-a0de-999fae34baa7" />

### Prediction Results

<img width="1834" height="910" alt="image" src="https://github.com/user-attachments/assets/d231608d-7ca5-4663-a8af-262e53ba3e30" />

<img width="1833" height="918" alt="image" src="https://github.com/user-attachments/assets/3d6e6b20-c3d3-47a5-bd2c-0376d1b9f509" />

---


