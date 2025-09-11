# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
import shap
import os

@st.cache_data
def load_processed_data():
    """Load processed data for CLV analysis."""
    processed_dir = 'data/processed/'
    churn_rates = pd.read_csv(os.path.join(processed_dir, 'plots/churn_rates.csv'))
    return churn_rates

@st.cache_resource
def load_models():
    """Load trained models, preprocessor, and encoders."""
    models_dir = 'models/'
    processed_dir = 'data/processed/'
    logistic = joblib.load(os.path.join(models_dir, 'logistic.pkl'))
    rf = joblib.load(os.path.join(models_dir, 'rf.pkl'))
    xgb = joblib.load(os.path.join(models_dir, 'xgb.pkl'))
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    encoders = joblib.load(os.path.join(processed_dir, 'encoders.pkl'))
    return logistic, rf, xgb, preprocessor, encoders

churn_rates = load_processed_data()
logistic, rf, xgb, preprocessor, encoders = load_models()

st.title("Telco Customer Churn Prediction")
tabs = st.tabs(["Predict", "Model Performance", "CLV Overview"])

with tabs[0]:
    st.header("Predict Churn")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        with col2:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=100.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1200.0)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            customer = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [senior_citizen],
'act': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'tenure_bucket': [pd.cut([tenure], bins=[0, 6, 12, 24, float('inf')], labels=['0-6m', '6-12m', '12-24m', '24m+'])[0]],
                'services_count': [sum(1 for x in [phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies] if x in ['Yes', 'DSL', 'Fiber optic'])],
                'monthly_to_total_ratio': [total_charges / max(1, tenure * monthly_charges)],
                'internet_no_tech_support': [1 if internet_service in ['DSL', 'Fiber optic'] and tech_support == 'No' else 0],
                'ExpectedTenure': [tenure + 12 if contract == 'Month-to-month' else tenure + 24],
                'CLV': [monthly_charges * (tenure + 12 if contract == 'Month-to-month' else tenure + 24)]
            })
            
            for col, le in encoders.items():
                if col in customer.columns:
                    customer[col] = le.transform(customer[col])
            
            X_train = pd.read_csv('data/processed/X_train.csv')
            customer = customer[X_train.columns]
            
            proba = xgb.predict_proba(customer)[:, 1][0]
            label = "High" if proba >= 0.66 else "Med" if proba >= 0.33 else "Low"
            
            st.write(f"**Churn Probability**: {proba * 100:.1f}% ({label})")
            st.write(f"**CLV**: ${customer['CLV'].iloc[0]:.2f}")
            st.write("**CLV Formula**: MonthlyCharges × ExpectedTenure")
            
            explainer = shap.TreeExplainer(xgb.named_steps['model'])
            customer_transformed = preprocessor.transform(customer)
            shap_values = explainer.shap_values(customer_transformed)
            shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
            st.write("**Local Explanation (SHAP)**")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, data=customer_transformed[0], feature_names=preprocessor.get_feature_names_out()))
            st.pyplot(fig)

with tabs[1]:
    st.header("Model Performance")
    metrics = {
        'Logistic Regression': {'Precision': 0.547, 'Recall': 0.725, 'F1': 0.624, 'AUC-ROC': 0.828},
        'Random Forest': {'Precision': 0.570, 'Recall': 0.666, 'F1': 0.614, 'AUC-ROC': 0.827},
        'XGBoost': {'Precision': 0.528, 'Recall': 0.690, 'F1': 0.598, 'AUC-ROC': 0.824}
    }
    st.write("**Metrics Table**")
    st.table(pd.DataFrame(metrics).T)
    
    fig, ax = plt.subplots()
    for model, name in [(logistic, "Logistic Regression"), (rf, "Random Forest"), (xgb, "XGBoost")]:
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_test = pd.read_csv('data/processed/y_test.csv')['Churn']
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    st.pyplot(fig)
    
    st.write("**Confusion Matrix (XGBoost)**")
    y_pred = xgb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)
    
    st.write("**Global Feature Importance**")
    logistic_importance = pd.read_csv('data/processed/plots/logistic_importance.csv')
    st.write("**Logistic Regression**")
    fig, ax = plt.subplots()
    plt.barh(logistic_importance['Feature'][:10], logistic_importance['Importance'][:10])
    plt.title('Logistic Regression Feature Importance')
    st.pyplot(fig)
    
    st.write("**Random Forest**")
    rf_plot = 'data/processed/plots/random_forest_shap_summary.png'
    if os.path.exists(rf_plot):
        st.image(rf_plot)
    else:
        st.error(f"File not found: {rf_plot}")
    
    st.write("**XGBoost**")
    xgb_plot = 'data/processed/plots/xgboost_shap_summary.png'
    if os.path.exists(xgb_plot):
        st.image(xgb_plot)
    else:
        st.error(f"File not found: {xgb_plot}")

with tabs[2]:
    st.header("CLV Overview")
    clv_dist_plot = 'data/processed/plots/clv_distribution.png'
    churn_quartile_plot = 'data/processed/plots/churn_by_quartile.png'
    
    if os.path.exists(clv_dist_plot):
        st.image(clv_dist_plot)
    else:
        st.error(f"File not found: {clv_dist_plot}")
    
    if os.path.exists(churn_quartile_plot):
        st.image(churn_quartile_plot)
    else:
        st.error(f"File not found: {churn_quartile_plot}")
    
    st.write("**Churn Rate by CLV Quartile**")
    st.table(churn_rates)
    st.write("**Takeaway**: High-value customers (Premium CLV quartile) have the lowest churn rates, making them critical for retention strategies. Focus on personalized offers and loyalty programs for this group to maximize revenue.")