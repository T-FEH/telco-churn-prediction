# src/interpretability.py
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.utils.validation import check_is_fitted

def load_data_and_models(processed_dir, models_dir):
    """Load processed data and trained models."""
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    logistic = joblib.load(os.path.join(models_dir, 'logistic.pkl'))
    rf = joblib.load(os.path.join(models_dir, 'rf.pkl'))
    xgb = joblib.load(os.path.join(models_dir, 'xgb.pkl'))
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    return X_test, logistic, rf, xgb, preprocessor

def compute_logistic_importance(logistic, X_test, preprocessor):
    """Compute feature importance for Logistic Regression using standardized coefficients."""
    # Ensure preprocessor is fitted
    check_is_fitted(preprocessor)
    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    # Extract coefficients
    coef = logistic.named_steps['model'].coef_[0]
    # Get standard deviations of features
    X_test_transformed = preprocessor.transform(X_test)
    std_devs = np.std(X_test_transformed, axis=0)
    # Compute importance: |coefficient * std_dev|
    importance = np.abs(coef * std_devs)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    return importance_df

def plot_logistic_importance(importance_df, output_dir):
    """Plot and save Logistic Regression feature importance."""
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='#1f77b4')
    plt.title('Logistic Regression Feature Importance')
    plt.xlabel('Importance (|coefficient * std_dev|)')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'logistic_importance.png'))
    plt.close()

def compute_shap_values(model, X_test, preprocessor, model_name):
    """Compute SHAP values for tree-based models."""
    # Ensure preprocessor is fitted
    check_is_fitted(preprocessor)
    X_test_transformed = preprocessor.transform(X_test)
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(X_test_transformed)
    # For binary classification, use positive class SHAP values
    if model_name == 'xgboost':
        return shap_values[1] if isinstance(shap_values, list) else shap_values
    return shap_values

def plot_shap_summary(shap_values, X_test, preprocessor, model_name, output_dir):
    """Plot and save SHAP summary for tree-based models."""
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.title(f'{model_name} SHAP Summary Plot')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_shap_summary.png'))
    plt.close()

def get_local_shap_explanation(model, preprocessor, customer, model_name):
    """Compute SHAP values for a single customer (for app)."""
    check_is_fitted(preprocessor)
    customer_transformed = preprocessor.transform(customer)
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer.shap_values(customer_transformed)
    # For binary classification, use positive class SHAP values
    if model_name == 'xgboost':
        return shap_values[1] if isinstance(shap_values, list) else shap_values, customer_transformed
    return shap_values, customer_transformed

def main():
    # File paths
    processed_dir = 'data/processed/'
    models_dir = 'models/'
    output_dir = 'data/processed/plots/'
    
    # Load data and models
    X_test, logistic, rf, xgb, preprocessor = load_data_and_models(processed_dir, models_dir)
    
    # Logistic Regression feature importance
    logistic_importance = compute_logistic_importance(logistic, X_test, preprocessor)
    plot_logistic_importance(logistic_importance, output_dir)
    
    # SHAP for Random Forest
    shap_values_rf = compute_shap_values(rf, X_test, preprocessor, 'random_forest')
    plot_shap_summary(shap_values_rf, X_test, preprocessor, 'Random Forest', output_dir)
    
    # SHAP for XGBoost
    shap_values_xgb = compute_shap_values(xgb, X_test, preprocessor, 'xgboost')
    plot_shap_summary(shap_values_xgb, X_test, preprocessor, 'XGBoost', output_dir)
    
    # Save importance for app
    logistic_importance.to_csv(os.path.join(output_dir, 'logistic_importance.csv'), index=False)
    
    print(f"Interpretability analysis complete. Plots saved to {output_dir}")

if __name__ == "__main__":
    main()