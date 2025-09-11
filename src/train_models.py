# src/train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os
from imblearn.over_sampling import SMOTE

def load_processed_data(processed_dir):
    """Load processed train, val, and test splits."""
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))['Churn']
    X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv'))['Churn']
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))['Churn']
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessor(X_train):
    """Create a preprocessing pipeline for numerical and categorical features."""
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', 'passthrough', categorical_cols)  # Already encoded in data_prep.py
        ])
    return preprocessor

def train_logistic(X_train, y_train, X_val, y_val):
    """Train and tune Logistic Regression."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {
        'model__C': [0.1, 1.0, 10.0],
        'model__penalty': ['l2']
    }
    pipeline = Pipeline([
        ('preprocessor', create_preprocessor(X_train)),
        ('model', model)
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Logistic Regression Best Params: {grid.best_params_}")
    return best_model

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train and tune Random Forest with class weighting."""
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, 30],
        'model__min_samples_leaf': [1, 2, 4],
        'model__min_samples_split': [2, 5]
    }
    pipeline = Pipeline([
        ('preprocessor', create_preprocessor(X_train)),
        ('model', model)
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='recall', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Random Forest Best Params: {grid.best_params_}")
    return best_model

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train and tune XGBoost."""
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    param_grid = {
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.01, 0.1],
        'model__n_estimators': [100, 200]
    }
    pipeline = Pipeline([
        ('preprocessor', create_preprocessor(X_train)),
        ('model', model)
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"XGBoost Best Params: {grid.best_params_}")
    return best_model

def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance."""
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    return X_train_resampled, y_train_resampled

def evaluate_model(model, X, y, model_name):
    """Evaluate model on Precision, Recall, F1, AUC-ROC."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = {
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'AUC-ROC': roc_auc_score(y, y_proba)
    }
    print(f"\n{model_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    return metrics

def test_high_risk_customer(model, encoders, X_train_columns):
    """Test a high-risk customer scenario."""
    customer = pd.DataFrame({
        'gender': ['Male'],
        'SeniorCitizen': [1],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [12],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [100.0],
        'TotalCharges': [1200.0],
        'tenure_bucket': ['12-24m'],
        'services_count': [2],
        'monthly_to_total_ratio': [1200.0 / (12 * 100.0)],
        'internet_no_tech_support': [1],
        'ExpectedTenure': [12 + 12],
        'CLV': [100.0 * (12 + 12)]
    })
    
    for col, le in encoders.items():
        if col in customer.columns:
            customer[col] = le.transform(customer[col])
    
    customer = customer[X_train_columns]
    proba = model.predict_proba(customer)[:, 1][0]
    print(f"\nHigh-Risk Customer Churn Probability: {proba * 100:.1f}%")
    return proba

def main():
    processed_dir = 'data/processed/'
    models_dir = 'models/'
    os.makedirs(models_dir, exist_ok=True)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(processed_dir)
    X_train_columns = X_train.columns.tolist()
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
    X_train_val = pd.concat([X_train_resampled, X_val], axis=0)
    y_train_val = pd.concat([y_train_resampled, y_val], axis=0)
    encoders = joblib.load(os.path.join(processed_dir, 'encoders.pkl'))
    
    logistic = train_logistic(X_train_resampled, y_train_resampled, X_val, y_val)
    rf = train_random_forest(X_train_resampled, y_train_resampled, X_val, y_val)
    xgb = train_xgboost(X_train_resampled, y_train_resampled, X_val, y_val)
    
    metrics = {}
    metrics['Logistic Regression'] = evaluate_model(logistic, X_test, y_test, "Logistic Regression")
    metrics['Random Forest'] = evaluate_model(rf, X_test, y_test, "Random Forest")
    metrics['XGBoost'] = evaluate_model(xgb, X_test, y_test, "XGBoost")
    
    joblib.dump(logistic, os.path.join(models_dir, 'logistic.pkl'))
    joblib.dump(rf, os.path.join(models_dir, 'rf.pkl'))
    joblib.dump(xgb, os.path.join(models_dir, 'xgb.pkl'))
    joblib.dump(logistic.named_steps['preprocessor'], os.path.join(models_dir, 'preprocessor.pkl'))
    
    for model, name in [(logistic, "Logistic Regression"), (rf, "Random Forest"), (xgb, "XGBoost")]:
        proba = test_high_risk_customer(model, encoders, X_train_columns)
        if proba < 0.6:
            print(f"Warning: {name} predicts {proba * 100:.1f}% for high-risk customer (expected >60%)")
    
    print(f"Models and preprocessor saved to {models_dir}")
    return metrics

if __name__ == "__main__":
    metrics = main()