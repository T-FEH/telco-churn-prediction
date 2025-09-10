# src/data_prep.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    """Load the Telco Customer Churn dataset."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Handle missing values and data types."""
    # Convert TotalCharges to numeric, replace errors with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Replace NaN in TotalCharges with 0 (new customers with no charges yet)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    # Convert SeniorCitizen to categorical
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')
    return df

def engineer_features(df):
    """Create business-driven features."""
    # tenure_bucket: 0–6m, 6–12m, 12–24m, 24m+
    bins = [0, 6, 12, 24, float('inf')]
    labels = ['0-6m', '6-12m', '12-24m', '24m+']
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    
    # services_count: total number of services
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['services_count'] = df[service_cols].apply(
        lambda x: sum(1 for col in x if col in ['Yes', 'DSL', 'Fiber optic']), axis=1
    )
    
    # monthly_to_total_ratio: TotalCharges / max(1, tenure * MonthlyCharges)
    df['monthly_to_total_ratio'] = df['TotalCharges'] / np.maximum(1, df['tenure'] * df['MonthlyCharges'])
    
    # Flag: Internet but no tech support
    df['internet_no_tech_support'] = ((df['InternetService'].isin(['DSL', 'Fiber optic'])) & 
                                     (df['TechSupport'] == 'No')).astype(int)
    
    return df

def encode_categorical(df):
    """Encode categorical variables using LabelEncoder."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.drop('customerID')  # Drop customerID before encoding
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def compute_clv(df):
    """Compute CLV based on Expected Tenure."""
    # Assumption: Expected Tenure = tenure + 12 months if no contract, else tenure + 24 months
    df['ExpectedTenure'] = df.apply(
        lambda x: x['tenure'] + 12 if x['Contract'] == 'Month-to-month' else x['tenure'] + 24, axis=1
    )
    df['CLV'] = df['MonthlyCharges'] * df['ExpectedTenure']
    return df

def split_data(df, target='Churn', train_size=0.6, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets with stratification."""
    # Separate features and target, drop customerID
    X = df.drop(columns=[target, 'customerID'])
    y = df[target]
    
    # Train + (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    
    # Split temp into val and test
    val_size_adjusted = val_size / (1 - train_size)  # Adjust for remaining data
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    """Save processed data splits."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

def main():
    # File paths
    raw_data_path = 'data/raw/Telco-Customer-Churn.csv'
    processed_data_path = 'data/processed/'
    
    # Load data
    df = load_data(raw_data_path)
    
    # Clean data
    df = clean_data(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Compute CLV
    df = compute_clv(df)
    
    # Encode categorical variables
    df, encoders = encode_categorical(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Save splits
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, processed_data_path)
    
    # Save encoders (for app predictions)
    import joblib
    joblib.dump(encoders, os.path.join(processed_data_path, 'encoders.pkl'))
    
    print("Data preparation complete. Splits and encoders saved to data/processed/")

if __name__ == "__main__":
    main()