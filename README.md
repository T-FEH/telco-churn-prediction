# Telco Customer Churn Prediction

## 📌 Overview
This project develops a machine learning model to predict customer churn for a telecommunications company using the **Telco Customer Churn dataset**.  

It includes:
- **Data preprocessing**
- **Feature engineering**
- **Model training** (Logistic Regression, Random Forest, XGBoost)
- **Interpretability analysis** (SHAP, feature importance)
- **Streamlit app** for interactive predictions and visualizations

---

## 📂 Project Structure
```

├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       ├── X\_train.csv
│       ├── X\_test.csv
│       ├── y\_test.csv
│       ├── encoders.pkl
│       ├── plots/
│       │   ├── clv\_distribution.png
│       │   ├── churn\_by\_quartile.png
│       │   ├── logistic\_importance.png
│       │   ├── random\_forest\_shap\_summary.png
│       │   ├── xgboost\_shap\_summary.png
│       ├── logistic\_importance.csv
│       └── churn\_rates.csv
├── models/
│   ├── logistic.pkl
│   ├── rf.pkl
│   ├── xgb.pkl
│   └── preprocessor.pkl
├── src/
│   ├── data\_prep.py
│   ├── clv\_analysis.py
│   ├── train\_models.py
│   └── interpretability.py
├── app.py
├── requirements.txt
├── .gitattributes
├── .gitignore
└── AI\_USAGE.md

````

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/telco-churn-prediction.git
cd telco-churn-prediction
````

### 2. Install Git LFS (for `.pkl` files)

```bash
sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run scripts in order

```bash
python3 src/data_prep.py
python3 src/clv_analysis.py
python3 src/train_models.py
python3 src/interpretability.py
```

### 6. Run the Streamlit app locally

```bash
streamlit run app.py
```

---

## 🚀 Usage

* **Data Preparation (`data_prep.py`)**

  * Cleans dataset
  * Engineers features (`tenure_bucket`, `services_count`, `internet_no_tech_support`, `CLV`)
  * Saves splits and encoders

* **CLV Analysis (`clv_analysis.py`)**

  * Generates CLV distribution and churn rate plots
  * Saves plots in `data/processed/plots/`

* **Model Training (`train_models.py`)**

  * Trains Logistic Regression, Random Forest, and XGBoost
  * Evaluates Precision, Recall, F1, AUC-ROC
  * Tests a high-risk customer scenario
  * Saves trained models

* **Interpretability (`interpretability.py`)**

  * Generates feature importance (Logistic Regression)
  * Generates SHAP plots (Random Forest, XGBoost)

* **Streamlit App (`app.py`)**

  * **Predict Tab**: Input customer details → churn probability, CLV, SHAP explanation (XGBoost)
  * **Model Performance Tab**: Metrics, ROC curves, XGBoost confusion matrix, feature importance
  * **CLV Overview Tab**: CLV distribution, churn rates by quartile, retention-focused takeaway


## 🌍 Deployment

* **Streamlit App (Community Cloud):** \[Insert app URL here]
* **Demo Video:** \[Insert video URL after recording]

---