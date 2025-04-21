# Fault Detection in SECOM Manufacturing Data

## 📌 Overview
This project implements a machine learning pipeline using **XGBoost** to detect faults in a semiconductor manufacturing process. It utilizes the **SECOM dataset**, performing data cleaning, scaling, and classification to identify defective product conditions.

## 🧪 Dataset
- Source: [UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/secom)
- Features: 590+ sensor readings from a manufacturing process
- Target: Fault status (`1` = faulty, `0` = normal)

## ⚙️ Model Pipeline
1. **Data Cleaning**: Drop unnecessary columns, convert types, replace invalid values
2. **Preprocessing**:
   - Imputation (`mean`)
   - Feature Scaling (`StandardScaler`)
3. **Modeling**:
   - Classifier: `XGBClassifier` (n_estimators=300, learning_rate=0.05, max_depth=6)
   - Metric: Accuracy Score
4. **Evaluation**:
   - Training/Test split (80/20, stratified)
   - Accuracy reporting

## 🚀 How to Run

### Requirements
- Python 3.7+
- Install dependencies:
  ```bash
  pip install pandas scikit-learn xgboost
