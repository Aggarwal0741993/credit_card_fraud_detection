
# Credit Card Fraud Detection using XGBoost and Optuna

This project builds an intelligent system to detect fraudulent credit card transactions using **XGBoost**, with advanced **hyperparameter tuning using Optuna** and proper **handling of class imbalance**.

---

## **Problem Statement**

In real-world credit card transaction data, fraudulent cases are extremely rare — often less than **0.2%** of total transactions. The objective is to **identify these rare frauds** while minimizing false alarms using precision-tuned machine learning techniques.

---

## **Dataset**

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Features**: 30 (V1–V28 from PCA + Time + Amount)
- **Target**: `Class` (0 = Legit, 1 = Fraud)

---

## **Project Highlights**

- Used **XGBoost**, **Logistic Regression**, and **Random Forest**
- Applied **Optuna** to perform **automated hyperparameter tuning**
- Optimized using **5-fold cross-validation** with recall as the scoring metric
- Final model selected: **XGBoost** with tuned parameters

---

## **Best Model Configuration (XGBoost)**

```python
XGBClassifier(
  n_estimators=76,
  learning_rate=0.163,
  max_depth=8,
  scale_pos_weight=17,
  use_label_encoder=False,
  eval_metric='logloss'
)
```

---

## **Final Test Set Performance**

| Metric      | Value  |
|-------------|--------|
| **Precision** (Fraud) | 0.94   |
| **Recall** (Fraud)    | 0.85   |
| **F1-Score** (Fraud)  | 0.89   |
| **ROC AUC Score**     | 0.93   |
| **Accuracy**          | ~100%  |

> The model achieves a **great balance between catching frauds and avoiding false alarms**, making it suitable for real-world deployment.

---

## **Key Techniques Used**

- **StandardScaler** for `Time` and `Amount` columns
- **Stratified train-test split**
- **XGBoost with scale_pos_weight for imbalance handling**
- **Optuna** for hyperparameter optimization with cross-validation
- **Evaluation** using precision, recall, F1-score, and ROC AUC

---

## **Next Steps (Future Enhancements)**

- Deploy as a **Streamlit app** for interactive fraud prediction
- Add **SHAP-based model explainability**
- Integrate with **real-time alert systems**
- Compare with neural networks or ensemble models

---

## **How to Run**

1. Clone the repo
2. Upload or download `creditcard.csv` dataset
3. Open `credit_card_fraud.ipynb` in Colab or Jupyter
4. Run all cells — training, tuning, and final evaluation are included

---

## **Author**

**Shubham Gautam**  
_Data Science & ML Enthusiast_  
[LinkedIn Profile](#) *(optional)*
