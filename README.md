# Customer Churn & Retention Analytics System

Predicts whether a telecom customer will leave (churn) using a Deep Neural Network.

**Tech:** Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib

## Dataset
IBM Telco Customer Churn: 7,043 customers, 19 features (tenure, contract, services, payment method, charges).
Target: `Churn` (Yes/No). 26.5% of customers churned, so the classes are imbalanced.

## What I did
1. **EDA:** plotted churn against contract, tenure, monthly charges, payment method and services, plus a correlation heatmap.
2. **Preprocessing:** converted `TotalCharges` from text to numeric (11 blank values), then used a Scikit-learn Pipeline for missing values (median), one-hot encoding and feature scaling. It is fitted on training data only to avoid data leakage.
3. **Class imbalance:** balanced class weights (churners weighted about 2.8× more).
4. **Model:** Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dropout(0.3) → Dense(1, Sigmoid), trained with Adam and binary cross-entropy, with early stopping on validation loss.
5. **Evaluation:** accuracy, precision, recall, F1, confusion matrix and ROC-AUC, plus a Logistic Regression baseline for comparison.

## Key findings
- Month-to-month contracts churn at 42.7% vs 2.8% for two-year contracts.
- New customers churn most: 47.4% churn in the first 12 months.
- Electronic-check users (45.3%) and fiber-optic users (41.9%) churn the most.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| DNN | 0.740 | 0.507 | 0.799 | 0.620 | 0.845 |
| Logistic Regression | 0.741 | 0.508 | 0.789 | 0.618 | 0.842 |

The model catches about 80% of customers who churn. Accuracy alone is misleading here, because always predicting "no churn" already scores 73.5%.

## How to run
```bash
pip install -r requirements.txt
python -m src.eda       # EDA plots → reports/figures/
python -m src.train     # train + evaluate, saves model to models/
python -m src.predict   # example: "Customer has 93% probability of churn."
```

## Project structure
```
data/        dataset
notebooks/   EDA notebook
src/         data loading, EDA, preprocessing, model, training, evaluation, prediction
models/      saved model + preprocessor
reports/     plots
```

## Future improvements
- Tune the decision threshold based on business cost
- Try Random Forest / Gradient Boosting
- Add explainability (SHAP)
