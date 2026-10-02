# Task 2: Credit Card Fraud Detection

**CodSoft Machine Learning Internship**

## Problem
Build a machine learning model that classifies credit card transactions as **fraudulent** or **legitimate**. Fraud is extremely rare (about 0.17% of transactions), so the main challenge is handling the **class imbalance** and evaluating with the right metrics.

## Dataset
[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- About 285,000 transactions made by European cardholders over two days
- Features: `Time`, `V1`-`V28` (PCA-transformed), `Amount`
- Target: `Class` (1 = fraud, 0 = legitimate)

> The CSV is about 150 MB, which is over GitHub's file limit, so it is **not included** in this repo. Download `creditcard.csv` from the link above and place it in this folder before running the notebook.

## Approach
1. **EDA:** missing values, duplicates, class distribution, transaction amounts, feature correlations
2. **Preprocessing:** removed duplicates, scaled `Time` and `Amount` (scaler fitted on the training set only), stratified 80/20 train/test split
3. **Imbalance handling:** compared three strategies
   - No handling (baseline)
   - `class_weight='balanced'`
   - SMOTE oversampling (training set only)
4. **Models:** Logistic Regression, Decision Tree, Random Forest
5. **Evaluation:** precision, recall, F1, ROC-AUC and PR-AUC for the fraud class, plus confusion matrices and precision-recall curves
6. **Extras:** Random Forest feature importance and decision-threshold tuning

## Why not accuracy?
A model that predicts "legitimate" for every transaction gets about 99.8% accuracy and detects no fraud at all. Precision, recall, F1 and PR-AUC on the fraud class give a much more honest picture.

## Results
Fill in this table with the output of section 7 of the notebook after you run it.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| LogReg (baseline) | | | | | |
| LogReg (class weight) | | | | | |
| LogReg (SMOTE) | | | | | |
| Decision Tree (class weight) | | | | | |
| Random Forest (class weight) | | | | | |
| Random Forest (SMOTE) | | | | | |

**Best model:** _(write which one won and why)_

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter
jupyter notebook fraud_detection.ipynb
```
Or upload the notebook and `creditcard.csv` to Google Colab and run all cells (add `!pip install imbalanced-learn` if it is missing).

## Files
- `fraud_detection.ipynb`: full analysis and model training
- `README.md`: this file

## Possible improvements
- Gradient Boosting / XGBoost / LightGBM
- Hyperparameter tuning with cross-validation
- Choosing the decision threshold from the real cost of a missed fraud versus a false alarm
