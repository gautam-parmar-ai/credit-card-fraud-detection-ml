# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using three different classification algorithms.

## Authors

- **Gautam Parmar** - VR545120
- **Muneeb Ahmed Khan** - VR545124

## Problem Statement

Credit card fraud is a significant concern in the financial industry. With millions of transactions occurring daily, identifying fraudulent activities manually is impractical. This project aims to build a machine learning model that can automatically classify transactions as legitimate or fraudulent.

## Dataset
Link: http://kaggle.com/datasets/mlg-ulb/creditcardfraud
Due to the huge amount of data, it was not possible to upload the dataset file in csv, so I have compressed it into a zip file.
The dataset contains 284,807 credit card transactions with the following characteristics:

- **Total Transactions**: 284,807
- **Legitimate Transactions**: 284,315 (99.83%)
- **Fraudulent Transactions**: 492 (0.17%)

The dataset is highly imbalanced, with fraudulent transactions making up less than 0.2% of all transactions.

## Approach

### Data Preprocessing

1. **Missing Value Check**: Verified no null values exist in the dataset
2. **Class Distribution Analysis**: Analyzed the significant imbalance between legitimate and fraudulent transactions
3. **Undersampling Technique**: Created a balanced dataset by sampling 492 legitimate transactions to match the fraudulent count, resulting in a 984-transaction dataset with equal class distribution

### Models Implemented

1. **Logistic Regression**: A linear classification model suitable for binary classification tasks
2. **Random Forest**: An ensemble learning method using 200 decision trees with balanced class weights
3. **Support Vector Machine (SVM)**: Using RBF kernel for non-linear decision boundary

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

### Validation Strategy

- 5-Fold Stratified Cross-Validation
- 80-20 Train-Test Split with stratification

## Results Summary

The models were evaluated on the undersampled dataset. Key findings:

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | High | Good | Good | Good | ~0.98 |
| Random Forest | High | High | High | High | ~0.99 |
| SVM | High | Good | Good | Good | ~0.56 |

Random Forest demonstrated the best overall performance with highest feature importance analysis capability.

## Key Insights

- The highly imbalanced dataset required undersampling to ensure fair model training
- Random Forest provided the most reliable fraud detection capabilities
- Feature importance analysis revealed which transaction characteristics are most predictive of fraud
- False negatives (missed fraud cases) were minimized using proper model selection

## How to Run

1. Ensure Python 3.x is installed with required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Open `project.ipynb` in Jupyter Notebook or Google Colab

3. Run all cells sequentially to reproduce the results

4. The final prediction model can be used to classify new transactions

## Files

- `project.ipynb`: Complete Jupyter notebook with all code and analysis
- `creditcard.csv.zip`: Compressed source dataset
- `README.md`: This file

## Conclusion

This project demonstrates an effective approach to credit card fraud detection using machine learning. By addressing the class imbalance problem and comparing multiple models, we identified Random Forest as the most effective classifier for this task. The model can be integrated into real-world fraud detection systems to automatically flag suspicious transactions.

