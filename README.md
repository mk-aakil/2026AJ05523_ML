# Shopper Insights ML

### Online Shopping Purchase Intention Prediction

A machine learning project to predict whether an online shopping session will result in a purchase, using customer browsing and session attributes.

## Problem Statement

Develop and compare five supervised classification algorithms for predicting online purchase intention.

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Online Shoppers Purchasing Intention
- **Instances:** 12,330
- **Input features:** 17
- **Target:** Revenue
- **Task:** Binary classification
- **Dataset URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## Github Repository Link

https://github.com/mk-aakil/2026AJ05523_ML

## Planned Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## Evaluation Metrics

Accuracy, ROC-AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient.

## Models Used and Performance Comparison

Five supervised machine learning classification algorithms were trained and evaluated using the Online Shoppers Purchasing Intention dataset. All models were evaluated on the same test dataset using six classification metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8844 | 0.8556 | 0.6633 | 0.5157 | 0.5803 | 0.5199 |
| KNN | 0.8763 | 0.7942 | 0.6878 | 0.3691 | 0.4804 | 0.4435 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Support Vector Machine | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved the highest AUC (0.8828) and precision (0.7366), but relatively low recall (0.3586), indicating that many actual purchasing sessions were missed. |
| Decision Tree | Achieved the highest F1 Score (0.5803) and MCC (0.5199), providing the strongest overall balance between identifying purchasing sessions and limiting classification errors. |
| KNN | Achieved reasonable accuracy (0.8763), but lower AUC and F1 scores than the strongest models. |
| Naive Bayes | Achieved exceptionally high recall (0.9738), but very low precision (0.1726) and accuracy (0.2729), suggesting a large number of false-positive purchase predictions. |
| Support Vector Machine | Achieved the highest accuracy (0.8873) and strong AUC (0.8824), F1 (0.5629), and MCC (0.5149), making it a close competitor to Decision Tree. |
| **Overall Winner for the Dataset** | **Decision Tree**, based on the highest F1 Score and MCC among the five evaluated classifiers. |

### Conclusion

The experimental results demonstrate that model selection should consider multiple evaluation metrics rather than accuracy alone. The dataset contains considerably more non-purchasing sessions than purchasing sessions, making F1 Score and Matthews Correlation Coefficient particularly relevant.

Although Support Vector Machine achieved the highest accuracy, Decision Tree achieved the best F1 Score and MCC, making it the preferred model for the overall classification task.

Naive Bayes achieved the highest recall, but its low precision indicates a substantial false-positive tradeoff. Logistic Regression demonstrated strong AUC and precision, while KNN provided moderate performance.

**Final Selected Model: Decision Tree Classifier.**


## Technology Stack

Python, pandas, NumPy, scikit-learn, Streamlit, Git, and GitHub.

## Getting Started

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the dataset:

```bash
python src/download_dataset.py
```

Verify the dataset:

```bash
python src/verify_dataset.py
```

## Application Design

The planned Streamlit application will use a customized Business Analytics Pro interface featuring dataset analysis, model comparison, evaluation metrics, and confusion matrices.
