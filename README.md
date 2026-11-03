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

## Planned Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## Evaluation Metrics

Accuracy, ROC-AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient.

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

## Development Status

Phase 1: Project initialization and dataset verification.

## Dataset Attribution

Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. DOI: 10.24432/C5F88Q.

Dataset license: CC BY 4.0.