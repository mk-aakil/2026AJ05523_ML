# Shopper Insights ML — Business Analytics Pro

## Integrate into your existing repository

1. Copy `app.py`, `.streamlit/config.toml` and the dependency lines in `requirements.txt` into the **root** of `shopper-insights-ml`. Do not replace existing dependencies needed by your notebooks; merge them.
2. Run `notebooks/02_model_development.ipynb` fully. It creates `model/metadata.json`, five fitted `.joblib` pipelines, `test_data.csv`, and `reports/model_comparison.csv`.
3. Confirm these files exist:

```
shopper-insights-ml/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
├── model/
│   ├── metadata.json
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
└── test_data.csv
```

4. From the project root with `.venv` activated:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

5. Upload `test_data.csv`, select each classifier, and click **Evaluate all five pretrained models**. Check that the scores match your training notebook on the same held-out test set (allowing for rounding).

## Git checkpoint

```bash
git status
git add app.py .streamlit/config.toml requirements.txt README.md
# If your saved model files are not already tracked, add them after checking their sizes:
git add model/metadata.json model/*.joblib
# Add the test file if it is required by your assignment and permitted by your data license:
git add test_data.csv
git diff --cached --stat
git commit -m "feat: build custom Streamlit model evaluation dashboard"
git push origin main
```

If a path is ignored by `.gitignore`, check the reason before adding it. Do not commit secrets or very large model files. GitHub has a 100 MB hard limit per regular Git file. Streamlit Cloud needs access to the trained model files at deployment time.

## Deploy

1. Push the app and its required artifacts to a GitHub repository.
2. Go to https://share.streamlit.io and create a new app from your repository.
3. Set the main file path to `app.py` and deploy.
4. Upload the same held-out `test_data.csv` to confirm cloud inference works.

## Important implementation notes

- Uploaded test CSVs must contain all 17 original predictors and `Revenue`.
- The app loads saved pipelines with `joblib.load`; it does **not** call `.fit()`.
- Uploaded CSV labels are used only for evaluation, not training.
- Test labels containing only one class cannot produce ROC-AUC.
- Never load untrusted `.joblib` files; pickle-based formats can execute code.
- If you encounter a version mismatch when loading models, pin the exact scikit-learn and numpy versions used to train the artifacts, then redeploy.
- Avoid repeatedly evaluating the same held-out test set to tune models; this is for comparison and reporting, not iterative test-set optimization.
