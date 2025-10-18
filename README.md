# Loan-Prediction-project
A Machine Learning project to predict loan approvals using Python.
Cell 1 — Install required libraries (run once): pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm streamlit joblib

Cell 2 — Imports: pandas, numpy, matplotlib.pyplot, seaborn, sklearn model_selection / preprocessing / metrics, RandomForestClassifier, LogisticRegression, xgboost, joblib

Cell 3 — Load dataset: read CSV from your project path in chunks to limit memory (limit to ~100k rows), concat chunks, show shape and head

Cell 4 — Data overview: df.info(), df.describe(), show head, display target distribution and default rate

Cell 5 — Exploratory plots: countplot for TARGET, histograms for numeric features, correlation heatmap, example categorical comparisons (e.g., gender vs TARGET)

Cell 6 — Missing data handling: identify top missing columns, drop columns with >40% missing, fill numeric columns with median and categorical with mode, confirm no remaining missing

Cell 7 — Encode categoricals: use LabelEncoder or category codes for object columns; verify converted types
Cell 8 — Scaling: fit StandardScaler on numeric features (exclude TARGET), transform numeric columns

Cell 9 — Train/test split: stratified split (test_size=0.3, random_state=42), print shapes

Cell 10 — Model training:

Train a baseline LogisticRegression
Train a RandomForestClassifier (reduced complexity if memory constrained)
Optionally train XGBoost with modest settings
Fit on training set
Cell 11 — Evaluation:

Generate classification_report and confusion_matrix for each model
Compute ROC-AUC using predicted probabilities
Plot confusion matrix, ROC curve, and top feature importances (from RF)
Cell 12 — Save artifacts for Streamlit:

Save chosen model with joblib (e.g., improved_loan_model.pkl)
Save fitted scaler (improved_loan_scaler.pkl)
Save ordered feature list used by the app (improved_loan_features.pkl)
Cell 13 — Notes & run instructions:

Remind to ensure TARGET exists and feature ordering matches the Streamlit app
To run the app from the project folder use: python -m streamlit run loan_app.py
If model files are large, recommend Git LFS or exclude from repo
