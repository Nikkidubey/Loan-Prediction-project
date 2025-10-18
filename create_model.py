import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic loan data that matches the app's expected features
n_samples = 10000

# Generate synthetic data
data = {
    'income': np.random.normal(50000, 20000, n_samples),
    'loan_amount': np.random.normal(30000, 15000, n_samples),
    'credit_score': np.random.beta(2, 2, n_samples),  # 0-1 range
    'employment_years': np.random.poisson(5, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'debt_to_income': np.random.beta(2, 5, n_samples),  # 0-1 range
    'loan_to_income': np.random.beta(2, 8, n_samples)  # 0-1 range
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure positive values for income and loan_amount
df['income'] = np.abs(df['income'])
df['loan_amount'] = np.abs(df['loan_amount'])

# Create target variable based on realistic loan default criteria
# Higher risk if: low credit score, high debt-to-income, high loan-to-income, young age, short employment
default_prob = (
    (1 - df['credit_score']) * 0.3 +  # Low credit score increases risk
    df['debt_to_income'] * 0.3 +      # High debt-to-income increases risk
    df['loan_to_income'] * 0.2 +      # High loan-to-income increases risk
    (1 - df['employment_years'] / 20) * 0.1 +  # Short employment increases risk
    (1 - df['age'] / 70) * 0.1        # Young age increases risk
)

# Add some noise
default_prob += np.random.normal(0, 0.1, n_samples)
default_prob = np.clip(default_prob, 0, 1)

# Create binary target (1 = default, 0 = no default)
df['target'] = (default_prob > 0.3).astype(int)

# Check class distribution
print(f"Default rate: {df['target'].mean():.2%}")

# Define feature columns (matching the app)
feature_cols = ['income', 'loan_amount', 'credit_score', 'employment_years', 'age', 'debt_to_income', 'loan_to_income']

# Prepare features and target
X = df[feature_cols]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

# Save the model and scaler
joblib.dump(model, "improved_loan_model.pkl")
joblib.dump(scaler, "improved_loan_scaler.pkl")
joblib.dump(feature_cols, "improved_loan_features.pkl")

print("\n✅ Model files saved successfully!")
print("Files created:")
print("- improved_loan_model.pkl")
print("- improved_loan_scaler.pkl") 
print("- improved_loan_features.pkl")

