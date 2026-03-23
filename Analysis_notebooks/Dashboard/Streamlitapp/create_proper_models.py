"""
Create proper model files with fitted scalers for the Ethiopian Student Performance Dashboard
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("Creating Proper Model Files with Fitted Scalers")
print("=" * 60)

# Define feature names (based on your data processing)
feature_names = [
    'Gender', 'Home_Internet_Access', 'Electricity_Access', 'School_Location',
    'Field_Choice', 'Parental_Involvement', 'Teacher_Student_Ratio',
    'School_Resources_Score', 'School_Academic_Score', 'Student_to_Resources_Ratio',
    'Overall_Textbook_Access_Composite', 'Overall_Avg_Attendance', 'Overall_Avg_Homework',
    'Overall_Avg_Participation', 'Age', 'Father_Education_Encoded', 'Mother_Education_Encoded',
    'Region_Encoded', 'School_Type_Freq', 'School_Type_Target', 'Career_Interest_Encoded',
    'Health_Issue_Flag', 'Health_Issue_Severity', 'Health_Issue_Target', 'Overall_Engagement_Score'
]

print(f"\n✓ Defined {len(feature_names)} features")

# Create realistic training data
np.random.seed(42)
n_samples = 5000

# Generate realistic synthetic data
X_train = pd.DataFrame(index=range(n_samples))

# Generate features with realistic distributions
X_train['Gender'] = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
X_train['Home_Internet_Access'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
X_train['Electricity_Access'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
X_train['School_Location'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 0=Rural, 1=Urban
X_train['Field_Choice'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 0=Social, 1=Natural
X_train['Parental_Involvement'] = np.random.uniform(0, 1, n_samples)
X_train['Teacher_Student_Ratio'] = np.random.uniform(30, 60, n_samples)
X_train['School_Resources_Score'] = np.random.uniform(0.3, 0.9, n_samples)
X_train['School_Academic_Score'] = np.random.uniform(0.3, 0.9, n_samples)
X_train['Student_to_Resources_Ratio'] = np.random.uniform(15, 30, n_samples)
X_train['Overall_Textbook_Access_Composite'] = np.random.uniform(0.2, 0.9, n_samples)
X_train['Overall_Avg_Attendance'] = np.random.uniform(60, 100, n_samples)
X_train['Overall_Avg_Homework'] = np.random.uniform(40, 100, n_samples)
X_train['Overall_Avg_Participation'] = np.random.uniform(40, 100, n_samples)
X_train['Age'] = np.random.uniform(14, 20, n_samples)
X_train['Father_Education_Encoded'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])
X_train['Mother_Education_Encoded'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])
X_train['Region_Encoded'] = np.random.uniform(0, 1, n_samples)
X_train['School_Type_Freq'] = np.random.uniform(0, 1, n_samples)
X_train['School_Type_Target'] = np.random.uniform(0, 1, n_samples)
X_train['Career_Interest_Encoded'] = np.random.uniform(0, 1, n_samples)
X_train['Health_Issue_Flag'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
X_train['Health_Issue_Severity'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05])
X_train['Health_Issue_Target'] = np.random.uniform(0, 1, n_samples)
X_train['Overall_Engagement_Score'] = (
    X_train['Overall_Avg_Attendance'] * 0.4 + 
    X_train['Overall_Avg_Homework'] * 0.3 + 
    X_train['Overall_Avg_Participation'] * 0.3
) / 100  # Scale to 0-1

# Generate target variables
# Overall Average Score (target for regression) - based on features with realistic relationships
y_train_reg = (
    20 +  # Base score
    15 * X_train['School_Resources_Score'] +  # Strong positive influence
    10 * X_train['Overall_Textbook_Access_Composite'] +  # Textbook access influence
    5 * X_train['Parental_Involvement'] +  # Parental involvement
    -0.2 * (X_train['Teacher_Student_Ratio'] - 40) +  # Negative influence of high teacher-student ratio
    0.1 * X_train['Overall_Avg_Attendance'] +  # Attendance influence
    0.1 * X_train['Overall_Avg_Homework'] +  # Homework influence
    0.05 * X_train['Overall_Avg_Participation'] +  # Participation influence
    np.random.normal(0, 5, n_samples)  # Add some noise
)

# Clip to realistic range (0-100)
y_train_reg = np.clip(y_train_reg, 30, 95)

# Risk classification target (based on overall average)
y_train_class = (y_train_reg < 50).astype(int)

print(f"\n✓ Generated {n_samples} training samples")
print(f"  Regression target range: {y_train_reg.min():.1f} - {y_train_reg.max():.1f}")
print(f"  Classification target: {y_train_class.sum()} positive samples ({(y_train_class.sum()/n_samples*100):.1f}%)")

# Create and train regression model
print("\n" + "=" * 60)
print("Training Regression Model (Gradient Boosting)")
print("=" * 60)

reg_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42,
    verbose=0
)

# Create and fit scaler for regression
reg_scaler = StandardScaler()
X_train_scaled = reg_scaler.fit_transform(X_train[feature_names])

# Train the model
reg_model.fit(X_train_scaled, y_train_reg)

# Evaluate regression model
y_pred_reg = reg_model.predict(X_train_scaled)
r2_score = 1 - np.sum((y_train_reg - y_pred_reg) ** 2) / np.sum((y_train_reg - y_train_reg.mean()) ** 2)
print(f"✓ Regression model trained")
print(f"  R² Score: {r2_score:.4f}")
print(f"  Feature importance range: {reg_model.feature_importances_.min():.4f} - {reg_model.feature_importances_.max():.4f}")

# Save regression model
reg_data = {
    'model': reg_model,
    'scaler': reg_scaler,
    'feature_names': feature_names
}

reg_path = "models/gradient_boosting_regression.pkl"
joblib.dump(reg_data, reg_path)
print(f"✓ Regression model saved to: {reg_path}")

# Create and train classification model
print("\n" + "=" * 60)
print("Training Classification Model (Gradient Boosting)")
print("=" * 60)

class_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42,
    verbose=0
)

# Create and fit scaler for classification
class_scaler = StandardScaler()
X_train_class_scaled = class_scaler.fit_transform(X_train[feature_names])

# Train the model
class_model.fit(X_train_class_scaled, y_train_class)

# Evaluate classification model
y_pred_class = class_model.predict(X_train_class_scaled)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_train_class, y_pred_class)
f1 = f1_score(y_train_class, y_pred_class)
try:
    roc_auc = roc_auc_score(y_train_class, class_model.predict_proba(X_train_class_scaled)[:, 1])
except:
    roc_auc = 0.5

print(f"✓ Classification model trained")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")

# Save classification model
class_data = {
    'model': class_model,
    'scaler': class_scaler,
    'feature_names': feature_names
}

class_path = "models/classification_model.pkl"
joblib.dump(class_data, class_path)
print(f"✓ Classification model saved to: {class_path}")

# Verify the models can be loaded
print("\n" + "=" * 60)
print("Verifying Models")
print("=" * 60)

# Test loading
test_reg = joblib.load(reg_path)
test_class = joblib.load(class_path)

print(f"✓ Regression model loaded successfully")
print(f"  - Has model: {test_reg.get('model') is not None}")
print(f"  - Has scaler: {test_reg.get('scaler') is not None}")
print(f"  - Has features: {len(test_reg.get('feature_names', []))}")

print(f"✓ Classification model loaded successfully")
print(f"  - Has model: {test_class.get('model') is not None}")
print(f"  - Has scaler: {test_class.get('scaler') is not None}")
print(f"  - Has features: {len(test_class.get('feature_names', []))}")

# Test prediction with a sample
sample = X_train.iloc[[0]][feature_names]
sample_scaled = test_reg['scaler'].transform(sample)
sample_pred = test_reg['model'].predict(sample_scaled)
print(f"\n✓ Test prediction successful: {sample_pred[0]:.2f}")

print("\n" + "=" * 60)
print("✅ Model creation complete!")
print("You can now run the Streamlit app: streamlit run app.py")
print("=" * 60)