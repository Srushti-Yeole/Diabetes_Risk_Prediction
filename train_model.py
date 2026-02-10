import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

# Load and preprocess data
def preprocess_data(df):
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Replace zeros with NaN for medical values that can't be zero
    columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_to_process:
        df.loc[df[column] == 0, column] = np.nan
    
    # Fill NaN with median for each column
    for column in columns_to_process:
        df[column] = df[column].fillna(df[column].median())
    
    # Add some feature engineering
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_BMI'] = df['Age'] * df['BMI']
    df['Glucose_Age'] = df['Glucose'] * df['Age']
    
    return df

# Load data
df = pd.read_csv("diabetes.csv")
df = preprocess_data(df)

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameters to search
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 15, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__class_weight': ['balanced']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# Fit the model
print("Training model with cross-validation...")
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Make predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Print metrics
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.named_steps['classifier'].feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

# Save the model and scaler
print("\nSaving model and scaler...")
joblib.dump(best_model, "model.pkl", protocol=4)