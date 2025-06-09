import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC

def create_dummy_models():
    """
    Create and save dummy models and preprocessing pipeline 
    for testing the dashboard when actual models are not available.
    """
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a simple preprocessing pipeline
    numeric_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
                       'JobSatisfaction', 'WorkLifeBalance', 'JobLevel']
    categorical_features = ['OverTime']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Save the preprocessing pipeline
    pipeline_path = os.path.join(models_dir, 'preprocessing_pipeline.pkl')
    joblib.dump(preprocessor, pipeline_path)
    print(f"Saved preprocessing pipeline to {pipeline_path}")
    
    # Create and save dummy models
    X = np.random.rand(100, len(numeric_features) + len(categorical_features))
    y = np.random.randint(0, 2, 100)
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X, y)
    lr_path = os.path.join(models_dir, 'logistic_regression.pkl')
    joblib.dump(lr_model, lr_path)
    print(f"Saved logistic regression model to {lr_path}")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X, y)
    rf_path = os.path.join(models_dir, 'random_forest.pkl')
    joblib.dump(rf_model, rf_path)
    print(f"Saved random forest model to {rf_path}")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
    xgb_model.fit(X, y)
    xgb_path = os.path.join(models_dir, 'xgboost.pkl')
    joblib.dump(xgb_model, xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")
    
    # SVM
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X, y)
    svm_path = os.path.join(models_dir, 'svm.pkl')
    joblib.dump(svm_model, svm_path)
    print(f"Saved SVM model to {svm_path}")

if __name__ == "__main__":
    create_dummy_models()
    print("Created dummy models for dashboard testing.")
    print("Run the dashboard with: streamlit run dashboard.py")
