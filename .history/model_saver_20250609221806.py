import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def save_models_and_pipeline(models_dict, preprocessor, output_dir='models'):
    """
    Save trained models and preprocessing pipeline to the specified directory.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary with model names as keys and trained model objects as values
    preprocessor : object
        Trained preprocessing pipeline or transformer
    output_dir : str
        Directory where models will be saved
    """
    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessing pipeline
    pipeline_path = os.path.join(output_dir, 'preprocessing_pipeline.pkl')
    joblib.dump(preprocessor, pipeline_path)
    print(f"Saved preprocessing pipeline to {pipeline_path}")
    
    # Save each model
    for model_name, model in models_dict.items():
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model to {model_path}")

def extract_models_from_notebooks():
    """
    Extract and save trained models from the notebook environments.
    This is a placeholder function - in practice, you would run this
    after training your models in the notebooks.
    """
    # Load your dataset (replace with actual path)
    try:
        df = pd.read_csv("cleaned_dataset.csv")
        print("Loaded dataset successfully.")
    except FileNotFoundError:
        print("Dataset not found. Using sample data for demonstration.")
        # Create sample data for demonstration
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10, 
            n_redundant=5, n_classes=2, random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['Attrition'] = y
    
    # Create preprocessing pipeline
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target variable from features if present
    if 'Attrition' in numerical_cols:
        numerical_cols.remove('Attrition')
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', Pipeline([
                ('encoder', LabelEncoder())
            ]), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Create dummy models for demonstration
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # Dictionary to store models
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'xgboost': XGBClassifier(random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    
    # In a real scenario, you would load your actual trained models here
    # For demonstration, we'll just create some dummy models
    print("Creating dummy models for demonstration...")
    
    # Create a simple feature matrix and target vector
    if 'Attrition' in df.columns:
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Fit the preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Fit each model
    for name, model in models.items():
        model.fit(X_processed, y)
        print(f"Fitted {name} model.")
    
    # Save models and preprocessor
    save_models_and_pipeline(models, preprocessor)
    
    return models, preprocessor

if __name__ == "__main__":
    print("Extracting and saving models from notebooks...")
    models, preprocessor = extract_models_from_notebooks()
    print("Models and preprocessing pipeline have been saved successfully.")
    print("You can now run the dashboard with: streamlit run dashboard.py")
