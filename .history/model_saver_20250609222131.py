import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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
    # Load your dataset
    try:
        data_path = "cleaned_dataset.csv"
        df = pd.read_csv(data_path)
        print("Loaded dataset successfully.")
    except FileNotFoundError:
        print(f"Dataset not found at {data_path}. Creating sample data for demonstration.")
        # Create sample data for demonstration
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10, 
            n_redundant=5, n_classes=2, random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['Attrition'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for binary encoding of the target variable
    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':
        print("Converting Attrition column from categorical to numeric")
        df['Attrition_Numeric'] = (df['Attrition'] == 'Yes').astype(int)
        target_column = 'Attrition_Numeric'
    elif 'Attrition' in df.columns:
        target_column = 'Attrition'
    else:
        # Use the last column as target if Attrition is not found
        target_column = df.columns[-1]
        print(f"Attrition column not found. Using {target_column} as target.")
    
    # Create features and target
    X = df.drop([target_column], axis=1)
    if 'Attrition' in df.columns and target_column != 'Attrition':
        X = X.drop(['Attrition'], axis=1)
    y = df[target_column]
    
    print(f"Target variable: {target_column}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Features: {X.columns.tolist()}")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Create preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit the preprocessor
    print("Fitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed training data shape: {X_train_processed.shape}")
    
    # Create models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # Dictionary to store models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(n_estimators=100, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Fit each model
    print("Training models...")
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(X_train_processed, y_train)
    
    # Save models and preprocessor
    save_models_and_pipeline(models, preprocessor)
    
    return models, preprocessor

if __name__ == "__main__":
    print("Extracting and saving models from notebooks...")
    models, preprocessor = extract_models_from_notebooks()
    print("Models and preprocessing pipeline have been saved successfully.")
    print("You can now run the dashboard with: streamlit run dashboard.py")
