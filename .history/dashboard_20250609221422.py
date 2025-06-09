import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction Dashboard",
    page_icon="👥",
    layout="wide"
)

# Define paths to saved models and preprocessing pipeline
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')

def load_models():
    """Load all trained models from the models directory"""
    models = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.pkl') and filename != 'preprocessing_pipeline.pkl':
            model_path = os.path.join(MODELS_DIR, filename)
            model_name = filename.replace('.pkl', '')
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model {model_name}: {e}")
    return models

def load_preprocessing_pipeline():
    """Load the preprocessing pipeline"""
    try:
        return joblib.load(PIPELINE_PATH)
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        return None

def get_model_metrics():
    """Return a dataframe with model metrics"""
    # In a real scenario, you'd load these from saved files or compute them
    # This is sample data - replace with your actual metrics
    metrics = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM'],
        'Accuracy': [0.85, 0.89, 0.91, 0.88],
        'Precision': [0.83, 0.87, 0.90, 0.85],
        'Recall': [0.81, 0.86, 0.88, 0.84],
        'F1-Score': [0.82, 0.87, 0.89, 0.84],
        'ROC AUC': [0.88, 0.92, 0.94, 0.90]
    }
    return pd.DataFrame(metrics)

def predict_attrition(model, features, preprocessing_pipeline):
    """Predict employee attrition using the selected model"""
    try:
        # Convert input features to dataframe
        input_df = pd.DataFrame([features])
        
        # Preprocess the input data
        X_processed = preprocessing_pipeline.transform(input_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_processed)[0]
        prediction = model.predict(X_processed)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def display_sample_data():
    """Display sample dataset structure and correlation heatmap"""
    # Sample data - replace with your actual dataset
    sample_data = pd.DataFrame({
        'Age': [25, 36, 45, 30, 52],
        'Salary': [50000, 65000, 90000, 55000, 75000],
        'YearsAtCompany': [2, 5, 10, 3, 15],
        'JobSatisfaction': [3, 4, 5, 2, 3],
        'Attrition': [1, 0, 0, 1, 0]
    })
    
    st.subheader("Sample Data")
    st.dataframe(sample_data)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sample_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Attrition distribution
    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots(figsize=(8, 8))
    attrition_counts = sample_data['Attrition'].value_counts()
    ax.pie(attrition_counts, labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

def main():
    st.title("Employee Attrition Prediction Dashboard")
    
    # Load models and preprocessing pipeline
    models = load_models()
    preprocessing_pipeline = load_preprocessing_pipeline()
    
    if not models or preprocessing_pipeline is None:
        st.warning("Please ensure that trained models and preprocessing pipeline exist in the 'models' directory.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "Data Insights"])
    
    with tab1:
        st.header("Predict Employee Attrition")
        
        # Create two columns for input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 65, 30)
            salary = st.number_input("Monthly Income", 1000, 20000, 5000)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 10)
            
        with col2:
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
            job_level = st.slider("Job Level", 1, 5, 2)
            overtime = st.selectbox("Overtime", ["Yes", "No"])
        
        # More input fields can be added based on your model requirements
        
        # Create a dictionary of features
        features = {
            'Age': age,
            'MonthlyIncome': salary,
            'YearsAtCompany': years_at_company,
            'DistanceFromHome': distance_from_home,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'JobLevel': job_level,
            'OverTime': 1 if overtime == "Yes" else 0,
            # Add other required features with default values
        }
        
        # Model selection
        model_name = st.selectbox("Select Model", list(models.keys()))
        
        if st.button("Predict"):
            prediction, prediction_proba = predict_attrition(models[model_name], features, preprocessing_pipeline)
            
            if prediction is not None:
                st.subheader("Prediction Result")
                
                # Display prediction with probability
                if prediction == 1:
                    st.error(f"Employee is likely to leave (Probability: {prediction_proba[1]:.2f})")
                else:
                    st.success(f"Employee is likely to stay (Probability: {prediction_proba[0]:.2f})")
                
                # Display probability gauge
                prob_to_display = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                st.progress(float(prob_to_display))
    
    with tab2:
        st.header("Model Performance Comparison")
        
        # Display model metrics
        metrics_df = get_model_metrics()
        st.dataframe(metrics_df)
        
        # Display bar chart of metrics
        st.subheader("Metrics Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_plot = metrics_df.drop('Model', axis=1)
        metrics_to_plot.index = metrics_df['Model']
        metrics_to_plot.plot(kind='bar', ax=ax)
        
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with tab3:
        st.header("Data Insights")
        display_sample_data()

if __name__ == "__main__":
    main()
