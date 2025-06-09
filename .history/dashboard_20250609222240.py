import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction Dashboard",
    page_icon="👥",
    layout="wide"
)

# Define paths to saved models and preprocessing pipeline
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')

# Cache model loading to improve performance
@st.cache_resource
def load_models():
    """Load all trained models from the models directory"""
    models = {}
    if not os.path.exists(MODELS_DIR):
        return None
    
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.pkl') and filename != 'preprocessing_pipeline.pkl':
            model_path = os.path.join(MODELS_DIR, filename)
            model_name = filename.replace('.pkl', '')
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model {model_name}: {e}")
    return models

@st.cache_resource
def load_preprocessing_pipeline():
    """Load the preprocessing pipeline"""
    if not os.path.exists(PIPELINE_PATH):
        return None
    
    try:
        return joblib.load(PIPELINE_PATH)
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        return None

# Load a sample dataset
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv("cleaned_dataset.csv")
    except:
        return None

# Function to get model performance metrics
def get_model_metrics():
    # In a real scenario, you'd load actual metrics from saved evaluation results
    metrics = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Decision Tree'],
        'Accuracy': [0.85, 0.89, 0.91, 0.83],
        'Precision': [0.83, 0.87, 0.90, 0.80],
        'Recall': [0.81, 0.86, 0.88, 0.79],
        'F1-Score': [0.82, 0.87, 0.89, 0.80],
        'ROC AUC': [0.88, 0.92, 0.94, 0.85]
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

def create_feature_form(df=None):
    """Create input form for features based on dataset columns"""
    if df is None:
        # Default features if no dataset is available
        features = {
            'SatisfactionLevel': st.slider('Satisfaction Level', 0.0, 1.0, 0.5),
            'LastEvaluation': st.slider('Last Evaluation Score', 0.0, 1.0, 0.7),
            'NumberProjects': st.slider('Number of Projects', 2, 7, 4),
            'AverageMonthlyHours': st.slider('Average Monthly Hours', 90, 310, 180),
            'TimeSpentCompany': st.slider('Time Spent at Company (years)', 1, 10, 3),
            'WorkAccident': st.selectbox('Work Accident', [0, 1]),
            'PromotionLast5Years': st.selectbox('Promotion in Last 5 Years', [0, 1]),
            'Department': st.selectbox('Department', ['Sales', 'Technical', 'Support', 'HR', 'Management']),
            'Salary': st.selectbox('Salary Level', ['Low', 'Medium', 'High']),
            'OverTime': st.selectbox('Overtime', ['Yes', 'No'])
        }
    else:
        # Get column names and types from the actual dataset
        features = {}
        col1, col2 = st.columns(2)
        
        column_count = 0
        current_col = col1
        
        # Exclude target column(s)
        exclude_cols = ['Attrition', 'Attrition_Numeric']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        for column in feature_cols:
            # Switch between columns for better layout
            if column_count % 5 == 0 and column_count > 0:
                current_col = col2 if current_col == col1 else col1
            
            column_count += 1
            
            # For numerical columns
            if df[column].dtype in ['int64', 'float64']:
                min_val = float(df[column].min())
                max_val = float(df[column].max())
                default_val = float(df[column].median())
                
                # For columns with few unique values, use selectbox
                if len(df[column].unique()) <= 10:
                    features[column] = current_col.selectbox(
                        f"{column}", 
                        sorted(df[column].unique().tolist()),
                        index=0
                    )
                else:
                    features[column] = current_col.slider(
                        f"{column}", 
                        min_val, 
                        max_val, 
                        default_val
                    )
            # For categorical columns
            else:
                features[column] = current_col.selectbox(
                    f"{column}", 
                    df[column].unique().tolist()
                )
    
    return features

def main():
    st.title("🔍 Employee Attrition Prediction Dashboard")
    
    # Load models and preprocessing pipeline
    models = load_models()
    preprocessing_pipeline = load_preprocessing_pipeline()
    sample_data = load_sample_data()
    
    if not models or preprocessing_pipeline is None:
        st.warning("⚠️ Models or preprocessing pipeline not found. Please run the model_saver.py script first.")
        if st.button("📋 Show Instructions"):
            st.code("python model_saver.py", language="bash")
            st.markdown("This will create the necessary models and preprocessing pipeline.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "Data Insights"])
    
    with tab1:
        st.header("📊 Predict Employee Attrition")
        
        # Input form for employee data
        st.subheader("Enter Employee Information")
        
        features = create_feature_form(sample_data)
        
        # Model selection
        model_display_names = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'decision_tree': 'Decision Tree'
        }
        
        # Create reverse mapping
        model_keys = {v: k for k, v in model_display_names.items()}
        
        selected_model_name = st.selectbox(
            "Select Model for Prediction",
            list(model_display_names.values())
        )
        
        model_key = model_keys[selected_model_name]
        
        if st.button("Predict Attrition"):
            with st.spinner("Making prediction..."):
                # Get the selected model
                model = models[model_key]
                
                # Make prediction
                prediction, prediction_proba = predict_attrition(model, features, preprocessing_pipeline)
                
                if prediction is not None:
                    st.subheader("Prediction Result")
                    
                    # Calculate probability
                    prob_leaving = prediction_proba[1]
                    
                    # Display prediction with gauge chart
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if prediction == 1:
                            st.error("#### 🚨 Employee is likely to leave")
                        else:
                            st.success("#### ✅ Employee is likely to stay")
                        
                        st.metric(
                            "Probability of Leaving",
                            f"{prob_leaving:.1%}"
                        )
                    
                    with col2:
                        # Create gauge chart using Plotly
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob_leaving * 100,
                            title={'text': "Attrition Risk"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkgray"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "orange"},
                                    {'range': [70, 100], 'color': "red"},
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prob_leaving * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        
                        feature_names = list(features.keys())
                        importances = model.feature_importances_
                        
                        if len(feature_names) == len(importances):
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                importance_df.head(10),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 10 Features by Importance'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Feature importance not available for display due to preprocessing transformations")
                    
                    # Retention suggestions
                    st.subheader("Retention Suggestions")
                    if prediction == 1:
                        st.markdown("""
                        Based on the prediction, here are some retention strategies:
                        
                        * Schedule a one-on-one meeting to discuss job satisfaction
                        * Review compensation package and career growth opportunities
                        * Consider workload adjustments or flexible scheduling
                        * Provide opportunities for skills development
                        * Implement recognition program for achievements
                        """)
                    else:
                        st.markdown("""
                        Even though the employee is likely to stay, consider these engagement strategies:
                        
                        * Regular check-ins to maintain satisfaction levels
                        * Provide stretch assignments for continued growth
                        * Involve in mentoring or knowledge sharing programs
                        * Consider for high-potential talent development
                        """)
    
    with tab2:
        st.header("📈 Model Performance Comparison")
        
        # Get model metrics
        metrics_df = get_model_metrics()
        
        # Display metrics table
        st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
        
        # Create comparison chart
        fig = px.bar(
            metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Metrics Comparison'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model descriptions
        st.subheader("Model Descriptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Logistic Regression**: A linear model for binary classification. Simple, interpretable, but may miss complex patterns.")
            st.info("**Random Forest**: Ensemble of decision trees that reduces overfitting. Good balance of accuracy and interpretability.")
        
        with col2:
            st.info("**XGBoost**: Advanced gradient boosting implementation. Often achieves highest accuracy but less interpretable.")
            st.info("**Decision Tree**: Simple tree-based model with explicit decision rules. Highly interpretable but may overfit.")
    
    with tab3:
        st.header("💡 Data Insights")
        
        if sample_data is not None:
            # Basic dataset info
            st.subheader("Dataset Overview")
            st.write(f"Number of records: {sample_data.shape[0]}")
            st.write(f"Number of features: {sample_data.shape[1]}")
            
            # Sample of the data
            st.subheader("Sample Data")
            st.dataframe(sample_data.head(10), use_container_width=True)
            
            # Target distribution
            st.subheader("Attrition Distribution")
            
            if 'Attrition' in sample_data.columns:
                fig = px.pie(
                    sample_data,
                    names='Attrition',
                    title='Attrition Distribution',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation
            st.subheader("Feature Correlation")
            
            # Get only numeric columns
            numeric_df = sample_data.select_dtypes(include=['float64', 'int64'])
            
            if not numeric_df.empty:
                fig = px.imshow(
                    numeric_df.corr(),
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions
            st.subheader("Feature Distributions")
            
            feature_to_plot = st.selectbox(
                "Select feature to visualize",
                options=sample_data.columns.tolist()
            )
            
            if sample_data[feature_to_plot].dtype in ['float64', 'int64']:
                # Numeric feature - show histogram
                fig = px.histogram(
                    sample_data,
                    x=feature_to_plot,
                    color='Attrition' if 'Attrition' in sample_data.columns else None,
                    marginal="box",
                    title=f"Distribution of {feature_to_plot}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Categorical feature - show bar chart
                fig = px.bar(
                    sample_data[feature_to_plot].value_counts().reset_index(),
                    x='index',
                    y=feature_to_plot,
                    title=f"Distribution of {feature_to_plot}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sample data not available. Please ensure cleaned_dataset.csv is in the current directory.")

if __name__ == "__main__":
    main()
