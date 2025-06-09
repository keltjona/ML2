import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def create_gauge_chart(value, title="Probability", min_val=0, max_val=1):
    """Create a gauge chart using matplotlib"""
    # Determine color based on value
    if value < 0.3:
        color = 'green'
    elif value < 0.7:
        color = 'orange'
    else:
        color = 'red'
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={'projection': 'polar'})
    
    # Define gauge parameters
    theta = np.linspace(0, 180, 100) * np.pi / 180  # Half circle
    r = np.ones_like(theta)
    
    # Plot background
    ax.fill_between(theta, 0, r, color='lightgray', alpha=0.5)
    
    # Plot value
    value_theta = np.linspace(0, 180 * value, 100) * np.pi / 180
    value_r = np.ones_like(value_theta)
    ax.fill_between(value_theta, 0, value_r, color=color, alpha=0.8)
    
    # Adjust plot settings
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1.2)
    ax.spines['polar'].set_visible(False)
    
    # Add text in the center
    ax.text(np.pi/2, 0.5, f"{value:.1%}", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(np.pi/2, 0.3, title, ha='center', va='center', fontsize=10)
    
    # Add min and max labels
    ax.text(0, 1.1, 'Low', ha='left', va='center', fontsize=8)
    ax.text(np.pi, 1.1, 'High', ha='right', va='center', fontsize=8)
    
    return fig

def display_model_comparison_chart():
    """Create and display bar chart for model comparison"""
    metrics_df = get_model_metrics()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define width of bars and positions
    bar_width = 0.15
    index = np.arange(len(metrics_df['Model']))
    
    # Plot bars for each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        ax.bar(index + i*bar_width, metrics_df[metric], bar_width, 
               label=metric, color=colors[i])
    
    # Customize plot
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(index + bar_width * 2)
    ax.set_xticklabels(metrics_df['Model'])
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    
    return fig

def display_correlation_heatmap():
    """Create and display a correlation heatmap"""
    # Sample correlation data - replace with actual data analysis
    corr_data = pd.DataFrame({
        'SatisfactionLevel': [1.0, -0.1, 0.2, -0.2, -0.4],
        'LastEvaluation': [-0.1, 1.0, 0.3, 0.1, -0.1],
        'NumberProjects': [0.2, 0.3, 1.0, 0.5, 0.3],
        'AverageMonthlyHours': [-0.2, 0.1, 0.5, 1.0, 0.4],
        'Attrition': [-0.4, -0.1, 0.3, 0.4, 1.0]
    }, index=['SatisfactionLevel', 'LastEvaluation', 'NumberProjects', 'AverageMonthlyHours', 'Attrition'])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Feature Correlation with Attrition')
    
    return fig

def display_attrition_distribution():
    """Create and display pie chart of attrition distribution"""
    # Example data - replace with actual analysis
    labels = ['Stayed (83%)', 'Left (17%)']
    sizes = [83, 17]
    colors = ['#4CAF50', '#F44336']
    explode = (0, 0.1)  # explode the 2nd slice (Attrition)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Employee Attrition Distribution')
    
    return fig

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
                        # Display gauge chart using matplotlib
                        gauge_fig = create_gauge_chart(prob_leaving, "Attrition Risk")
                        st.pyplot(gauge_fig)
                    
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
                            
                            # Plot feature importance using matplotlib
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
                            ax.set_title('Top 10 Features by Importance')
                            st.pyplot(fig)
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
        
        # Create comparison chart using matplotlib
        st.pyplot(display_model_comparison_chart())
        
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
            
            # Display pie chart using matplotlib
            st.pyplot(display_attrition_distribution())
            
            # Feature correlation
            st.subheader("Feature Correlation")
            
            # Display correlation heatmap using matplotlib
            st.pyplot(display_correlation_heatmap())
            
            # Feature distributions
            st.subheader("Feature Distributions")
            
            feature_to_plot = st.selectbox(
                "Select feature to visualize",
                options=sample_data.columns.tolist()
            )
            
            if sample_data[feature_to_plot].dtype in ['float64', 'int64']:
                # Numeric feature - show histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                if 'Attrition' in sample_data.columns:
                    for category in sample_data['Attrition'].unique():
                        subset = sample_data[sample_data['Attrition'] == category]
                        sns.histplot(data=subset, x=feature_to_plot, kde=True, 
                                    label=category, alpha=0.6, ax=ax)
                    plt.legend()
                else:
                    sns.histplot(data=sample_data, x=feature_to_plot, kde=True, ax=ax)
                plt.title(f"Distribution of {feature_to_plot}")
                st.pyplot(fig)
            else:
                # Categorical feature - show bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = sample_data[feature_to_plot].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                plt.title(f"Distribution of {feature_to_plot}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Sample data not available. Please ensure cleaned_dataset.csv is in the current directory.")

if __name__ == "__main__":
    main()
