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
    try:
        return joblib.load(PIPELINE_PATH)
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        return None

# Function to load model metrics - in practice, you would compute these during model training
def get_model_metrics():
    # These are example metrics - replace with your actual model metrics
    metrics = {
        'Model': ['logistic_regression', 'random_forest', 'xgboost', 'decision_tree'],
        'Accuracy': [0.85, 0.89, 0.91, 0.83],
        'Precision': [0.83, 0.87, 0.90, 0.80],
        'Recall': [0.81, 0.86, 0.88, 0.79],
        'F1-Score': [0.82, 0.87, 0.89, 0.80],
        'ROC AUC': [0.88, 0.92, 0.94, 0.85]
    }
    return pd.DataFrame(metrics)

def display_model_performance_comparison():
    """Create and display a bar chart comparing model performances"""
    metrics_df = get_model_metrics()
    
    # Melt the dataframe for easier plotting
    metrics_melted = pd.melt(metrics_df, id_vars=['Model'], 
                             var_name='Metric', value_name='Score')
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig

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

def display_attrition_distribution():
    """Display sample attrition distribution pie chart"""
    # Replace with actual data distribution
    labels = ['Stayed', 'Left']
    sizes = [83, 17]  # Example distribution (83% stayed, 17% left)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
    ax.axis('equal')
    plt.title('Employee Attrition Distribution')
    
    return fig

def display_correlation_heatmap():
    """Display sample correlation heatmap"""
    # Create sample correlation matrix - replace with actual data
    corr_data = {
        'Age': [1.0, -0.1, 0.2, -0.3, -0.2],
        'Salary': [-0.1, 1.0, 0.5, 0.3, -0.4],
        'JobSatisfaction': [0.2, 0.5, 1.0, 0.6, -0.7],
        'WorkLifeBalance': [-0.3, 0.3, 0.6, 1.0, -0.5],
        'Attrition': [-0.2, -0.4, -0.7, -0.5, 1.0]
    }
    corr_matrix = pd.DataFrame(corr_data, 
                              index=['Age', 'Salary', 'JobSatisfaction', 'WorkLifeBalance', 'Attrition'])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Feature Correlation with Attrition')
    
    return fig

def main():
    st.title("🔍 Employee Attrition Prediction Dashboard")
    st.markdown("""
    This dashboard allows you to predict employee attrition using multiple machine learning models.
    Enter employee details, select a model, and get predictions and insights.
    """)
    
    # Load models and preprocessing pipeline
    models = load_models()
    preprocessing_pipeline = load_preprocessing_pipeline()
    
    if not models or preprocessing_pipeline is None:
        st.warning("⚠️ Some models or the preprocessing pipeline could not be loaded. Please run the model_saver.py script first.")
        st.info("➡️ Run: `python model_saver.py` to create sample models for demonstration.")
        return
    
    # Display friendly model names for selection
    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'decision_tree': 'Decision Tree'
    }
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "Data Insights"])
    
    with tab1:
        st.header("🧮 Predict Employee Attrition")
        
        # Create columns for input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Factors")
            age = st.slider("Age", 18, 65, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 10)
            
        with col2:
            st.subheader("Job Factors")
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", 
                                                "Laboratory Technician", "Manufacturing Director",
                                                "Healthcare Representative", "Manager", "Other"])
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            job_level = st.slider("Job Level", 1, 5, 2)
            
        with col3:
            st.subheader("Satisfaction & Compensation")
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
            monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
            percent_salary_hike = st.slider("Last Salary Hike (%)", 0, 25, 15)
            overtime = st.selectbox("Works Overtime", ["Yes", "No"])
        
        # Create a dictionary of features
        features = {
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'DistanceFromHome': distance_from_home,
            'JobRole': job_role,
            'Department': department,
            'YearsAtCompany': years_at_company,
            'JobLevel': job_level,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'MonthlyIncome': monthly_income,
            'PercentSalaryHike': percent_salary_hike,
            'OverTime': overtime
        }
        
        # Model selection using the display names
        selected_display_name = st.selectbox("Select Model", list(model_display_names.values()))
        
        # Get the actual model name from the selected display name
        selected_model_name = [k for k, v in model_display_names.items() if v == selected_display_name][0]
        
        if st.button("Predict"):
            # Get the selected model
            model = models[selected_model_name]
            
            # Make prediction
            prediction, prediction_proba = predict_attrition(model, features, preprocessing_pipeline)
            
            if prediction is not None:
                st.subheader("Prediction Result")
                
                # Create columns for prediction display
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    # Display prediction result
                    if prediction == 1:
                        st.error("### 🚨 Employee is likely to leave")
                        risk_level = "High Risk"
                    else:
                        st.success("### ✅ Employee is likely to stay")
                        risk_level = "Low Risk"
                    
                    # Display probability
                    leave_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                    st.metric("Attrition Probability", f"{leave_prob:.1%}")
                    
                    # Create a gauge chart for probability visualization
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_aspect('equal')
                    
                    # Draw gauge background
                    background = plt.Circle((0.5, 0), 0.5, color='lightgray', fill=True)
                    ax.add_artist(background)
                    
                    # Draw colored gauge based on probability
                    if leave_prob < 0.3:
                        color = 'green'
                    elif leave_prob < 0.7:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    gauge = plt.Rectangle((0, 0), leave_prob, 0.2, color=color, alpha=0.8)
                    ax.add_artist(gauge)
                    
                    # Add labels
                    ax.text(0.5, 0.5, f"{leave_prob:.1%}", ha='center', va='center', 
                            fontsize=16, fontweight='bold')
                    ax.text(0.5, 0.3, f"{risk_level}", ha='center', va='center')
                    ax.text(0.1, 0.1, "Low", ha='center', va='center', fontsize=8)
                    ax.text(0.9, 0.1, "High", ha='center', va='center', fontsize=8)
                    
                    # Remove axes
                    ax.set_axis_off()
                    
                    st.pyplot(fig)
                
                with pred_col2:
                    # Display key risk factors based on the model (for demonstration)
                    st.subheader("Key Risk Factors")
                    
                    risk_factors = []
                    if overtime == "Yes":
                        risk_factors.append(("Overtime", "High", "Employee works overtime"))
                    if job_satisfaction < 3:
                        risk_factors.append(("Job Satisfaction", "High", "Low satisfaction score"))
                    if work_life_balance < 3:
                        risk_factors.append(("Work-Life Balance", "Medium", "Below average score"))
                    if years_at_company < 2:
                        risk_factors.append(("Tenure", "Medium", "Less than 2 years at company"))
                    if monthly_income < 3000:
                        risk_factors.append(("Compensation", "High", "Below market salary"))
                    
                    # If no risk factors found, add a placeholder
                    if not risk_factors:
                        risk_factors.append(("Overall", "Low", "No significant risk factors identified"))
                    
                    # Display risk factors in a table
                    risk_df = pd.DataFrame(risk_factors, columns=["Factor", "Risk Level", "Description"])
                    st.table(risk_df)
    
    with tab2:
        st.header("📊 Model Performance Comparison")
        
        # Display model metrics table
        metrics_df = get_model_metrics()
        metrics_df['Model'] = metrics_df['Model'].map(model_display_names)
        st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
        
        # Display performance comparison chart
        st.pyplot(display_model_performance_comparison())
        
        # Display model descriptions
        st.subheader("Model Descriptions")
        
        model_descriptions = {
            "Logistic Regression": "A simple statistical model that predicts binary outcomes. It's interpretable but may miss complex patterns in the data.",
            "Random Forest": "An ensemble of decision trees that reduces overfitting and handles non-linear relationships well. Good balance between accuracy and interpretability.",
            "XGBoost": "An advanced implementation of gradient boosting with high predictive power. Often achieves the best performance but is less interpretable.",
            "Decision Tree": "A simple, highly interpretable model that creates explicit decision rules. Easy to explain but may overfit with complex data."
        }
        
        for model, description in model_descriptions.items():
            st.markdown(f"**{model}**: {description}")
    
    with tab3:
        st.header("📈 Data Insights")
        
        # Create columns for visualizations
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            st.subheader("Attrition Distribution")
            st.pyplot(display_attrition_distribution())
            
            st.markdown("""
            ### Key Statistics:
            - Overall attrition rate: ~17%
            - Highest among Sales department
            - Employees with 1-2 years tenure most likely to leave
            - Low job satisfaction strongly correlated with attrition
            """)
        
        with vis_col2:
            st.subheader("Feature Correlation with Attrition")
            st.pyplot(display_correlation_heatmap())
            
            st.markdown("""
            ### Top Attrition Predictors:
            1. **Job Satisfaction** (-0.70): Higher satisfaction = lower attrition
            2. **Work-Life Balance** (-0.50): Better balance = lower attrition
            3. **Monthly Income** (-0.40): Higher salary = lower attrition
            4. **Age** (-0.20): Older employees tend to stay longer
            """)

if __name__ == "__main__":
    main()
