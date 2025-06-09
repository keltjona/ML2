# Employee Attrition Prediction Dashboard

An interactive dashboard for predicting employee attrition using machine learning models.

## Overview

This project provides a Streamlit-based dashboard that allows HR professionals to:

- Predict whether an employee is likely to leave the company
- Compare performance metrics across different ML models
- Explore data insights and visualizations related to employee attrition

## Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Models**

   ```bash
   python model_saver.py
   ```

   This script will:
   - Load the cleaned dataset
   - Train machine learning models (Logistic Regression, Random Forest, XGBoost, Decision Tree)
   - Save the models and preprocessing pipeline

3. **Run the Dashboard**

   ```bash
   streamlit run dashboard.py
   ```

## Features

### Prediction Tab
- Input employee details
- Select a model for prediction
- View prediction results with probability gauge
- Get retention suggestions based on prediction

### Model Comparison Tab
- Compare performance metrics (accuracy, precision, recall, F1-score)
- View visualization of model performance
- Read model descriptions and use cases

### Data Insights Tab
- Explore the dataset structure
- View attrition distribution
- Examine feature correlations
- Analyze feature distributions

## Project Structure

```
ML2/
├── dashboard.py          # Main Streamlit dashboard application
├── model_saver.py        # Script to train and save models
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── cleaned_dataset.csv   # Dataset for model training
└── models/               # Directory for saved models (created by model_saver.py)
    ├── preprocessing_pipeline.pkl
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── decision_tree.pkl
```

## Models

The dashboard uses four different machine learning models:

1. **Logistic Regression** - Simple, interpretable model for baseline predictions
2. **Random Forest** - Ensemble model with good balance of accuracy and interpretability
3. **XGBoost** - Advanced gradient boosting model for high accuracy
4. **Decision Tree** - Transparent model with explicit decision rules

## Requirements

- Python 3.8+
- Libraries listed in requirements.txt
