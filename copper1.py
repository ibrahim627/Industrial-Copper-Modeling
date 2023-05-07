import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st

data = pd.read_csv('copper_data.csv')

data = data[~data['Material_Reference'].str.startswith('00000')]

X_regression = data.drop(['Selling_Price'], axis=1)
y_regression = data['Selling_Price']
X_classification = data.drop(['Status'], axis=1)
y_classification = data['Status'].map({'WON': 1, 'LOST': 0})

plt.figure(figsize=(12, 6))
sns.boxplot(data=X_regression)
plt.title('Boxplot - Outlier Detection')
plt.show()

corr_matrix = X_regression.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
X_regression = X_regression.drop(to_drop, axis=1)
X_classification = X_classification.drop(to_drop, axis=1)

X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42)
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train_regression, y_train_regression)
y_pred_regression = regression_model.predict(X_test_regression)
regression_metrics = {
    'MSE': mean_squared_error(y_test_regression, y_pred_regression),
    'MAE': mean_absolute_error(y_test_regression, y_pred_regression),
    'R2': r2_score(y_test_regression, y_pred_regression)
}

classification_model = RandomForestClassifier()
classification_model.fit(X_train_classification, y_train_classification)
y_pred_classification = classification_model.predict(X_test_classification)
classification_metrics = {
    'Accuracy': accuracy_score(y_test_classification, y_pred_classification),
    'Precision': precision_score(y_test_classification, y_pred_classification),
    'Recall': recall_score(y_test_classification, y_pred_classification),
    'F1-Score': f1_score(y_test_classification, y_pred_classification),
    'AUC': roc_auc_score(y_test_classification, y_pred_classification)
}

def preprocess_regression_input(input_data):

    return preprocessed_data

def preprocess_classification_input(input_data):
    
    return preprocessed_data

def regression_prediction(input_data):
    preprocessed_data = preprocess_regression_input(input_data)
    predicted_price = regression_model.predict(preprocessed_data)
    return predicted_price

def classification_prediction(input_data):
    pre
