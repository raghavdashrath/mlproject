import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Create output directory
os.makedirs('models', exist_ok=True)

# 1. Data Preparation
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr  =  LogisticRegression()
lr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)
print(accuracy_score(y_pred,y_test))
joblib.dump(lr,'models/model1.pkl')


# # Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




loaded_model = joblib.load('models/model1.pkl')

# Predict using the loaded model
sample_input = [[6.1, 2.8, 4.7, 1.2]]  # Example input
prediction = loaded_model.predict(sample_input)

print(f"Predicted class: {iris.target_names[prediction][0]}")
