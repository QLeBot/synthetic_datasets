import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ChatGPT generated code
def evaluate_dataset(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Drop non-relevant features
    df = df.drop(columns=['Transaction ID', 'Date', 'Customer ID'])  # Drop IDs & Date
    
    # Define High Spender threshold (75th percentile of Total Amount)
    threshold = df['Total Amount'].quantile(0.75)
    
    # Create target variable (High Spender: 1, Low Spender: 0)
    df['High Spender'] = (df['Total Amount'] > threshold).astype(int)
    
    # Drop "Total Amount" since we're predicting it as a category
    df = df.drop(columns=['Total Amount'])
    
    # Convert categorical features to numeric
    df = pd.get_dummies(df, columns=['Gender', 'Product Category'], drop_first=True)

    # Define features & target variable
    X = df.drop(columns=['High Spender'])
    y = df['High Spender']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Compare two datasets
dataset1_metrics = evaluate_dataset("sample_data/retail_sales_dataset.csv")
#dataset2_metrics = evaluate_dataset("dataset2.csv", "target_column")

print("Dataset 1 Performance:", dataset1_metrics)
#print("Dataset 2 Performance:", dataset2_metrics)

# MistralAI generated code
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')

# Preprocess the data (example: handling missing values, encoding categorical variables)
# Assume that the target variable is named 'target' and all other columns are features

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(dataset1.drop('target', axis=1), dataset1['target'], test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(dataset2.drop('target', axis=1), dataset2['target'], test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

# Train the model on dataset1
model = RandomForestClassifier(random_state=42)
model.fit(X1_train, y1_train)

# Evaluate the model on dataset1
y1_pred = model.predict(X1_test)
accuracy1 = accuracy_score(y1_test, y1_pred)
precision1 = precision_score(y1_test, y1_pred, average='weighted')
recall1 = recall_score(y1_test, y1_pred, average='weighted')
f1_1 = f1_score(y1_test, y1_pred, average='weighted')

# Train the model on dataset2
model.fit(X2_train, y2_train)

# Evaluate the model on dataset2
y2_pred = model.predict(X2_test)
accuracy2 = accuracy_score(y2_test, y2_pred)
precision2 = precision_score(y2_test, y2_pred, average='weighted')
recall2 = recall_score(y2_test, y2_pred, average='weighted')
f1_2 = f1_score(y2_test, y2_pred, average='weighted')

# Compare the results
print("Dataset 1 Evaluation Metrics:")
print(f"Accuracy: {accuracy1}")
print(f"Precision: {precision1}")
print(f"Recall: {recall1}")
print(f"F1-Score: {f1_1}")

print("\nDataset 2 Evaluation Metrics:")
print(f"Accuracy: {accuracy2}")
print(f"Precision: {precision2}")
print(f"Recall: {recall2}")
print(f"F1-Score: {f1_2}")
"""