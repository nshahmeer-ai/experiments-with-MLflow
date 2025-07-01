import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub

# Initialize Dagshub & MLflow
dagshub.init(repo_owner='nshahmeer-ai',
             repo_name='experiments-with-MLflow', mlflow=True)
mlflow.set_tracking_uri(
    "https://dagshub.com/nshahmeer-ai/experiments-with-MLflow.mlflow")
mlflow.set_experiment("wine_classification_experiment")

# Load data
wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.10, random_state=42)

# Params
max_depth = 8
n_estimators = 5

with mlflow.start_run():
    rf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = rf.score(x_test, y_test)

    # Log metrics and params
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.set_tag("Shahmeer", "Experiment with MLflow")

    # 🚨 Use artifact_path instead of name
    mlflow.sklearn.log_model(rf, artifact_path="RandomForestClassifier")

    # Save model
    joblib.dump(rf, "rf_model.joblib")
    mlflow.log_artifact("rf_model.joblib")

print(f"Accuracy: {accuracy}")