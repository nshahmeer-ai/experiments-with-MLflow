import mlflow
import mlflow.sklearn
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Load the wine dataset
wine = load_wine()
x=wine.data
y=wine.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Deine the parameters for the model
max_depth = 5
n_estimators = 10

with mlflow.start_run():
    rf= RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = rf.score(x_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # create a confusion matrix
    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("confusion_matrix.png")

    # log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)


print(f"Accuracy: {accuracy}")
# Save the model 