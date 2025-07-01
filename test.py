import mlflow
print("Printing the tracking URI scheme below")
print(mlflow.get_tracking_uri())
print('/n')


print(mlflow.set_tracking_uri("http://127.0.0.1:5000/"))
print("Printing the tracking URI scheme below")
print(mlflow.get_tracking_uri())
print('/n')


