import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.tensorflow
import os

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
    os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN

if MLFLOW_EXPERIMENT:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

data_path = os.path.join('mlops', 'MNIST.csv')
if not os.path.exists(data_path):
    data_path = 'MNIST.csv'

df = pd.read_csv(data_path, header=0)
labels = df.iloc[:, 0]
images = df.iloc[:, 1:].values

images = images.reshape(-1, 8, 8, 1).astype('float32')
images = images / 255.0 

images_flat = images.reshape(images.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with mlflow.start_run():
    model = Sequential([
        Flatten(input_shape=(64,)), 
        
        Dense(2, activation='relu'),
        Dense(10, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"modle acurcy: {accuracy:.4f}")

    mlflow.log_param("epochs", 5)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", loss)

    mlflow.tensorflow.log_model(model, "mnist_classifier")

    run_id = mlflow.active_run().info.run_id
    print(f"mlflo run id is: {run_id}")
    with open('model_info.txt', 'w') as f:
        f.write(run_id)

print("traning finish and run id save to file")
