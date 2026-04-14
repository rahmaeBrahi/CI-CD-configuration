import mlflow
import os

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
    os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"cheking mlflo run id: {run_id}")

run = mlflow.get_run(run_id)

accuracy = run.data.metrics["accuracy"]
print(f"modle acurcy for run id {run_id}: {accuracy:.4f}")

threshold = 0.85

if accuracy >= threshold:
    print(f"modle acurcy {accuracy:.4f} is good for the treshold {threshold:.2f}. go to deploy.")
else:
    print(f"modle acurcy {accuracy:.4f} is low for the treshold {threshold:.2f}. stop deploy.")
    exit(1)