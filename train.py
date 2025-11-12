from pathlib import Path
import joblib
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris_experiment")

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}


with mlflow.start_run(run_name="iris_model") as run:
    run_id = run.info.run_id

    # Log parameters
    mlflow.log_params(params)

    # Train
    print("Training model...")
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    print("Model trained.")

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print("Accuracy:", accuracy)

    # Save
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "iris_model.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    print(f"Model saved to {model_path.resolve()}")


    mlflow.sklearn.log_model(model, artifact_path="model")

    model_uri = f"runs:/{run_id}/model"
    model_name = "iris_model"

    result = mlflow.register_model(model_uri, model_name)
    print(f"Model registered: name={result.name}, version={result.version}")
