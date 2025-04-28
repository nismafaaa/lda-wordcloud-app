import mlflow
import time

def test_mlflow_connection():
    # Set the tracking URI to the MLflow container
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Set experiment name
    mlflow.set_experiment("TestExperiment")
    
    print("Starting MLflow run...")
    with mlflow.start_run():
        # Log a dummy parameter
        mlflow.log_param("test_param", "test_value")
        
        # Log a dummy metric
        mlflow.log_metric("test_metric", 1)
        
        # Simulate some processing time
        time.sleep(2)
        
    print("Test run completed.")

if __name__ == "__main__":
    test_mlflow_connection()
