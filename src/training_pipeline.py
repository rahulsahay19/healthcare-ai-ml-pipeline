import argparse
import mlflow
import mlflow.sklearn

from mlflow import register_model
from mlflow.tracking import MlflowClient

from src.config import MODEL_CONFIG
from src.utils import (
    load_model_table,
    time_based_split,
    save_feature_schema_full,
    save_local_model,
    hash_input,
    write_prediction_log
)
from src.evaluate import evaluate_model, is_eligible_for_production
from src.train import build_model_pipeline, get_input_feature_columns


# This is the main function that orchestrates the complete training pipeline:
# loading model_table, splitting data, training the pipeline, evaluating it,
# logging to MLflow, registering the model, and handling stage promotion.


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")
mlflow.set_experiment("healthcare-classification")

def run_pipeline(model_type: str):
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Invalid model type: {model_type}")

    config = MODEL_CONFIG[model_type]

    print("=" * 70)
    print(f"Starting training pipeline for model: {model_type}")
    print("=" * 70)

    # Step 1: Load model_table
    df = load_model_table()
    print(f"model_table loaded ✓ Shape: {df.shape}")

    # Step 2: Time-based split
    train_df, test_df = time_based_split(df, config["sort_column"])
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    # Step 3: Identify raw input feature columns
    input_feature_columns = get_input_feature_columns(config)
    print("Input feature columns:", input_feature_columns)

    # Step 4: Build X/y from raw dataframe
    X_train = train_df[input_feature_columns].copy()
    y_train = train_df[config["target_column"]].copy()

    X_test = test_df[input_feature_columns].copy()
    y_test = test_df[config["target_column"]].copy()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")

    # Step 5: Build preprocessing + model pipeline
    pipeline = build_model_pipeline(config)
    print("Preprocessing + model pipeline ready ✓")

    # Step 6: Train pipeline
    pipeline.fit(X_train, y_train)
    print("Model training completed ✓")

    # Step 7: Save local trained pipeline
    model_path = save_local_model(pipeline, config["local_model_file"])
    print("Local model saved at:", model_path)

    # Step 8: Save raw feature schema
    schema_path = save_feature_schema_full(MODEL_CONFIG)
    print("Feature schema saved at:", schema_path)

    # Step 9: Predict on test data
    predictions = pipeline.predict(X_test)

    # Step 10: Evaluate
    metrics = evaluate_model(
        y_test=y_test,
        predictions=predictions,
        positive_label=config["positive_label_for_recall"]
    )

    print("Evaluation completed ✓")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Weighted F1   : {metrics['weighted_f1']:.4f}")
    print(f"Target Recall : {metrics['target_recall']:.4f}")

    # Step 11: Log full pipeline to MLflow
    with mlflow.start_run(run_name=config["run_name"]) as run:
        mlflow.log_params(config["params"])
        mlflow.log_param("target_column", config["target_column"])
        mlflow.log_param("sort_column", config["sort_column"])
        mlflow.log_param("input_feature_count", len(input_feature_columns))
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("weighted_f1", metrics["weighted_f1"])
        mlflow.log_metric(
            f"{config['positive_label_for_recall'].lower()}_recall",
            metrics["target_recall"]
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model"
        )

        run_id = run.info.run_id
        print("Pipeline logged to MLflow ✓")
        print("Run ID:", run_id)

    # Step 12: Register model
    result = register_model(
        model_uri=model_info.model_uri,
        name=config["registered_model_name"]
    )

    model_version = result.version
    print("Registered Model Name :", result.name)
    print("Registered Version    :", model_version)

    client = MlflowClient()

    # Step 13: Move to staging
    client.transition_model_version_stage(
        name=config["registered_model_name"],
        version=model_version,
        stage="Staging"
    )
    print(
        f"Model {config['registered_model_name']} "
        f"version {model_version} moved to Staging ✓"
    )

    # Step 14: Production eligibility check
    eligible = is_eligible_for_production(
        accuracy=metrics["accuracy"],
        target_recall=metrics["target_recall"],
        accuracy_threshold=config["promotion_accuracy_threshold"],
        recall_threshold=config["promotion_recall_threshold"]
    )

    if eligible:
        # Step 15: Move to production
        client.transition_model_version_stage(
            name=config["registered_model_name"],
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(
            f"Model {config['registered_model_name']} "
            f"version {model_version} moved to Production ✓"
        )

        # Step 16: Load production pipeline
        production_model = mlflow.sklearn.load_model(
            model_uri=f"models:/{config['registered_model_name']}/Production"
        )

        # Step 17: Sample prediction from production pipeline
        sample_df = X_test.head(5).copy()
        sample_predictions = production_model.predict(sample_df)
        print("Sample Predictions:", sample_predictions)

        # Step 18: Prediction logging
        input_payload = sample_df.head(1).to_dict(orient="records")[0]
        input_hash = hash_input(input_payload)

        log_file = write_prediction_log(
            model_name=config["registered_model_name"],
            model_version=str(model_version),
            input_hash=input_hash,
            prediction=str(sample_predictions[0])
        )

        print("Prediction logged ✓")
        print("Log file location:", log_file)
    else:
        print("Model is NOT eligible for Production")

    print("=" * 70)
    print(f"Training pipeline completed for model: {model_type}")
    print("=" * 70)


# This function parses command-line arguments to determine which model type
# (risk or claim) should be trained.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training pipeline for risk or claim model"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["risk", "claim"],
        help="Model type to train"
    )
    return parser.parse_args()


# Entry point
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.model)