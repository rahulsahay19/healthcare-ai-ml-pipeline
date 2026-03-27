import os
import json
import hashlib
import joblib
import pandas as pd
from datetime import datetime


# Utility functions for the healthcare risk and claim prediction project.

def get_base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# This function constructs the file path for the model table CSV file.
def get_model_table_path() -> str:
    return os.path.join(get_base_dir(), "outputs", "model_table.csv")


# This function loads the model table from the outputs folder and returns it as a pandas DataFrame.
def load_model_table() -> pd.DataFrame:
    model_table_path = get_model_table_path()

    if not os.path.exists(model_table_path):
        raise FileNotFoundError(f"model_table.csv not found at: {model_table_path}")

    return pd.read_csv(model_table_path)


# This function performs a time-based split of the dataframe into training and testing sets
# based on the specified sort column and split ratio.
def time_based_split(
    df: pd.DataFrame,
    sort_column: str,
    split_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sort_column not in df.columns:
        raise ValueError(f"Sort column '{sort_column}' not found in dataframe")

    df = df.copy()
    df[sort_column] = pd.to_datetime(df[sort_column])
    df = df.sort_values(sort_column).reset_index(drop=True)

    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


# Save Feature Schema
def save_feature_schema_full(config_dict: dict) -> str:
    """
    Saves full feature schema for both risk and claim models.
    This acts as a contract between model and API.
    """

    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    schema_path = os.path.join(output_dir, "feature_schema.json")

    schema = {
        "risk_model_features": config_dict["risk"]["input_features"],
        "claim_model_features": config_dict["claim"]["input_features"],
        "risk_target": config_dict["risk"]["target_column"],
        "claim_target": config_dict["claim"]["target_column"],
        "risk_time_column": config_dict["risk"]["sort_column"],
        "claim_time_column": config_dict["claim"]["sort_column"],
        "split_strategy": "earliest 80 percent train, latest 20 percent test"
    }

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4)

    return schema_path


# This function loads the feature schema JSON file from the outputs directory.
def load_feature_schema() -> dict:
    base_dir = get_base_dir()
    schema_path = os.path.join(base_dir, "outputs", "feature_schema.json")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"feature_schema.json not found at: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


# This function saves the trained pipeline/model to a local file in the models directory using joblib.
# It returns the path to the saved file.
def save_local_model(model, file_name: str) -> str:
    base_dir = get_base_dir()
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, file_name)
    joblib.dump(model, model_path)

    return model_path


# This function loads a locally saved pipeline/model from the models directory.
def load_local_model(file_name: str):
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "models", file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    return joblib.load(model_path)


# This function generates a SHA-256 hash of the input payload dictionary.
# It first converts the dictionary to a JSON string with sorted keys to ensure consistent hashing,
# and then computes the hash of the string.
def hash_input(payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode()).hexdigest()


# This function writes a prediction log entry to a log file in the logs directory.
# Each log entry includes the timestamp, model name, model version, input hash, and the prediction result.
# The log is stored in JSON format, with one entry per line.
def write_prediction_log(
    model_name: str,
    model_version: str,
    input_hash: str,
    prediction: str
) -> str:
    base_dir = get_base_dir()
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "predictions.log")

    prediction_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "input_hash": input_hash,
        "prediction": prediction
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction_log) + "\n")

    return log_file