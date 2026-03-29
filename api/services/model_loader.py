import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()


def get_model_version(model_name: str, stage: str) -> str:
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise Exception(f"No model found for {model_name} in stage {stage}")
    return versions[0].version


def load_risk_model():
    model_name = "HealthcareRiskRFModel"
    stage = "Production"   # Risk is in Production

    model_uri = f"models:/{model_name}/{stage}"

    model = mlflow.pyfunc.load_model(model_uri)
    version = get_model_version(model_name, stage)

    return model, model_name, version


def load_claim_model():
    model_name = "HealthcareClaimRFModel"
    stage = "Staging"   # Claim is NOT in Production as this has failed threshold test

    model_uri = f"models:/{model_name}/{stage}"

    model = mlflow.pyfunc.load_model(model_uri)
    version = get_model_version(model_name, stage)

    return model, model_name, version