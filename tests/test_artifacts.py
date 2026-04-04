from pathlib import Path

def test_risk_model_exists():
    model_path = Path("models/risk_model_complete_pipeline.joblib")
    assert model_path.exists(), "Risk model artifact is missing"

def test_claim_model_exists():
    model_path = Path("models/claim_model_complete_pipeline.joblib")
    assert model_path.exists(), "Claim model artifact is missing"