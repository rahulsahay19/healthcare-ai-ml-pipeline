from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

RISK_MODEL_PATH = BASE_DIR / "models" / "risk_model_complete_pipeline.joblib"
CLAIM_MODEL_PATH = BASE_DIR / "models" / "claim_model_complete_pipeline.joblib"

def load_risk_model():
    if not RISK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Risk Model not found at: {RISK_MODEL_PATH}")
    return joblib.load(RISK_MODEL_PATH)
    
    

def load_claim_model():
    if not CLAIM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Claim Model not found at: {CLAIM_MODEL_PATH}")
    return joblib.load(CLAIM_MODEL_PATH)
    