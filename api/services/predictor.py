import pandas as pd
from api.services.model_loader import load_claim_model, load_risk_model

def predict_risk_result(data: dict):
    model = load_risk_model()
    # Convert input to dataframe
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]

    response = {
        "prediction": prediction
    }
    
    # Checking for probabilities attribute
    if hasattr(model, "predict_proba"):
        probabilities = model.predict(input_df)[0]
        if hasattr(probabilities, "tolist"):
            response["probabilities"] = probabilities.tolist()
        else:
            response["probabilities"] = probabilities
    
    return response

def predict_claim_result(data: dict):
    model = load_claim_model()
    # Convert input to dataframe
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    response = {
        "prediction": prediction
    }
    
    # Checking for probabilities attribute
    if hasattr(model, "predict_proba"):
        probabilities = model.predict(input_df)[0]
        if hasattr(probabilities, "tolist"):
            response["probabilities"] = probabilities.tolist()
        else:
            response["probabilities"] = probabilities
    
    return response