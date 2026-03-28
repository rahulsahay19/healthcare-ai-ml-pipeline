from fastapi import APIRouter
from api.schemas.risk_schema import RiskPredictionRequest
from api.services.predictor import predict_risk_result

router = APIRouter()

@router.post("/risk")
def risk_score(request: RiskPredictionRequest):
    result = predict_risk_result(request.model_dump())
    return result