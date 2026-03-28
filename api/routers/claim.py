from fastapi import APIRouter
from api.schemas.claim_schema import ClaimPredictionRequest
from api.services.predictor import predict_claim_result

router = APIRouter()

@router.post("/claim")
def claim_status(request: ClaimPredictionRequest):
    result = predict_claim_result(request.model_dump())
    return result