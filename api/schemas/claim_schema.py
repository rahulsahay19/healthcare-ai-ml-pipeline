from pydantic import BaseModel, Field


class ClaimPredictionRequest(BaseModel):
    age: int = Field(..., example=52)
    gender: str = Field(..., example="M")
    city: str = Field(..., example="Bangalore")
    insurance_provider: str = Field(..., example="CareOne")
    chronic_flag: int = Field(..., example=1)
    department: str = Field(..., example="Cardiology")
    visit_type: str = Field(..., example="ER")
    doctor_id: int = Field(..., example=101)
    length_of_stay_hours: float = Field(..., example=48)
    risk_score: str = Field(..., example="High")
    billed_amount: float = Field(..., example=65000.0)
    days_since_registration: int = Field(..., example=300)
    visit_frequency: int = Field(..., example=4)
    avg_los_per_patient: float = Field(..., example=36.5)
    provider_rejection_rate: float = Field(..., example=0.257)
    visit_month: int = Field(..., example=3)
    visit_dayofweek: int = Field(..., example=2)
    high_cost_visit_flag: int = Field(..., example=1)