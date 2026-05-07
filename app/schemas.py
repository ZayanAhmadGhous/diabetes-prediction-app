from pydantic import BaseModel #type:ignore

class PatientData(BaseModel):
    age: int
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    gender: str
    smoking_history: str