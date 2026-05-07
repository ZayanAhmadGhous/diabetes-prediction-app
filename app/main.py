# from fastapi import FastAPI
# from schemas import PatientData
# from predict import make_prediction

# app = FastAPI()

# @app.get("/")
# def home():
#     return {"message": "AI Disease Prediction API"}

# @app.post("/predict")
# def predict(data: PatientData):
#     result = make_prediction(data)
#     return {"prediction": int(result)}




from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from predict import encode_input
from pathlib import Path
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "model" / "model.pkl")
scaler = joblib.load(BASE_DIR / "model" / "scaler.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "result": None
    }
)


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    bmi: float = Form(...),
    HbA1c_level: float = Form(...),
    blood_glucose_level: float = Form(...),
    gender: str = Form(...),
    smoking_history: str = Form(...)
):

    data = type("obj", (object,), {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level,
        "gender": gender,
        "smoking_history": smoking_history
    })

    input_data = encode_input(data)
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    result = "⚠️ High risk of diabetes" if prediction == 1 else "✅ Low risk of diabetes"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result
        }
    )