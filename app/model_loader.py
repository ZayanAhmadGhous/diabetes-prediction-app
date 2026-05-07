import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  

MODEL_PATH = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)