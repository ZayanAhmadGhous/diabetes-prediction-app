from model_loader import model, scaler
# # from features import FEATURES
# # import numpy as np
# # from pathlib import Path
# # import joblib

# # BASE_DIR = Path(__file__).resolve().parent.parent
# # FEATURES = joblib.load(BASE_DIR / "model" / "features.pkl")


# # def encode_input(data):
# #     row = {feature: 0 for feature in FEATURES}

# #     # numerical + binary
# #     row['age'] = data.age
# #     row['hypertension'] = data.hypertension
# #     row['heart_disease'] = data.heart_disease
# #     row['bmi'] = data.bmi
# #     row['HbA1c_level'] = data.HbA1c_level
# #     row['blood_glucose_level'] = data.blood_glucose_level

# #     # gender encoding
# #     if data.gender.lower() == "male":
# #         row['gender_Male'] = 1

# #     # smoking encoding
# #     key = f"smoking_history_{data.smoking_history.lower()}"
# #     if key in row:
# #         row[key] = 1

# #     return [row[feature] for feature in FEATURES]


# # def make_prediction(data):
# #     input_data = encode_input(data)

# #     input_scaled = scaler.transform([input_data])

# #     prediction = model.predict(input_scaled)

# #     return int(prediction[0])
    
    
# #     # encode input
# #     # input_data = encode_input(data)

# #     # # convert to numpy array (IMPORTANT for sklearn)
# #     # input_array = np.array(input_data).reshape(1, -1)

# #     # # safety debug (remove in production if needed)
# #     # if input_array.shape[1] != scaler.n_features_in_:
# #     #     raise ValueError(
# #     #         f"Feature mismatch: model expects {scaler.n_features_in_}, "
# #     #         f"but got {input_array.shape[1]}"
# #     #     )

# #     # # scale input
# #     # input_scaled = scaler.transform(input_array)

# #     # # predict
# #     # prediction = model.predict(input_scaled)

# #     # return int(prediction[0])







# from model_loader import model, scaler
# import joblib
# import numpy as np
# from pathlib import Path


# # ---------- Load feature order (MUST match training) ----------
# BASE_DIR = Path(__file__).resolve().parent.parent
# FEATURES = joblib.load(BASE_DIR / "model" / "features.pkl")


# # ---------- Input Encoding ----------
# def encode_input(data):
#     """
#     Converts API input into model-ready feature vector
#     matching training feature order exactly.
#     """

#     # initialize all features as 0
#     row = {feature: 0 for feature in FEATURES}

#     # ---------------- numerical features ----------------
#     row["age"] = data.age
#     row["hypertension"] = data.hypertension
#     row["heart_disease"] = data.heart_disease
#     row["bmi"] = data.bmi
#     row["HbA1c_level"] = data.HbA1c_level
#     row["blood_glucose_level"] = data.blood_glucose_level

#     # ---------------- gender encoding ----------------
#     gender = data.gender.lower()
#     if f"gender_{gender.capitalize()}" in row:
#         row[f"gender_{gender.capitalize()}"] = 1

#     # ---------------- smoking history encoding ----------------
#     smoking_key = f"smoking_history_{data.smoking_history.lower()}"
#     if smoking_key in row:
#         row[smoking_key] = 1

#     # ---------------- final ordered vector ----------------
#     return [row[feature] for feature in FEATURES]


# # ---------- Prediction Function ----------
# def make_prediction(data):
#     """
#     Takes user input → encodes → scales → predicts
#     """

#     # encode input
#     input_data = encode_input(data)

#     # convert to numpy array (IMPORTANT for sklearn)
#     input_array = np.array(input_data).reshape(1, -1)

#     # safety debug (remove in production if needed)
#     if input_array.shape[1] != scaler.n_features_in_:
#         raise ValueError(
#             f"Feature mismatch: model expects {scaler.n_features_in_}, "
#             f"but got {input_array.shape[1]}"
#         )

#     # scale input
#     input_scaled = scaler.transform(input_array)

#     # predict
#     prediction = model.predict(input_scaled)

#     return int(prediction[0])




import joblib
import numpy as np
from pathlib import Path

# ---------- FIXED BASE PATH ----------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------- LOAD FEATURES ----------
FEATURES = joblib.load(BASE_DIR / "model" / "features.pkl")


# ---------- ENCODING ----------
def encode_input(data):
    row = {feature: 0 for feature in FEATURES}

    # numeric features
    row["age"] = data.age
    row["hypertension"] = data.hypertension
    row["heart_disease"] = data.heart_disease
    row["bmi"] = data.bmi
    row["HbA1c_level"] = data.HbA1c_level
    row["blood_glucose_level"] = data.blood_glucose_level

    # gender (safe)
    gender_key = f"gender_{data.gender.capitalize()}"
    if gender_key in row:
        row[gender_key] = 1

    # smoking (safe)
    smoking_key = f"smoking_history_{data.smoking_history.lower()}"
    if smoking_key in row:
        row[smoking_key] = 1

    return [row[f] for f in FEATURES]





# ---------- PREDICTION ----------
def make_prediction(data):
    input_data = encode_input(data)

    input_array = np.array(input_data).reshape(1, -1)

    # 🔥 HARD SAFETY CHECK
    if input_array.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature mismatch! Model expects {scaler.n_features_in_}, "
            f"but got {input_array.shape[1]}"
        )

    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    return int(prediction[0])