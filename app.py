import gradio as gr
import joblib
import numpy as np
import os

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

# Feature order (IMPORTANT - same as training)
FEATURES = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
    'gender_Male',
    'smoking_history_current',
    'smoking_history_ever',
    'smoking_history_former',
    'smoking_history_never',
    'smoking_history_not current'
]

def encode_input(age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking):

    row = {feature: 0 for feature in FEATURES}

    row['age'] = age
    row['hypertension'] = hypertension
    row['heart_disease'] = heart_disease
    row['bmi'] = bmi
    row['HbA1c_level'] = hba1c
    row['blood_glucose_level'] = glucose

    if gender.lower() == "male":
        row['gender_Male'] = 1

    key = f"smoking_history_{smoking.lower()}"
    if key in row:
        row[key] = 1

    return [row[feature] for feature in FEATURES]


def predict(age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking):
    try:
        input_data = encode_input(age, hypertension, heart_disease, bmi, hba1c, glucose, gender, smoking)

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]

        return "⚠️ High risk of diabetes" if prediction == 1 else "✅ Low risk of diabetes"

    except Exception as e:
        return f"Error: {str(e)}"


# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Hypertension (0 = No, 1 = Yes)"),
        gr.Radio([0, 1], label="Heart Disease (0 = No, 1 = Yes)"),
        gr.Number(label="BMI"),
        gr.Number(label="HbA1c Level"),
        gr.Number(label="Blood Glucose Level"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Dropdown(["current", "ever", "former", "never", "not current"], label="Smoking History")
    ],
    outputs="text",
    title="🧠 AI Diabetes Prediction System",
    description="Enter patient details to predict diabetes risk"
)

if __name__ == "__main__":
    interface.launch()