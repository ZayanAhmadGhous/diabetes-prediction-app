# 🩺 Diabetes Prediction Web Application

A machine learning–powered web application that predicts the likelihood of diabetes based on user health parameters.  
Built using **FastAPI** for the backend and a custom **HTML/CSS frontend** for user interaction.

---

## 🚀 Features

- 🔍 Predicts diabetes risk using a trained ML model  
- ⚡ Real-time API responses with FastAPI  
- 🎨 Clean and responsive HTML/CSS user interface  
- 🧠 Scalable and modular backend architecture  
- 🚀 Lightweight and fast inference system  

---

## 📊 Input Features

The model uses the following health indicators:

- Age  
- Hypertension  
- Heart Disease  
- BMI (Body Mass Index)  
- HbA1c Level  
- Blood Glucose Level  
- Gender (Male/Female)  
- Smoking History  

---

## 🧠 Machine Learning Model

- **Algorithm:** Classification model (e.g., Logistic Regression / Random Forest)  
- **Preprocessing:** Feature scaling and encoding  

### Libraries Used
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## 🛠️ Tech Stack

- **Backend:** FastAPI  
- **Frontend:** HTML, CSS  
- **Language:** Python  
- **ML Framework:** Scikit-learn  
- **Data Processing:** NumPy, Pandas  

---

## 📁 Project Structure

```text
DIABETES_PREDICTION/
│
├── app/
│   ├── static/                  # CSS
│   ├── templates/               # HTML UI
│   ├── app_ui.py
│   ├── features.py
│   ├── main.py                  # Entry point
│   ├── model_loader.py
│   ├── predict.py
│   └── schemas.py
│
├── data/
│   └── diabetes_prediction_dataset.csv
│
├── model/
│   ├── features.pkl
│   ├── model.pkl
│   ├── preprocess.py
│   ├── scaler.pkl
│   ├── train.py
│   └── test.ipynb
│
├── notebooks/
│   └── diabetes_prediction_model.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore

```


## ⚙️ Installation & Setup

### 1. Clone the Repository
```
git clone https://github.com/ZayanAhmadGhous/diabetes-prediction-app.git
cd diabetes-prediction-app
```


### 2. Create Virtual Environment

```
python -m venv venv
```

### 3. Activate Environment

Linux / Mac
```
source venv/bin/activate
```
Windows
```
venv\Scripts\activate
```

### 4. Install Dependencies
```
pip install -r requirements.txt
```

### 5. Run the FastAPI Server

```
uvicorn app.main:app --reload
```

### 6. Open in Browser
```
http://127.0.0.1:8000
```

📡 API Endpoints


🔹 Home
```
GET /
```

🔹 Predict Diabetes Risk
```
POST /predict
```

Request Example
```
{
  "age": 45,
  "hypertension": 1,
  "heart_disease": 0,
  "bmi": 28.5,
  "HbA1c_level": 6.1,
  "blood_glucose_level": 140,
  "gender": "Male",
  "smoking_history": "former"
}
```


Response Example
```
{
  "prediction": "0"
}

```
## 📈 Future Improvements

🔐 Add authentication system

☁️ Deploy on cloud platforms (AWS / Render / Hugging Face)

🧠 Improve accuracy using ensemble models

🐳 Add Docker containerization


## 👨‍💻 Author

Zayan Ahmad Ghous


Machine Learning & MLOps Enthusiast
