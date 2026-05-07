🩺 Diabetes Prediction Web App

A machine learning-powered web application that predicts the likelihood of diabetes based on user health parameters. The system is built using FastAPI for backend services and a custom HTML/CSS frontend for user interaction.

🚀 Live Features
Predict diabetes risk using trained ML model
Real-time API response using FastAPI
Clean and responsive HTML/CSS UI
Scalable backend architecture
Lightweight and fast inference
📊 Input Features

The model uses the following health indicators:

Age
Hypertension
Heart Disease
BMI (Body Mass Index)
HbA1c Level
Blood Glucose Level
Gender (Male/Female)
Smoking History
🧠 Machine Learning Model
Algorithm: Trained classification model (e.g., Logistic Regression / Random Forest / etc.)
Preprocessing: Feature scaling & encoding
Libraries:
NumPy
Pandas
Scikit-learn
Matplotlib (for visualization during training)
🛠️ Tech Stack
Backend: FastAPI
Frontend: HTML, CSS
ML Framework: Scikit-learn
Language: Python
Data Processing: NumPy, Pandas

📁 Project Structure
DIABETES_PREDICTION/
│
├── app/                          
│   ├── static/                  
│   ├── templates/               
│   ├── app_ui.py                
│   ├── features.py              
│   ├── main.py                  
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


⚙️ How to Run Locally
1. Clone the repository
git clone https://github.com/ZayanAhmadGhous/diabetes-prediction-app.git
cd diabetes-prediction-app
2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3. Install dependencies
pip install -r requirements.txt
4. Run FastAPI server
uvicorn main:app --reload
5. Open in browser
http://127.0.0.1:8000
📌 API Endpoints
🔹 Home
GET /
🔹 Prediction
POST /predict

Request Example:

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

Response:

{
  "prediction": "0"
}

📈 Future Improvements
Add authentication system
Deploy on cloud (huggingFace)
Improve model accuracy with ensemble learning
Adding Dockers

👨‍💻 Author

Developed by Zayan Ahmad Ghous
Machine Learning & MLOps Enthusiast