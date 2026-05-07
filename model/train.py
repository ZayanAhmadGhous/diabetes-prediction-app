# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# import joblib
# from preprocess import preprocess_data

# df = pd.read_csv(r"D:\Project\diabeties_predictioin\data\diabetes_prediction_dataset.csv")

# X, y = preprocess_data(df)

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# joblib.dump(model, "model.pkl")



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from preprocess import preprocess_data

df = pd.read_csv(r"D:\Project\diabeties_predictioin\data\diabetes_prediction_dataset.csv")

# preprocess
X,X_scaled, y = preprocess_data(df)

# SAVE FEATURES (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "model/features.pkl")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# SAVE BOTH
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")