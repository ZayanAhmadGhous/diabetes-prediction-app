# import joblib 
# from sklearn.preprocessing import StandardScaler 
# from app.features import FEATURES

# scaler = StandardScaler()

# def preprocess_data(df):
#     X = df[FEATURES]
#     y = df["diabetes"]

#     X_scaled = scaler.fit_transform(X)

#     joblib.dump(scaler, "model/scaler.pkl")

#     return X_scaled, y


import joblib
from sklearn.preprocessing import StandardScaler
from app.features import FEATURES

scaler = StandardScaler()

def preprocess_data(df):
    # keep DataFrame (DO NOT convert to numpy yet)
    X = df[FEATURES]
    y = df["diabetes"]

    # scale but KEEP structure in training step
    X_scaled = scaler.fit_transform(X)

    # save scaler
    joblib.dump(scaler, "model/scaler.pkl")

    # return ORIGINAL columns for tracking
    return X, X_scaled, y