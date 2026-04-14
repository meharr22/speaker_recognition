import numpy as np
import joblib
from .feature import extract_features

def predict(file_path, model_path, labels):

    model = joblib.load(model_path)
    scaler = joblib.load(model_path + "_scaler")

    features = extract_features(file_path)
    features = scaler.transform([features])

    prediction = model.predict_proba(features)

    idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return labels[idx], confidence