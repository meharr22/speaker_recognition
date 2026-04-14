import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .feature import extract_features

def train_model(dataset_path, model_path):

    labels = sorted(os.listdir(dataset_path))
    X, y = [], []

    for label_idx, label in enumerate(labels):
        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):
            try:
                features = extract_features(os.path.join(folder, file))
                X.append(features)
                y.append(label_idx)
            except:
                pass

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, model_path + "_scaler")

    return labels