import os
import joblib
from sklearn.ensemble import IsolationForest

def train_isolation_forest(X_train):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model

def save_model(model, path="models/isolation_forest.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path="models/isolation_forest.pkl"):
    return joblib.load(path)
