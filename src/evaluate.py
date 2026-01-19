
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def get_anomaly_scores(model, X):
    scores = model.decision_function(X)
    return -scores


def choose_threshold(scores, percentile=95):
    return np.percentile(scores, percentile)


def predict_anomalies(scores, threshold):
    return (scores >= threshold).astype(int)


def evaluate_model(model, X_test, y_test, threshold):
    scores = get_anomaly_scores(model, X_test)
    y_pred = predict_anomalies(scores, threshold)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    return scores, y_pred, cm, report
