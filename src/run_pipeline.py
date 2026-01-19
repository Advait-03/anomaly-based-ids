import os
import json
import pandas as pd

from src.preprocess import preprocess_data
from src.train_isolation_forest import train_isolation_forest, save_model
from src.evaluate import get_anomaly_scores, choose_threshold, evaluate_model


COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]


def load_txt(path):
    return pd.read_csv(path, names=COLUMNS)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(project_root, "data", "raw", "KDDTrain+.txt")
    test_path = os.path.join(project_root, "data", "raw", "KDDTest+.txt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Dataset not found in data/raw. Add KDDTrain+.txt and KDDTest+.txt")

    df_train = load_txt(train_path)
    df_test = load_txt(test_path)

    df_train["is_attack"] = df_train["label"].apply(lambda x: 0 if x == "normal" else 1)
    df_test["is_attack"] = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)

    X_train_processed, X_test_processed, _ = preprocess_data(df_train, df_test)

    X_train_normal = X_train_processed[df_train["is_attack"] == 0]
    model = train_isolation_forest(X_train_normal)
    save_model(model, os.path.join(project_root, "models", "isolation_forest.pkl"))

    train_scores = get_anomaly_scores(model, X_train_processed)
    threshold = choose_threshold(train_scores, percentile=95)

    scores, preds, cm, report = evaluate_model(
        model,
        X_test_processed,
        df_test["is_attack"],
        threshold
    )

    os.makedirs(os.path.join(project_root, "results"), exist_ok=True)

    results_path = os.path.join(project_root, "results", "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "threshold": float(threshold),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
            },
            f,
            indent=2
        )

    print("Pipeline completed")
    print("Model saved to: models/isolation_forest.pkl")
    print("Results saved to: results/results.json")
    print("\nConfusion Matrix:\n", cm)
    print("\nReport:\n", report)


if __name__ == "__main__":
    main()
