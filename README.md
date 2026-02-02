# Anomaly-Based Intrusion Detection System (IDS) – NSL-KDD

## Overview
This project implements an anomaly-based Intrusion Detection System (IDS) using unsupervised machine learning.
The system learns baseline normal network behavior and flags suspicious traffic as anomalies.

## Dataset
- **NSL-KDD**
  - `KDDTrain+.txt`
  - `KDDTest+.txt`

Dataset files are **not committed** to the repository.

## Approach

### Preprocessing
- One-hot encoding for categorical features:
  - `protocol_type`
  - `service`
  - `flag`
- Standard scaling for numeric features
- Preprocessing is **fit only on normal training traffic** to avoid data leakage

### Model
- Isolation Forest
- Trained **only on normal samples**
- Unsupervised anomaly detection

### Decision Rule
- The model outputs anomaly scores
- Scores are converted to alerts using a **percentile-based threshold**
- A 95th percentile threshold is used as a standard baseline
- Lower percentiles are evaluated to analyze the detection vs false-positive trade-off

## Project Structure

## How to Run

### 1) Create virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt

python -m src.run_pipeline


Output

Trained model saved to: models/isolation_forest.pkl

Evaluation results saved to: results/results.json

Confusion matrix and classification report printed in terminal

Results Summary

Best threshold tested: 80th percentile

Attack Recall: 16.61%

False Positive Rate (FPR): 0.16%

The IDS behaves conservatively, producing very low false positives at the cost of lower attack recall.

Limitations and Future Work

Low recall indicates that many attacks resemble normal traffic in feature space

Threshold tuning significantly affects detection performance

Future improvements may include:

Tuning Isolation Forest contamination

Comparing with One-Class SVM

Feature reduction to improve anomaly separation