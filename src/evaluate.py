"""
evaluate.py
-----------
Loads a trained model and evaluates it on the held-out test split, producing
all the metrics discussed in report Chapter 5:
    - Accuracy
    - Precision
    - Recall (Sensitivity)
    - F1-score
    - ROC-AUC
    - Confusion Matrix

Usage
-----
    python src/evaluate.py --model models/trained_model.h5

Author: SK Salma, P. Durgamma, Nithiasri S
"""

import argparse

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from tensorflow import keras

from data_preprocessing import preprocess, load_preprocessors
from utils import plot_confusion_matrix, plot_roc_curve, print_metrics_report, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Breast Cancer MLP model.")
    parser.add_argument("--model", type=str, default="models/trained_model.h5")
    parser.add_argument("--dataset", type=str, default="dataset/data.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold applied to sigmoid output.")
    return parser.parse_args()


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Run inference on the test set and compute the full metrics suite.

    Returns
    -------
    dict of metric name -> value, plus raw predictions for downstream plots.
    """
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


def main():
    args = parse_args()

    print("[1/3] Loading test data...")
    _, X_test, _, y_test, _, _, _ = preprocess(csv_path=args.dataset)

    print("[2/3] Loading trained model...")
    model = keras.models.load_model(args.model)

    print("[3/3] Running evaluation...")
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, args.threshold)
    print_metrics_report(metrics)

    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    cm = plot_confusion_matrix(y_test, y_pred, save_path="images/confusion_matrix.png")
    print(f"Confusion Matrix:\n{cm}")

    roc_auc = plot_roc_curve(y_test, y_prob, save_path="images/roc_curve.png")
    print(f"ROC-AUC (recomputed from curve): {roc_auc:.4f}")

    save_json(metrics, "models/evaluation_metrics.json")
    print("\nEvaluation artifacts saved: images/confusion_matrix.png, "
          "images/roc_curve.png, models/evaluation_metrics.json")


if __name__ == "__main__":
    main()
