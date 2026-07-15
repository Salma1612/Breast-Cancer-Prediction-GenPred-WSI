"""
utils.py
--------
Shared utility functions used across the breast cancer prediction pipeline:
- Reproducibility helpers
- Plotting helpers (training curves, confusion matrix, ROC curve)
- Directory / path helpers

Author: SK Salma, P. Durgamma, Nithiasri S
Project: Breast Cancer Prediction using Deep Learning (MLP)
"""

import os
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across python, numpy and (if available) tensorflow so
    that experiments are reproducible.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        # TensorFlow is not installed in this environment; numpy/python
        # seeding above is still applied.
        pass


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> str:
    """Create `path` (and parents) if it does not already exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: dict, path: str) -> None:
    """Save a dictionary as a pretty-printed JSON file."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> dict:
    """Load a dictionary from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def plot_training_history(history, save_path_prefix: str = "images/") -> None:
    """
    Plot and save training/validation accuracy and loss curves.

    Parameters
    ----------
    history : dict or keras.callbacks.History
        Either a Keras History object (history.history) or a plain dict with
        keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
    save_path_prefix : str
        Folder in which 'accuracy.png' and 'loss.png' will be written.
    """
    ensure_dir(save_path_prefix)
    hist = history.history if hasattr(history, "history") else history

    # --- Accuracy curve -----------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(hist["accuracy"], label="Training Accuracy", linewidth=2)
    plt.plot(hist["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_prefix, "accuracy.png"), dpi=150)
    plt.close()

    # --- Loss curve ----------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(hist["loss"], label="Training Loss", linewidth=2)
    plt.plot(hist["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_prefix, "loss.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str = "images/confusion_matrix.png",
                           class_names=("Benign", "Malignant")) -> np.ndarray:
    """
    Compute and save a confusion matrix heatmap.

    Returns
    -------
    np.ndarray : the raw confusion matrix
    """
    ensure_dir(os.path.dirname(save_path) or ".")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def plot_roc_curve(y_true, y_prob, save_path: str = "images/roc_curve.png") -> float:
    """
    Compute and save the ROC curve; returns the AUC score.
    """
    ensure_dir(os.path.dirname(save_path) or ".")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", linewidth=2,
              label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return roc_auc


def print_metrics_report(metrics: dict) -> None:
    """Pretty-print a metrics dictionary to the console."""
    print("\n" + "=" * 45)
    print(" MODEL EVALUATION SUMMARY")
    print("=" * 45)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<20}: {value:.4f}")
        else:
            print(f"{key:<20}: {value}")
    print("=" * 45 + "\n")
