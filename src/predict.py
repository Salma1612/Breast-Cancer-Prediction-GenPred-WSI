"""
predict.py
----------
Run inference with the trained model on new, unseen samples.

Usage
-----
    # Predict on a CSV file of raw (unscaled) samples with the 30 WDBC features
    python src/predict.py --input dataset/sample_input.csv

    # Predict on a single sample using the built-in demo record
    python src/predict.py --demo

Author: SK Salma, P. Durgamma, Nithiasri S
"""

import argparse

import numpy as np
import pandas as pd
from tensorflow import keras

from data_preprocessing import load_preprocessors, clean_dataset

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# A real (benign) record from the WDBC dataset, used as a demo when no
# --input file is supplied.
DEMO_SAMPLE = {
    "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46,
    "area_mean": 566.3, "smoothness_mean": 0.09779, "compactness_mean": 0.08129,
    "concavity_mean": 0.06664, "concave points_mean": 0.04781, "symmetry_mean": 0.1885,
    "fractal_dimension_mean": 0.05766, "radius_se": 0.2699, "texture_se": 0.7886,
    "perimeter_se": 2.058, "area_se": 23.56, "smoothness_se": 0.008462,
    "compactness_se": 0.0146, "concavity_se": 0.02387, "concave points_se": 0.01315,
    "symmetry_se": 0.0198, "fractal_dimension_se": 0.0023, "radius_worst": 15.11,
    "texture_worst": 19.26, "perimeter_worst": 99.7, "area_worst": 711.2,
    "smoothness_worst": 0.144, "compactness_worst": 0.1773, "concavity_worst": 0.239,
    "concave points_worst": 0.1288, "symmetry_worst": 0.2977,
    "fractal_dimension_worst": 0.07259,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict breast tumor malignancy.")
    parser.add_argument("--model", type=str, default="models/trained_model.h5")
    parser.add_argument("--preprocessors-dir", type=str, default="models")
    parser.add_argument("--input", type=str, default=None,
                         help="CSV file with raw (unscaled) WDBC features.")
    parser.add_argument("--demo", action="store_true",
                         help="Run prediction on a single built-in demo sample.")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def predict(model, scaler, X_raw: np.ndarray, threshold: float = 0.5):
    """
    Scale raw features and run the model, returning probabilities and labels.
    """
    X_scaled = scaler.transform(X_raw)
    probabilities = model.predict(X_scaled, verbose=0).ravel()
    labels = np.where(probabilities >= threshold, "Malignant", "Benign")
    return probabilities, labels


def main():
    args = parse_args()

    print("Loading model and preprocessors...")
    model = keras.models.load_model(args.model)
    scaler, _ = load_preprocessors(args.preprocessors_dir)

    if args.demo or args.input is None:
        print("No --input file supplied, running on the built-in demo sample.\n")
        df = pd.DataFrame([DEMO_SAMPLE])
    else:
        df = pd.read_csv(args.input)
        df = clean_dataset(df)
        df = df[[c for c in FEATURE_NAMES if c in df.columns]]

    X_raw = df[FEATURE_NAMES].values
    probabilities, labels = predict(model, scaler, X_raw, args.threshold)

    for i, (prob, label) in enumerate(zip(probabilities, labels)):
        print(f"Sample {i + 1}: Prediction = {label:<10} "
              f"(P(malignant) = {prob:.4f}, threshold = {args.threshold})")


if __name__ == "__main__":
    main()
