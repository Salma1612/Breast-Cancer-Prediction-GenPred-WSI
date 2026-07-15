"""
data_preprocessing.py
----------------------
Handles loading, cleaning, encoding, normalizing and splitting of the
Breast Cancer Wisconsin Diagnostic Dataset (WDBC), as described in
Chapter 3 & Chapter 4 of the project report.

Pipeline:
    1. Load raw CSV (data.csv) OR fall back to sklearn's bundled copy of
       the same dataset (identical 569 samples / 30 features).
    2. Drop irrelevant columns ('id', 'Unnamed: 32').
    3. Label-encode the diagnosis column (B -> 0, M -> 1).
    4. Min-Max scale all 30 numeric features to the range [0, 1].
    5. Stratified train/test split (75% / 25%) to preserve class balance.

Author: SK Salma, P. Durgamma, Nithiasri S
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.25  # 75% train / 25% test, as per report section 3.4 / 4.1


def load_dataset(csv_path: str = "dataset/data.csv") -> pd.DataFrame:
    """
    Load the WDBC dataset.

    If a local CSV (downloaded from Kaggle / UCI, see dataset/README.md) is
    present at `csv_path`, it is used directly. Otherwise the function falls
    back to the identical dataset bundled with scikit-learn
    (`sklearn.datasets.load_breast_cancer`) so the pipeline always works
    out-of-the-box, even without an internet connection.

    Returns
    -------
    pd.DataFrame
        Raw dataframe containing the 30 features + 'diagnosis' column.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df

    print(f"[INFO] '{csv_path}' not found — falling back to the built-in "
          f"scikit-learn copy of the Breast Cancer Wisconsin dataset.")
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    # sklearn uses column names like 'mean radius', 'radius error',
    # 'worst concave points'. Rename them to the exact standard WDBC/Kaggle
    # CSV convention ('radius_mean', 'radius_se', 'concave points_worst',
    # note the retained space in multi-word feature names) so column names
    # are identical regardless of which data source is used.
    # Maps sklearn's base feature name -> WDBC/Kaggle CSV base name.
    # Every base name is underscore-joined EXCEPT "concave points", which
    # keeps its literal space in the original Kaggle CSV header.
    base_features = {
        "radius": "radius", "texture": "texture", "perimeter": "perimeter",
        "area": "area", "smoothness": "smoothness", "compactness": "compactness",
        "concavity": "concavity", "concave points": "concave points",
        "symmetry": "symmetry", "fractal dimension": "fractal_dimension",
    }
    rename_map = {}
    for sk_base, wdbc_base in base_features.items():
        rename_map[f"mean {sk_base}"] = f"{wdbc_base}_mean"
        rename_map[f"{sk_base} error"] = f"{wdbc_base}_se"
        rename_map[f"worst {sk_base}"] = f"{wdbc_base}_worst"
    df = df.rename(columns=rename_map)

    # sklearn encodes target as 0 = malignant, 1 = benign (inverse of the
    # WDBC convention used in the report). We remap to match the report:
    # B (benign) -> 0 (pre-encoding), M (malignant) -> 1, stored in a
    # 'diagnosis' target column with the familiar WDBC feature names.
    df["diagnosis"] = df["target"].map({1: "B", 0: "M"})
    df = df.drop(columns=["target"])
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove irrelevant columns as described in report section 4.1
    ('id' and 'Unnamed: 32', which are artifacts of the original Kaggle CSV
    export and carry no diagnostic information).
    """
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(axis=1, how="all")  # drop fully-empty columns, if any
    return df


def encode_labels(df: pd.DataFrame, target_col: str = "diagnosis"):
    """
    Label-encode the diagnosis column: Benign (B) -> 0, Malignant (M) -> 1.

    Returns
    -------
    df : pd.DataFrame with the target column replaced by its encoded values
    encoder : fitted sklearn LabelEncoder (classes_ == ['B', 'M'])
    """
    encoder = LabelEncoder()
    encoder.fit(["B", "M"])  # force consistent ordering: B=0, M=1
    df[target_col] = encoder.transform(df[target_col])
    return df, encoder


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Apply Min-Max scaling (fit on train only, transform both) as per
    Equation (1) in the report: x_scaled = (x - x_min) / (x_max - x_min).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(csv_path: str = "dataset/data.csv", target_col: str = "diagnosis"):
    """
    Full end-to-end preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    scaler   : fitted MinMaxScaler
    encoder  : fitted LabelEncoder
    feature_names : list[str]
    """
    df = load_dataset(csv_path)
    df = clean_dataset(df)
    df, encoder = encode_labels(df, target_col)

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # preserve class ratio, as per report section 4.1
    )

    X_train, X_test, scaler = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test, scaler, encoder, feature_names


def save_preprocessors(scaler, encoder, out_dir: str = "models") -> None:
    """Persist the fitted scaler and label encoder for later inference."""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(out_dir, "label_encoder.pkl"))


def load_preprocessors(in_dir: str = "models"):
    """Load a previously fitted scaler and label encoder."""
    scaler = joblib.load(os.path.join(in_dir, "scaler.pkl"))
    encoder = joblib.load(os.path.join(in_dir, "label_encoder.pkl"))
    return scaler, encoder


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess()
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Feature count    : {X_train.shape[1]}")
    print(f"Class balance (train): "
          f"{np.bincount(y_train.astype(int))} (0=Benign, 1=Malignant)")
    save_preprocessors(scaler, encoder)
    print("Scaler and label encoder saved to 'models/'.")
