"""
train.py
--------
End-to-end training script for the Breast Cancer Prediction MLP.

Usage
-----
    python src/train.py
    python src/train.py --epochs 100 --batch-size 32 --dataset dataset/data.csv

This script:
    1. Loads and preprocesses the WDBC dataset (src/data_preprocessing.py).
    2. Builds the MLP model (src/model.py).
    3. Trains it with EarlyStopping + ModelCheckpoint.
    4. Saves the trained model, scaler, and label encoder to models/.
    5. Saves accuracy/loss curves to images/.

Author: SK Salma, P. Durgamma, Nithiasri S
"""

import argparse
import os

from data_preprocessing import preprocess, save_preprocessors
from model import build_mlp_model, get_callbacks
from utils import set_seed, plot_training_history, ensure_dir, save_json

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Breast Cancer MLP model.")
    parser.add_argument("--dataset", type=str, default="dataset/data.csv",
                         help="Path to the WDBC CSV file (falls back to sklearn's "
                              "bundled copy if not found).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                         help="EarlyStopping patience (epochs of no improvement).")
    parser.add_argument("--val-split", type=float, default=0.15,
                         help="Fraction of the training set held out for validation.")
    parser.add_argument("--model-out", type=str, default="models/trained_model.h5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir("models")
    ensure_dir("images")

    print("[1/4] Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess(
        csv_path=args.dataset
    )
    save_preprocessors(scaler, encoder, out_dir="models")
    print(f"      Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]} "
          f"| Features: {X_train.shape[1]}")

    print("[2/4] Building model...")
    model = build_mlp_model(input_dim=X_train.shape[1])
    model.summary()

    print("[3/4] Training model...")
    callbacks = get_callbacks(patience=args.patience, checkpoint_path=args.model_out)
    history = model.fit(
        X_train, y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # In case ModelCheckpoint didn't fire (e.g. very short run), always save
    # the final model explicitly too.
    model.save(args.model_out)
    print(f"      Model saved to: {args.model_out}")

    print("[4/4] Saving training curves...")
    plot_training_history(history, save_path_prefix="images/")

    final_metrics = {
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "epochs_run": len(history.history["accuracy"]),
    }
    save_json(final_metrics, "models/training_summary.json")
    print("      Training summary saved to models/training_summary.json")
    print("\nTraining complete. Run `python src/evaluate.py` to evaluate on the test set.")


if __name__ == "__main__":
    main()
