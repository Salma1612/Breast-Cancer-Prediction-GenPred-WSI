"""
model.py
--------
Defines the Multi-Layer Perceptron (MLP) architecture used for breast
cancer classification, exactly as specified in report section 4.2:

    Layer            Neurons   Activation   Notes
    --------------------------------------------------------
    Input            30        -            30 diagnostic features
    Hidden Layer 1   16        ReLU         Captures nonlinear patterns
    Dropout          -         p = 0.3      Regularization
    Hidden Layer 2   8         ReLU         Further feature abstraction
    Dropout          -         p = 0.3      Regularization
    Output Layer     1         Sigmoid      Probability of malignancy

Compilation (report section 4.3):
    Optimizer : Adam (learning rate = 0.001)
    Loss      : Binary Cross-Entropy
    Metrics   : Accuracy

Author: SK Salma, P. Durgamma, Nithiasri S
"""

from tensorflow import keras
from tensorflow.keras import layers


INPUT_DIM = 30
HIDDEN_UNITS = (16, 8)
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001


def build_mlp_model(input_dim: int = INPUT_DIM,
                     hidden_units=HIDDEN_UNITS,
                     dropout_rate: float = DROPOUT_RATE,
                     learning_rate: float = LEARNING_RATE) -> keras.Model:
    """
    Build and compile the feed-forward MLP described in the project report.

    Parameters
    ----------
    input_dim : int
        Number of input features (30 for the WDBC dataset).
    hidden_units : tuple[int, int]
        Number of neurons in hidden layer 1 and hidden layer 2.
    dropout_rate : float
        Dropout probability applied after each hidden layer.
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    keras.Model : compiled, ready-to-train Keras model.
    """
    model = keras.Sequential(name="Breast_Cancer_MLP")
    model.add(keras.Input(shape=(input_dim,), name="input_features"))

    # Hidden Layer 1
    model.add(layers.Dense(hidden_units[0], activation="relu", name="hidden_layer_1"))
    model.add(layers.Dropout(dropout_rate, name="dropout_1"))

    # Hidden Layer 2
    model.add(layers.Dense(hidden_units[1], activation="relu", name="hidden_layer_2"))
    model.add(layers.Dropout(dropout_rate, name="dropout_2"))

    # Output Layer
    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(patience: int = 10, checkpoint_path: str = "models/trained_model.h5"):
    """
    Build the training callbacks used in report section 4.4 / 4.5:
      - EarlyStopping: halts training if val_accuracy does not improve for
        `patience` consecutive epochs, restoring the best weights.
      - ModelCheckpoint: saves the best-performing model to disk.

    Returns
    -------
    list[keras.callbacks.Callback]
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0,
    )
    return [early_stop, checkpoint]


if __name__ == "__main__":
    model = build_mlp_model()
    model.summary()
