import numpy as np
import tensorflow as tf

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    LSTM,
    Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/ml_cnn_lstm")

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"

MODEL_PATH = DATA_DIR / "cnn_lstm_40frames.keras"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

BATCH_SIZE = 16
EPOCHS = 50


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("             LOADING CNN + LSTM DATA")
print("=" * 70)

X = np.load(X_PATH)
y = np.load(Y_PATH)

print()
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", len(np.unique(y)))


# ============================================================
# CHECK DATA
# ============================================================

if X.ndim != 3:

    raise ValueError(
        f"Expected X to have 3 dimensions, got {X.ndim}"
    )

if X.shape[1] != 40:

    raise ValueError(
        f"Expected 40 frames, got {X.shape[1]}"
    )

if X.shape[2] != 126:

    raise ValueError(
        f"Expected 126 features/frame, got {X.shape[2]}"
    )


# ============================================================
# LABELS
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

num_classes = len(
    encoder.classes_
)

print(
    "Number of classes:",
    num_classes
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("                TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y_encoded
)


print()
print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# CNN + LSTM MODEL
# ============================================================

print()
print("=" * 70)
print("              BUILDING CNN + LSTM")
print("=" * 70)


model = Sequential([

    Input(
        shape=(40, 126)
    ),

    # --------------------------------------------------------
    # CNN BLOCK
    # --------------------------------------------------------

    Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.25),

    # --------------------------------------------------------
    # SECOND CNN BLOCK
    # --------------------------------------------------------

    Conv1D(
        filters=256,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.25),

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    LSTM(
        128,
        return_sequences=False
    ),

    Dropout(0.35),

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6
)


checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("                  TRAINING")
print("=" * 70)


history = model.fit(

    X_train,
    y_train,

    validation_split=0.15,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("                    RESULTS")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(

    X_test,
    y_test,

    verbose=0
)


print()
print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


print()

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            str(c)
            for c in encoder.classes_
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# Save label encoder
np.save(
    DATA_DIR / "label_classes.npy",
    encoder.classes_
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("              TRAINING COMPLETE")
print("=" * 70)

print()
print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print()
print(
    "Labels saved to:"
)

print(
    DATA_DIR / "label_classes.npy"
)

print()