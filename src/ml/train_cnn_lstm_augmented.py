import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/ml_cnn_lstm")

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"

OUTPUT_MODEL = DATA_DIR / "cnn_lstm_40frames_augmented.keras"
OUTPUT_LABELS = DATA_DIR / "label_classes_augmented.npy"

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

TARGET_FRAMES = 40
FEATURES = 126
NUM_CLASSES = 49

AUGMENTATIONS_PER_SAMPLE = 2

BATCH_SIZE = 16
EPOCHS = 60

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("          CNN + LSTM WITH LANDMARK AUGMENTATION")
print("=" * 70)

print("\nLoading dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# Expected:
# X = (859, 40, 126)

# ============================================================
# SAFETY CHECK
# ============================================================

if X.ndim != 3:
    raise ValueError(
        f"Expected X to have 3 dimensions, got {X.shape}"
    )

if X.shape[1] != TARGET_FRAMES:
    raise ValueError(
        f"Expected {TARGET_FRAMES} frames, got {X.shape[1]}"
    )

if X.shape[2] != FEATURES:
    raise ValueError(
        f"Expected {FEATURES} features, got {X.shape[2]}"
    )

# ============================================================
# LABEL ENCODING
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nNumber of classes:", len(encoder.classes_))

# ============================================================
# TRAIN / TEST SPLIT
#
# IMPORTANT:
# Split BEFORE augmentation.
#
# This prevents augmented copies of the same video
# from appearing in both train and test sets.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=SEED,
    stratify=y_encoded
)

print("\nOriginal split:")
print("Training:", X_train.shape)
print("Testing :", X_test.shape)

# ============================================================
# LANDMARK AUGMENTATION
# ============================================================

def augment_sequence(sequence):
    """
    sequence shape:
        (40,126)

    Internally:
        (40,42,3)

    Returns:
        (40,126)
    """

    seq = sequence.reshape(
        TARGET_FRAMES,
        42,
        3
    ).copy()

    # --------------------------------------------------------
    # 1. GLOBAL SCALE
    # --------------------------------------------------------

    scale = np.random.uniform(
        0.92,
        1.08
    )

    seq *= scale

    # --------------------------------------------------------
    # 2. SMALL X/Y TRANSLATION
    #
    # Since the model uses normalized landmarks,
    # keep this small.
    # --------------------------------------------------------

    translation = np.random.normal(
        0,
        0.015,
        size=(1, 1, 3)
    )

    translation[:, :, 2] = 0

    seq += translation

    # --------------------------------------------------------
    # 3. SMALL ROTATION AROUND Z AXIS
    # --------------------------------------------------------

    angle = np.random.uniform(
        -0.12,
        0.12
    )

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x = seq[:, :, 0].copy()
    y = seq[:, :, 1].copy()

    seq[:, :, 0] = (
        x * cos_a -
        y * sin_a
    )

    seq[:, :, 1] = (
        x * sin_a +
        y * cos_a
    )

    # --------------------------------------------------------
    # 4. SMALL LANDMARK NOISE
    # --------------------------------------------------------

    noise = np.random.normal(
        0,
        0.004,
        size=seq.shape
    )

    seq += noise.astype(
        np.float32
    )

    # --------------------------------------------------------
    # 5. TEMPORAL JITTER
    #
    # Slightly vary which neighboring frames represent
    # the same motion.
    #
    # We don't reverse the sequence because that would
    # change the meaning of motion-based signs.
    # --------------------------------------------------------

    if np.random.random() < 0.5:

        indices = np.arange(
            TARGET_FRAMES
        )

        # Small random perturbation
        # while preserving ordering.

        jitter = np.random.choice(
            [-1, 0, 1],
            size=TARGET_FRAMES,
            p=[0.15, 0.70, 0.15]
        )

        new_indices = np.clip(
            indices + jitter,
            0,
            TARGET_FRAMES - 1
        )

        # Force monotonic ordering

        new_indices = np.maximum.accumulate(
            new_indices
        )

        seq = seq[new_indices]

    return seq.reshape(
        TARGET_FRAMES,
        FEATURES
    ).astype(np.float32)


# ============================================================
# CREATE AUGMENTED TRAINING SET
# ============================================================

print("\nCreating augmented training data...")

augmented_X = [X_train]
augmented_y = [y_train]

for copy_index in range(
    AUGMENTATIONS_PER_SAMPLE
):

    print(
        f"Augmentation round "
        f"{copy_index + 1}/"
        f"{AUGMENTATIONS_PER_SAMPLE}"
    )

    X_aug = np.empty_like(
        X_train,
        dtype=np.float32
    )

    for i in range(
        len(X_train)
    ):

        X_aug[i] = augment_sequence(
            X_train[i]
        )

    augmented_X.append(
        X_aug
    )

    augmented_y.append(
        y_train.copy()
    )

X_train_aug = np.concatenate(
    augmented_X,
    axis=0
)

y_train_aug = np.concatenate(
    augmented_y,
    axis=0
)

print("\nFinal training set:")
print(
    "X_train_aug:",
    X_train_aug.shape
)

print(
    "y_train_aug:",
    y_train_aug.shape
)

print(
    "X_test:",
    X_test.shape
)

# ============================================================
# SHUFFLE TRAINING DATA
# ============================================================

shuffle_indices = np.random.permutation(
    len(X_train_aug)
)

X_train_aug = X_train_aug[
    shuffle_indices
]

y_train_aug = y_train_aug[
    shuffle_indices
]

# ============================================================
# CNN + LSTM MODEL
# ============================================================

print("\nBuilding CNN + LSTM...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(TARGET_FRAMES, FEATURES)
    ),

    # --------------------------------------------------------
    # CNN BLOCK 1
    # --------------------------------------------------------

    tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv1D(
        128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Dropout(
        0.25
    ),

    # --------------------------------------------------------
    # CNN BLOCK 2
    # --------------------------------------------------------

    tf.keras.layers.Conv1D(
        256,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Dropout(
        0.25
    ),

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    tf.keras.layers.LSTM(
        128
    ),

    tf.keras.layers.Dropout(
        0.35
    ),

    # --------------------------------------------------------
    # DENSE
    # --------------------------------------------------------

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.30
    ),

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    tf.keras.layers.Dense(
        NUM_CLASSES,
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

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_MODEL,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

history = model.fit(

    X_train_aug,
    y_train_aug,

    validation_split=0.15,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    shuffle=True,

    verbose=1
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    OUTPUT_MODEL
)

# ============================================================
# TEST
# ============================================================

print("\nEvaluating on untouched test set...")

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

predictions = np.argmax(
    probabilities,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=3
    )
)

# ============================================================
# SAVE LABELS
# ============================================================

np.save(
    OUTPUT_LABELS,
    encoder.classes_
)

print("\nSaved:")
print(
    OUTPUT_MODEL
)

print(
    OUTPUT_LABELS
)

print("\nTraining complete.")