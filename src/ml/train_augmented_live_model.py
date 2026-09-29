import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/ml_live")

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"

OUTPUT_MODEL = DATA_DIR / "augmented_live_random_forest.joblib"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Number of augmented copies per original sample
AUGMENTATION_COPIES = 3


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("          LOADING LIVE LANDMARK DATASET")
print("=" * 70)

X = np.load(X_PATH)
y = np.load(Y_PATH)

print()
print("Original X shape:", X.shape)
print("Original y shape:", y.shape)
print("Number of classes:", len(np.unique(y)))


# ============================================================
# CHECK FEATURE SIZE
# ============================================================

if X.shape[1] != 2520:

    raise ValueError(
        f"Expected 2520 features, got {X.shape[1]}"
    )


# ============================================================
# AUGMENTATION
# ============================================================

def augment_sample(sample, rng):
    """
    Input:
        sample = 2520 features

    Shape:
        20 frames × 42 landmarks × 3 coordinates

    Augmentations:
        - small global scale
        - small x/y translation
        - small rotation around Z
        - small coordinate noise

    The transformations are intentionally small.
    """

    frames = sample.reshape(
        20,
        42,
        3
    ).copy()

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    scale = rng.uniform(
        0.92,
        1.08
    )

    frames *= scale

    # --------------------------------------------------------
    # TRANSLATION
    # --------------------------------------------------------

    translation = np.array([
        rng.uniform(-0.025, 0.025),
        rng.uniform(-0.025, 0.025),
        rng.uniform(-0.015, 0.015)
    ])

    frames += translation

    # --------------------------------------------------------
    # SMALL ROTATION AROUND Z AXIS
    # --------------------------------------------------------

    angle = rng.uniform(
        -0.12,
        0.12
    )

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotation = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0]
    ])

    frames = frames @ rotation.T

    # --------------------------------------------------------
    # SMALL LANDMARK NOISE
    # --------------------------------------------------------

    noise = rng.normal(
        0,
        0.006,
        frames.shape
    )

    frames += noise

    return frames.reshape(-1).astype(
        np.float32
    )


# ============================================================
# CREATE AUGMENTED DATASET
# ============================================================

print()
print("=" * 70)
print("             CREATING AUGMENTED DATA")
print("=" * 70)

rng = np.random.default_rng(
    RANDOM_STATE
)

augmented_X = []
augmented_y = []


# Keep original samples
for i in range(len(X)):

    augmented_X.append(
        X[i].astype(np.float32)
    )

    augmented_y.append(
        y[i]
    )


# Create augmented copies
for copy_number in range(
    AUGMENTATION_COPIES
):

    print(
        f"Creating augmentation "
        f"{copy_number + 1}/"
        f"{AUGMENTATION_COPIES}..."
    )

    for i in range(len(X)):

        augmented_sample = augment_sample(
            X[i],
            rng
        )

        augmented_X.append(
            augmented_sample
        )

        augmented_y.append(
            y[i]
        )


X_aug = np.asarray(
    augmented_X,
    dtype=np.float32
)

y_aug = np.asarray(
    augmented_y
)


print()
print("Original samples:", len(X))
print("Augmented samples:", len(X_aug))
print("Features:", X_aug.shape[1])
print("Classes:", len(np.unique(y_aug)))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("                TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_aug,
    y_aug,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_aug
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("             TRAINING AUGMENTED RF")
print("=" * 70)

model = RandomForestClassifier(

    n_estimators=500,

    max_features="sqrt",

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    min_samples_leaf=1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("                    RESULTS")
print("=" * 70)

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    OUTPUT_MODEL
)


print()
print("=" * 70)
print("               TRAINING COMPLETE")
print("=" * 70)

print()
print("Model saved to:")

print(
    OUTPUT_MODEL
)

print()