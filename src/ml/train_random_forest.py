import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

MOTION_DIR = Path("data/ml_motion")
ORIGINAL_DIR = Path("data/ml")
METADATA_PATH = Path("data/video_metadata_split.csv")

OUTPUT_DIR = Path("data/ml_motion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MOTION FEATURES
# ============================================================

X_train = np.load(MOTION_DIR / "X_train.npy")
X_validation = np.load(MOTION_DIR / "X_validation.npy")
X_test = np.load(MOTION_DIR / "X_test.npy")


# ============================================================
# LOAD ORIGINAL LABELS
#
# IMPORTANT:
# Motion features must use exactly the same labels and ordering
# as the original ML dataset.
# ============================================================

y_train = np.load(ORIGINAL_DIR / "y_train.npy")
y_validation = np.load(
    ORIGINAL_DIR / "y_validation.npy"
)
y_test = np.load(
    ORIGINAL_DIR / "y_test.npy"
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

mapping_path = (
    ORIGINAL_DIR / "label_mapping.csv"
)

mapping = pd.read_csv(mapping_path)

class_names = (
    mapping
    .sort_values("label_id")
    ["label"]
    .to_numpy()
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("=" * 70)
print("       RANDOM FOREST - POSITION + MOTION FEATURES")
print("=" * 70)

print("\nDataset:")
print(f"  X_train:      {X_train.shape}")
print(f"  X_validation: {X_validation.shape}")
print(f"  X_test:       {X_test.shape}")

print("\nLabels:")
print(f"  y_train:      {y_train.shape}")
print(f"  y_validation: {y_validation.shape}")
print(f"  y_test:       {y_test.shape}")


if len(X_train) != len(y_train):
    raise ValueError(
        "\nTraining sample mismatch!\n"
        f"Motion X_train: {len(X_train)}\n"
        f"Original y_train: {len(y_train)}\n\n"
        "The motion dataset must be regenerated from "
        "the same filtered dataset as data/ml."
    )


if len(X_validation) != len(y_validation):
    raise ValueError(
        "\nValidation sample mismatch!\n"
        f"Motion X_validation: {len(X_validation)}\n"
        f"Original y_validation: {len(y_validation)}\n\n"
        "The motion dataset must be regenerated from "
        "the same filtered dataset as data/ml."
    )


if len(X_test) != len(y_test):
    raise ValueError(
        "\nTest sample mismatch!\n"
        f"Motion X_test: {len(X_test)}\n"
        f"Original y_test: {len(y_test)}"
    )


# ============================================================
# EXPECTED SAMPLE COUNTS
# ============================================================

expected = {
    "train": 595,
    "validation": 127,
    "test": 130,
}

actual = {
    "train": len(X_train),
    "validation": len(X_validation),
    "test": len(X_test),
}

print("\nExpected alignment:")

for split in expected:

    print(
        f"  {split}: "
        f"{actual[split]} "
        f"(expected {expected[split]})"
    )

    if actual[split] != expected[split]:
        raise RuntimeError(
            f"\n{split} dataset is not aligned.\n"
            "Regenerate motion features first."
        )


print(
    "\nAll motion datasets are aligned "
    "with the original ML dataset."
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("Training Random Forest...")
print("=" * 70)


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# VALIDATION
# ============================================================

print("\nEvaluating validation set...")

validation_predictions = model.predict(
    X_validation
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print(
    f"Validation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


# ============================================================
# TEST
# ============================================================

print("\nEvaluating test set...")

test_predictions = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")

unique_test_labels = np.unique(
    y_test
)

target_names = [
    class_names[label]
    for label in unique_test_labels
]

print(
    classification_report(
        y_test,
        test_predictions,
        labels=unique_test_labels,
        target_names=target_names,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    OUTPUT_DIR
    / "motion_random_forest.joblib"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results = pd.DataFrame(
    {
        "actual_class_id": y_validation,
        "predicted_class_id":
            validation_predictions,
        "actual_label": [
            class_names[label]
            for label in y_validation
        ],
        "predicted_label": [
            class_names[label]
            for label
            in validation_predictions
        ],
    }
)

validation_results.to_csv(
    OUTPUT_DIR
    / "motion_validation_predictions.csv",
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = pd.DataFrame(
    {
        "actual_class_id": y_test,
        "predicted_class_id":
            test_predictions,
        "actual_label": [
            class_names[label]
            for label in y_test
        ],
        "predicted_label": [
            class_names[label]
            for label
            in test_predictions
        ],
    }
)

test_results.to_csv(
    OUTPUT_DIR
    / "motion_test_predictions.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("          MOTION RANDOM FOREST COMPLETE")
print("=" * 70)

print(
    f"\nValidation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy:       "
    f"{test_accuracy * 100:.2f}%"
)

print("\nSaved:")

print(
    "  motion_random_forest.joblib"
)

print(
    "  motion_validation_predictions.csv"
)

print(
    "  motion_test_predictions.csv"
)

print("\n" + "=" * 70)