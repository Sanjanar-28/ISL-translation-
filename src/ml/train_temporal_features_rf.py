from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/ml")
OUTPUT_DIR = Path("data/ml_temporal")

FRAMES = 20
LANDMARK_FEATURES = 126


# ============================================================
# BUILD TEMPORAL FEATURES
# ============================================================

def build_features(X):

    """
    Convert:

        (samples, 2520)

    into temporal statistical features.

    2520 = 20 frames × 126 features

    Output:

        (samples, 1008)

    Features:

        mean
        std
        minimum
        maximum
        range
        diff_mean
        diff_std
        diff_abs_mean
    """

    if X.ndim != 2:

        raise ValueError(
            f"Expected 2D input, "
            f"got shape {X.shape}"
        )

    expected_features = (
        FRAMES
        * LANDMARK_FEATURES
    )

    if X.shape[1] != expected_features:

        raise ValueError(
            f"Expected {expected_features} "
            f"input features, "
            f"got {X.shape[1]}"
        )

    # --------------------------------------------------------
    # Reshape:
    #
    # (samples, 2520)
    #
    # →
    #
    # (samples, 20 frames, 126 features)
    # --------------------------------------------------------

    X = X.reshape(
        -1,
        FRAMES,
        LANDMARK_FEATURES
    )

    # --------------------------------------------------------
    # Frame statistics
    # --------------------------------------------------------

    mean = X.mean(axis=1)

    std = X.std(axis=1)

    minimum = X.min(axis=1)

    maximum = X.max(axis=1)

    value_range = (
        maximum
        - minimum
    )

    # --------------------------------------------------------
    # Temporal movement
    # --------------------------------------------------------

    diff = np.diff(
        X,
        axis=1
    )

    diff_mean = diff.mean(axis=1)

    diff_std = diff.std(axis=1)

    diff_abs_mean = (
        np.abs(diff)
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    features = np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            value_range,
            diff_mean,
            diff_std,
            diff_abs_mean,
        ],
        axis=1
    )

    return features.astype(
        np.float32
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    mapping_path = (
        DATA_DIR
        / "label_mapping.csv"
    )

    mapping = pd.read_csv(
        mapping_path
    )

    class_names = (
        mapping
        .sort_values("label_id")
        ["label"]
        .to_numpy()
    )

    return class_names


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "       TEMPORAL FEATURE RF "
        "ALIGNED EXPERIMENT"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load ORIGINAL aligned dataset
    #
    # IMPORTANT:
    #
    # data/ml contains:
    #
    # Train:      595
    # Validation: 127
    # Test:       130
    #
    # These are the exact same samples used by the
    # original Random Forest.
    # --------------------------------------------------------

    print(
        "\nLoading aligned original dataset..."
    )

    X_train = np.load(
        DATA_DIR
        / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR
        / "y_train.npy"
    )

    X_validation = np.load(
        DATA_DIR
        / "X_validation.npy"
    )

    y_validation = np.load(
        DATA_DIR
        / "y_validation.npy"
    )

    X_test = np.load(
        DATA_DIR
        / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR
        / "y_test.npy"
    )

    print(
        "\nOriginal aligned dataset:"
    )

    print(
        f"  Train:      "
        f"{X_train.shape}"
    )

    print(
        f"  Validation: "
        f"{X_validation.shape}"
    )

    print(
        f"  Test:       "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # Validate sample counts
    # --------------------------------------------------------

    expected_sizes = {
        "train": 595,
        "validation": 127,
        "test": 130,
    }

    actual_sizes = {
        "train": len(X_train),
        "validation": len(
            X_validation
        ),
        "test": len(X_test),
    }

    for split in expected_sizes:

        if (
            actual_sizes[split]
            != expected_sizes[split]
        ):

            raise RuntimeError(
                f"{split} size mismatch: "
                f"expected "
                f"{expected_sizes[split]}, "
                f"got "
                f"{actual_sizes[split]}"
            )

    print(
        "\nSample alignment confirmed:"
    )

    print(
        "  Train:      595"
    )

    print(
        "  Validation: 127"
    )

    print(
        "  Test:       130"
    )

    # --------------------------------------------------------
    # Build temporal features
    # --------------------------------------------------------

    print(
        "\nBuilding temporal features..."
    )

    X_train_new = build_features(
        X_train
    )

    X_validation_new = build_features(
        X_validation
    )

    X_test_new = build_features(
        X_test
    )

    print(
        "\nTemporal representation:"
    )

    print(
        f"  Train:      "
        f"{X_train_new.shape}"
    )

    print(
        f"  Validation: "
        f"{X_validation_new.shape}"
    )

    print(
        f"  Test:       "
        f"{X_test_new.shape}"
    )

    # --------------------------------------------------------
    # Validate feature dimensions
    # --------------------------------------------------------

    expected_temporal_features = 1008

    for name, X in [
        ("train", X_train_new),
        ("validation", X_validation_new),
        ("test", X_test_new),
    ]:

        if (
            X.shape[1]
            != expected_temporal_features
        ):

            raise RuntimeError(
                f"{name}: expected "
                f"{expected_temporal_features} "
                f"temporal features, "
                f"got {X.shape[1]}"
            )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        X_train_new,
        y_train
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "\nEvaluating validation..."
    )

    validation_predictions = (
        model.predict(
            X_validation_new
        )
    )

    validation_accuracy = (
        accuracy_score(
            y_validation,
            validation_predictions
        )
    )

    print(
        f"Validation accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print(
        "\nEvaluating test..."
    )

    test_predictions = (
        model.predict(
            X_test_new
        )
    )

    test_accuracy = (
        accuracy_score(
            y_test,
            test_predictions
        )
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    class_names = (
        load_class_names()
    )

    unique_labels = np.unique(
        y_test
    )

    target_names = [
        class_names[label]
        for label
        in unique_labels
    ]

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / "X_train.npy",
        X_train_new
    )

    np.save(
        OUTPUT_DIR
        / "X_validation.npy",
        X_validation_new
    )

    np.save(
        OUTPUT_DIR
        / "X_test.npy",
        X_test_new
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        OUTPUT_DIR
        / "temporal_random_forest.joblib"
    )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    validation_results = (
        pd.DataFrame(
            {
                "actual_class_id":
                    y_validation,
                "predicted_class_id":
                    validation_predictions,
                "actual_label": [
                    class_names[label]
                    for label
                    in y_validation
                ],
                "predicted_label": [
                    class_names[label]
                    for label
                    in validation_predictions
                ],
            }
        )
    )

    validation_results.to_csv(
        OUTPUT_DIR
        / "temporal_validation_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_results = pd.DataFrame(
        {
            "actual_class_id": y_test,
            "predicted_class_id":
                test_predictions,
            "actual_label": [
                class_names[label]
                for label
                in y_test
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
        / "temporal_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "        TEMPORAL RF COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nValidation accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Test accuracy:       "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        "  X_train.npy"
    )

    print(
        "  X_validation.npy"
    )

    print(
        "  X_test.npy"
    )

    print(
        "  temporal_random_forest.joblib"
    )

    print(
        "  temporal_validation_predictions.csv"
    )

    print(
        "  temporal_test_predictions.csv"
    )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()