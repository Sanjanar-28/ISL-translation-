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
# CONFIG
# ============================================================

DATA_DIR = Path("data/ml_motion")
ORIGINAL_DIR = Path("data/ml")

MODEL_PATH = DATA_DIR / "motion_random_forest.joblib"

VALIDATION_PREDICTIONS_PATH = (
    DATA_DIR / "motion_validation_predictions.csv"
)

TEST_PREDICTIONS_PATH = (
    DATA_DIR / "motion_test_predictions.csv"
)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():

    mapping_path = ORIGINAL_DIR / "label_mapping.csv"

    if not mapping_path.exists():

        raise FileNotFoundError(
            f"Label mapping not found:\n{mapping_path}"
        )

    df = pd.read_csv(mapping_path)

    # Supports both possible column formats.
    if "label_id" in df.columns:
        id_column = "label_id"

    elif "class_id" in df.columns:
        id_column = "class_id"

    else:

        raise RuntimeError(
            "Could not find label ID column."
        )

    if "label" not in df.columns:

        raise RuntimeError(
            "Could not find label column."
        )

    return dict(
        zip(
            df[id_column],
            df["label"]
        )
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    y_true,
    y_pred,
    id_to_label,
    output_path,
):

    prediction_df = pd.DataFrame(
        {
            "actual_class_id": y_true,
            "predicted_class_id": y_pred,
            "actual_label": [
                id_to_label[int(x)]
                for x in y_true
            ],
            "predicted_label": [
                id_to_label[int(x)]
                for x in y_pred
            ],
        }
    )

    prediction_df.to_csv(
        output_path,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("       ALIGNED MOTION RANDOM FOREST")
    print("=" * 70)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [

        DATA_DIR / "X_train.npy",
        DATA_DIR / "X_validation.npy",
        DATA_DIR / "X_test.npy",

        ORIGINAL_DIR / "y_train.npy",
        ORIGINAL_DIR / "y_validation.npy",
        ORIGINAL_DIR / "y_test.npy",

        ORIGINAL_DIR / "label_mapping.csv",
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing required file:\n{file_path}"
            )

    print("\nAll required files found.")

    # --------------------------------------------------------
    # Load motion features
    # --------------------------------------------------------

    print("\nLoading motion datasets...")

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    # --------------------------------------------------------
    # Load aligned labels
    # --------------------------------------------------------

    y_train = np.load(
        ORIGINAL_DIR / "y_train.npy"
    )

    y_validation = np.load(
        ORIGINAL_DIR / "y_validation.npy"
    )

    y_test = np.load(
        ORIGINAL_DIR / "y_test.npy"
    )

    print("\nDataset shapes:")

    print(
        f"  X_train:      "
        f"{X_train.shape}"
    )

    print(
        f"  X_validation: "
        f"{X_validation.shape}"
    )

    print(
        f"  X_test:       "
        f"{X_test.shape}"
    )

    print(
        f"\n  y_train:      "
        f"{y_train.shape}"
    )

    print(
        f"  y_validation: "
        f"{y_validation.shape}"
    )

    print(
        f"  y_test:       "
        f"{y_test.shape}"
    )

    # --------------------------------------------------------
    # Validate sample alignment
    # --------------------------------------------------------

    print("\nChecking sample alignment...")

    checks = [

        (
            "Train",
            X_train.shape[0],
            y_train.shape[0],
        ),

        (
            "Validation",
            X_validation.shape[0],
            y_validation.shape[0],
        ),

        (
            "Test",
            X_test.shape[0],
            y_test.shape[0],
        ),
    ]

    for name, x_count, y_count in checks:

        if x_count != y_count:

            raise ValueError(
                f"{name} sample mismatch!\n"
                f"X samples: {x_count}\n"
                f"y samples: {y_count}"
            )

        print(
            f"  {name}: "
            f"{x_count} samples aligned"
        )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    expected_features = 5040

    for name, X in [

        ("Train", X_train),

        ("Validation", X_validation),

        ("Test", X_test),
    ]:

        if X.shape[1] != expected_features:

            raise ValueError(
                f"{name} expected "
                f"{expected_features} features, "
                f"found {X.shape[1]}"
            )

    print(
        f"\nFeature count confirmed: "
        f"{expected_features}"
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    id_to_label = load_label_mapping()

    print(
        f"Classes: "
        f"{len(id_to_label)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print(
        "\nTraining Motion Random Forest..."
    )

    model = RandomForestClassifier(

        n_estimators=500,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1,

        class_weight="balanced",

        min_samples_leaf=1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating validation..."
    )

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

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating test..."
    )

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

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification report:"
    )

    label_ids = sorted(
        id_to_label.keys()
    )

    target_names = [
        id_to_label[label_id]
        for label_id in label_ids
    ]

    print(
        classification_report(
            y_test,
            test_predictions,

            labels=label_ids,

            target_names=target_names,

            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_predictions(

        y_validation,

        validation_predictions,

        id_to_label,

        VALIDATION_PREDICTIONS_PATH,
    )

    save_predictions(

        y_test,

        test_predictions,

        id_to_label,

        TEST_PREDICTIONS_PATH,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "        ALIGNED MOTION RF COMPLETE"
    )
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
        "  data/ml_motion/"
        "motion_random_forest.joblib"
    )

    print(
        "  data/ml_motion/"
        "motion_validation_predictions.csv"
    )

    print(
        "  data/ml_motion/"
        "motion_test_predictions.csv"
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()