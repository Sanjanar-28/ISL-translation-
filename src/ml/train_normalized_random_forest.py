from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/ml_normalized")

MODEL_PATH = (
    DATA_DIR
    / "normalized_random_forest.joblib"
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading normalized datasets...")

    X_train = np.load(
        DATA_DIR
        / "X_train_normalized.npy"
    )

    y_train = np.load(
        DATA_DIR
        / "y_train_normalized.npy"
    )

    X_validation = np.load(
        DATA_DIR
        / "X_validation_normalized.npy"
    )

    y_validation = np.load(
        DATA_DIR
        / "y_validation_normalized.npy"
    )

    X_test = np.load(
        DATA_DIR
        / "X_test_normalized.npy"
    )

    y_test = np.load(
        DATA_DIR
        / "y_test_normalized.npy"
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("       NORMALIZED RANDOM FOREST")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_data()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    print("\nDataset:")

    print(
        f"  X_train:      {X_train.shape}"
    )

    print(
        f"  X_validation: {X_validation.shape}"
    )

    print(
        f"  X_test:       {X_test.shape}"
    )

    print(
        f"\nClasses in training: "
        f"{len(np.unique(y_train))}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(

        n_estimators=500,

        max_features="sqrt",

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        f"\nValidation accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "\nModel saved to:"
    )

    print(
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                    COMPARISON")
    print("=" * 70)

    print(
        "\nOriginal tuned Random Forest:"
    )

    print(
        "  Validation: 77.17%"
    )

    print(
        "  Test:       75.38%"
    )

    print(
        "\nQuality-filtered Random Forest:"
    )

    print(
        "  Validation: 67.57%"
    )

    print(
        "  Test:       75.22%"
    )

    print(
        "\nNormalized Random Forest:"
    )

    print(
        f"  Validation: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"  Test:       "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("       NORMALIZED RF COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()