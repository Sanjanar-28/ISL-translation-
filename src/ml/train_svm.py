from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/ml")
MODEL_PATH = DATA_DIR / "svm_rbf.joblib"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("                    SVM EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    )
    y_validation = np.load(
        DATA_DIR / "y_validation.npy"
    )

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    print(f"\nX_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test:       {X_test.shape}")

    # --------------------------------------------------------
    # SVM pipeline
    # --------------------------------------------------------

    print("\nTraining RBF SVM...")

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_pred = model.predict(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    print(
        f"\nValidation accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_pred = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            test_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\n" + "=" * 70)
    print("                    SVM COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()