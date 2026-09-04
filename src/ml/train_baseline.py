from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = Path("data/ml")


def main():

    print("\n" + "=" * 70)
    print("                 BASELINE ML MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    X_val = np.load(DATA_DIR / "X_validation.npy")
    y_val = np.load(DATA_DIR / "y_validation.npy")

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    print("\nDataset:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_validation: {X_val.shape}")
    print(f"  y_validation: {y_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    # --------------------------------------------------------
    # Train baseline
    # --------------------------------------------------------

    print("\nTraining KNN baseline...")

    model = KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    print(
        f"\nValidation accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Detailed report
    # --------------------------------------------------------

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    import joblib

    model_path = DATA_DIR / "baseline_knn.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )

    print("\n" + "=" * 70)
    print("                 BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()