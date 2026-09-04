from pathlib import Path

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_DIR = Path("data/ml")


def main():

    print("\n" + "=" * 70)
    print("              RANDOM FOREST HYPERPARAMETER TUNING")
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
    print(f"  X_validation: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")

    # --------------------------------------------------------
    # Parameter combinations
    # --------------------------------------------------------

    experiments = [

        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },

        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },

        {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },

        {
            "n_estimators": 500,
            "max_depth": 30,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },

        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },

        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },

        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },

        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.5,
        },

    ]

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    print("\nStarting experiments...\n")

    for i, params in enumerate(experiments, start=1):

        print("-" * 70)
        print(f"Experiment {i}/{len(experiments)}")
        print(params)

        model = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(
            X_train,
            y_train
        )

        val_predictions = model.predict(X_val)

        val_accuracy = accuracy_score(
            y_val,
            val_predictions
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        results.append(
            {
                "params": params,
                "validation_accuracy": val_accuracy,
                "model": model,
            }
        )

    # --------------------------------------------------------
    # Find best model
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda x: x["validation_accuracy"]
    )

    best_model = best["model"]
    best_params = best["params"]
    best_val_accuracy = best["validation_accuracy"]

    print("\n" + "=" * 70)
    print("                    BEST MODEL")
    print("=" * 70)

    print("\nParameters:")

    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Evaluate best model on TEST
    # --------------------------------------------------------

    test_predictions = best_model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    model_path = (
        DATA_DIR /
        "tuned_random_forest.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"\nBest model saved to:"
        f"\n{model_path}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                    ALL RESULTS")
    print("=" * 70)

    ranked = sorted(
        results,
        key=lambda x: x["validation_accuracy"],
        reverse=True
    )

    for rank, result in enumerate(
        ranked,
        start=1
    ):

        print(
            f"\n#{rank} "
            f"Validation: "
            f"{result['validation_accuracy'] * 100:.2f}%"
        )

        print(
            result["params"]
        )

    print("\n" + "=" * 70)
    print("              RANDOM FOREST TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()