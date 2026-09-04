from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/ml_normalized")

RESULTS_PATH = (
    DATA_DIR / "model_comparison_results.csv"
)

MODEL_DIR = DATA_DIR


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading normalized dataset...")

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_validation.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_validation.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    print("\nDataset shapes:")
    print(
        f"  X_train:       {X_train.shape}"
    )
    print(
        f"  y_train:       {y_train.shape}"
    )
    print(
        f"  X_validation:  {X_val.shape}"
    )
    print(
        f"  y_validation:  {y_val.shape}"
    )
    print(
        f"  X_test:        {X_test.shape}"
    )
    print(
        f"  y_test:        {y_test.shape}"
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if X_train.shape[1] != 2520:

        raise ValueError(
            "Expected 2520 features, "
            f"found {X_train.shape[1]}"
        )

    print(
        "\nFeature count verified: 2520"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

    print("\n")
    print("=" * 70)
    print(f"                 {name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidation prediction...")

    val_pred = model.predict(
        X_val
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("Test prediction...")

    test_pred = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_weighted_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nRESULTS")
    print("-" * 70)

    print(
        f"Validation accuracy : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Macro F1  : "
        f"{val_f1:.4f}"
    )

    print(
        f"Test accuracy       : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro F1       : "
        f"{test_macro_f1:.4f}"
    )

    print(
        f"Test Weighted F1    : "
        f"{test_weighted_f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            test_pred,
            zero_division=0
        )
    )

    return {
        "model": name,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "estimator": model,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("              MODEL COMPARISON")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_data()

    results = []

    # ========================================================
    # 1. RANDOM FOREST
    # ========================================================

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    rf_result = evaluate_model(
        "RANDOM FOREST",
        rf,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    results.append(
        rf_result
    )

    # ========================================================
    # 2. SVM
    # ========================================================

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    )

    svm_result = evaluate_model(
        "SVM",
        svm,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    results.append(
        svm_result
    )

    # ========================================================
    # 3. MLP
    # ========================================================

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=42,
        verbose=True,
    )

    mlp_result = evaluate_model(
        "MLP",
        mlp,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    results.append(
        mlp_result
    )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                 MODEL COMPARISON")
    print("=" * 70)

    comparison_rows = []

    for result in results:

        comparison_rows.append({
            "model": result["model"],
            "validation_accuracy":
                result["validation_accuracy"],
            "validation_macro_f1":
                result["validation_macro_f1"],
            "test_accuracy":
                result["test_accuracy"],
            "test_macro_f1":
                result["test_macro_f1"],
            "test_weighted_f1":
                result["test_weighted_f1"],
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"\nComparison saved to:"
        f"\n{RESULTS_PATH}"
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_index = comparison[
        "validation_accuracy"
    ].idxmax()

    best_name = comparison.loc[
        best_index,
        "model"
    ]

    print("\n")
    print("=" * 70)
    print("                 BEST MODEL")
    print("=" * 70)

    print(
        f"\nBest model based on "
        f"validation accuracy:"
    )

    print(
        f"  {best_name}"
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    for result in results:

        model_name = (
            result["model"]
            .lower()
            .replace(" ", "_")
        )

        model_path = (
            MODEL_DIR /
            f"{model_name}_comparison.joblib"
        )

        joblib.dump(
            result["estimator"],
            model_path
        )

        print(
            f"\nSaved {result['model']}:"
            f"\n  {model_path}"
        )

    print("\n")
    print("=" * 70)
    print("              MODEL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()