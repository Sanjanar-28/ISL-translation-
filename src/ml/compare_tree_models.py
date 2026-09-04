from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/ml")
RESULTS_DIR = Path("data/model_comparison")

RANDOM_STATE = 42
N_JOBS = -1


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """
    Load the ORIGINAL (non-normalized, non-quality-filtered)
    dataset.
    """

    print("\nLoading original dataset...")

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    )

    y_validation = np.load(
        DATA_DIR / "y_validation.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
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
# MODEL CONFIGURATIONS
# ============================================================

def get_models():
    """
    Return different Random Forest and Extra Trees
    configurations to compare.
    """

    models = {

        # ----------------------------------------------------
        # RANDOM FOREST EXPERIMENTS
        # ----------------------------------------------------

        "RF_300_sqrt": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_sqrt": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_800_sqrt": RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_log2": RandomForestClassifier(
            n_estimators=500,
            max_features="log2",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_30pct": RandomForestClassifier(
            n_estimators=500,
            max_features=0.30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_depth20": RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_depth30": RandomForestClassifier(
            n_estimators=500,
            max_depth=30,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_leaf2": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_leaf3": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "RF_500_no_weight": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        # ----------------------------------------------------
        # EXTRA TREES EXPERIMENTS
        # ----------------------------------------------------

        "ET_300_sqrt": ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "ET_500_sqrt": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "ET_800_sqrt": ExtraTreesClassifier(
            n_estimators=800,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "ET_500_log2": ExtraTreesClassifier(
            n_estimators=500,
            max_features="log2",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "ET_500_30pct": ExtraTreesClassifier(
            n_estimators=500,
            max_features=0.30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "ET_500_depth30": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=30,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),

        "ET_500_leaf2": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
    }

    return models


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    print("\n" + "=" * 72)
    print("        ORIGINAL DATASET: TREE MODEL COMPARISON")
    print("=" * 72)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_dataset()

    print("\nDataset:")

    print(f"  X_train:      {X_train.shape}")
    print(f"  X_validation: {X_validation.shape}")
    print(f"  X_test:       {X_test.shape}")

    print(
        f"\nNumber of training classes: "
        f"{len(np.unique(y_train))}"
    )

    # --------------------------------------------------------
    # GET MODELS
    # --------------------------------------------------------

    models = get_models()

    print(
        f"\nModels to compare: {len(models)}"
    )

    results = []

    best_model = None
    best_name = None
    best_validation_accuracy = -1.0

    # --------------------------------------------------------
    # TRAIN AND EVALUATE
    # --------------------------------------------------------

    for index, (name, model) in enumerate(
        models.items(),
        start=1,
    ):

        print("\n" + "-" * 72)

        print(
            f"[{index}/{len(models)}] "
            f"Training: {name}"
        )

        # Train
        model.fit(
            X_train,
            y_train,
        )

        # Validation predictions
        validation_predictions = model.predict(
            X_validation
        )

        validation_accuracy = accuracy_score(
            y_validation,
            validation_predictions,
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        # Store result
        results.append(
            {
                "model": name,
                "validation_accuracy": validation_accuracy,
            }
        )

        # Track best model
        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            best_model = model

            best_name = name

            print(
                ">>> NEW BEST MODEL"
            )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="validation_accuracy",
        ascending=False,
    ).reset_index(
        drop=True
    )

    results_df["validation_accuracy_percent"] = (
        results_df["validation_accuracy"] * 100
    )

    print("\n" + "=" * 72)
    print("              VALIDATION RANKING")
    print("=" * 72)

    print(
        results_df[
            [
                "model",
                "validation_accuracy_percent",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE VALIDATION RESULTS
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_DIR
        / "tree_model_validation_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # TEST ONLY THE BEST MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("                 BEST MODEL")
    print("=" * 72)

    print(
        f"\nBest model: {best_name}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "\nEvaluating BEST validation model "
        "on the test set..."
    )

    test_predictions = best_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    print(
        f"\nBest model test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    best_model_path = (
        RESULTS_DIR
        / "best_tree_model.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path,
    )

    print(
        f"\nBest model saved to:"
    )

    print(
        f"  {best_model_path}"
    )

    # --------------------------------------------------------
    # SAVE FINAL SUMMARY
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "best_model": best_name,
                "best_validation_accuracy": (
                    best_validation_accuracy
                ),
                "test_accuracy": test_accuracy,
            }
        ]
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "best_tree_model_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BASELINE COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("               BASELINE COMPARISON")
    print("=" * 72)

    print(
        "\nOriginal tuned RF baseline:"
    )

    print(
        "  Validation: 77.17%"
    )

    print(
        "  Test:       75.38%"
    )

    print(
        "\nBest tree experiment:"
    )

    print(
        f"  Model:      {best_name}"
    )

    print(
        f"  Validation: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        f"  Test:       "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("           TREE MODEL COMPARISON COMPLETE")
    print("=" * 72)

    print(
        "\nFiles created:"
    )

    print(
        "  tree_model_validation_results.csv"
    )

    print(
        "  best_tree_model.joblib"
    )

    print(
        "  best_tree_model_summary.csv"
    )

    print("\n" + "=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()