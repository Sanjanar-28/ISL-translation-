from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

ORIGINAL_MODEL = Path(
    "data/model_comparison/best_tree_model.joblib"
)

MOTION_MODEL = Path(
    "data/ml_motion/motion_random_forest.joblib"
)

TEMPORAL_MODEL = Path(
    "data/ml_temporal/temporal_random_forest.joblib"
)

ORIGINAL_X = Path(
    "data/ml/X_test.npy"
)

MOTION_X = Path(
    "data/ml_motion/X_test.npy"
)

TEMPORAL_X = Path(
    "data/ml_temporal/X_test.npy"
)

Y_TEST = Path(
    "data/ml/y_test.npy"
)

LABEL_MAPPING = Path(
    "data/ml/label_mapping.csv"
)

OUTPUT_DIR = Path(
    "data/ml_ensemble"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 72)
    print("           ORIGINAL + MOTION + TEMPORAL ENSEMBLE")
    print("=" * 72)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading test datasets...")

    X_original = np.load(ORIGINAL_X)
    X_motion = np.load(MOTION_X)
    X_temporal = np.load(TEMPORAL_X)
    y_test = np.load(Y_TEST)

    print(f"Original: {X_original.shape}")
    print(f"Motion:   {X_motion.shape}")
    print(f"Temporal: {X_temporal.shape}")
    print(f"Labels:   {y_test.shape}")

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading models...")

    original_model = joblib.load(
        ORIGINAL_MODEL
    )

    motion_model = joblib.load(
        MOTION_MODEL
    )

    temporal_model = joblib.load(
        TEMPORAL_MODEL
    )

    # --------------------------------------------------------
    # Verify sample counts
    # --------------------------------------------------------

    sample_counts = {
        len(X_original),
        len(X_motion),
        len(X_temporal),
        len(y_test),
    }

    if len(sample_counts) != 1:

        raise ValueError(
            "Test sample counts do not match."
        )

    # --------------------------------------------------------
    # Verify feature compatibility
    # --------------------------------------------------------

    if (
        original_model.n_features_in_
        != X_original.shape[1]
    ):

        raise ValueError(
            "Original model feature mismatch."
        )

    if (
        motion_model.n_features_in_
        != X_motion.shape[1]
    ):

        raise ValueError(
            "Motion model feature mismatch."
        )

    if (
        temporal_model.n_features_in_
        != X_temporal.shape[1]
    ):

        raise ValueError(
            "Temporal model feature mismatch."
        )

    print(
        "\nFeature compatibility confirmed."
    )

    # --------------------------------------------------------
    # Verify class mappings
    # --------------------------------------------------------

    print("\nModel class IDs:")

    print(
        "Original:",
        original_model.classes_
    )

    print(
        "Motion:",
        motion_model.classes_
    )

    print(
        "Temporal:",
        temporal_model.classes_
    )

    # --------------------------------------------------------
    # Predictions and probabilities
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    original_proba = (
        original_model.predict_proba(
            X_original
        )
    )

    motion_proba = (
        motion_model.predict_proba(
            X_motion
        )
    )

    temporal_proba = (
        temporal_model.predict_proba(
            X_temporal
        )
    )

    # --------------------------------------------------------
    # Individual predictions
    # --------------------------------------------------------

    original_pred = (
        original_model.predict(
            X_original
        )
    )

    motion_pred = (
        motion_model.predict(
            X_motion
        )
    )

    temporal_pred = (
        temporal_model.predict(
            X_temporal
        )
    )

    # --------------------------------------------------------
    # Individual accuracies
    # --------------------------------------------------------

    original_acc = accuracy_score(
        y_test,
        original_pred
    )

    motion_acc = accuracy_score(
        y_test,
        motion_pred
    )

    temporal_acc = accuracy_score(
        y_test,
        temporal_pred
    )

    print("\nIndividual model accuracy:")

    print(
        f"Original: "
        f"{original_acc * 100:.2f}%"
    )

    print(
        f"Motion:   "
        f"{motion_acc * 100:.2f}%"
    )

    print(
        f"Temporal: "
        f"{temporal_acc * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Check probability dimensions
    # --------------------------------------------------------

    print("\nProbability shapes:")

    print(
        "Original:",
        original_proba.shape
    )

    print(
        "Motion:",
        motion_proba.shape
    )

    print(
        "Temporal:",
        temporal_proba.shape
    )

    if not (
        original_proba.shape
        ==
        motion_proba.shape
        ==
        temporal_proba.shape
    ):

        raise ValueError(
            "Probability shapes do not match."
        )

    # --------------------------------------------------------
    # Load label mapping
    # --------------------------------------------------------

    mapping_df = pd.read_csv(
        LABEL_MAPPING
    )

    id_to_label = dict(
        zip(
            mapping_df["label_id"],
            mapping_df["label"]
        )
    )

    # --------------------------------------------------------
    # ENSEMBLE 1
    # Original + Motion
    # --------------------------------------------------------

    print(
        "\nEvaluating Original + Motion..."
    )

    ensemble_om_proba = (
        0.70 * original_proba
        +
        0.30 * motion_proba
    )

    ensemble_om_pred = (
        np.argmax(
            ensemble_om_proba,
            axis=1
        )
    )

    ensemble_om_acc = accuracy_score(
        y_test,
        ensemble_om_pred
    )

    # --------------------------------------------------------
    # ENSEMBLE 2
    # Original + Motion + Temporal
    # --------------------------------------------------------

    print(
        "Evaluating 3-model ensemble..."
    )

    ensemble_all_proba = (

        0.60 * original_proba

        +

        0.25 * motion_proba

        +

        0.15 * temporal_proba
    )

    ensemble_all_pred = (
        np.argmax(
            ensemble_all_proba,
            axis=1
        )
    )

    ensemble_all_acc = accuracy_score(
        y_test,
        ensemble_all_pred
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print(
        "                    RESULTS"
    )
    print("=" * 72)

    print(
        f"\nOriginal RF:          "
        f"{original_acc * 100:.2f}%"
    )

    print(
        f"Motion RF:            "
        f"{motion_acc * 100:.2f}%"
    )

    print(
        f"Temporal RF:          "
        f"{temporal_acc * 100:.2f}%"
    )

    print(
        f"\nOriginal + Motion:    "
        f"{ensemble_om_acc * 100:.2f}%"
    )

    print(
        f"3-Model Ensemble:     "
        f"{ensemble_all_acc * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = pd.DataFrame(
        {
            "actual_class_id": y_test,

            "actual_label": [
                id_to_label.get(x, x)
                for x in y_test
            ],

            "original_prediction": original_pred,

            "motion_prediction": motion_pred,

            "temporal_prediction": temporal_pred,

            "ensemble_om_prediction":
                ensemble_om_pred,

            "ensemble_all_prediction":
                ensemble_all_pred,
        }
    )

    results.to_csv(
        OUTPUT_DIR
        / "ensemble_predictions.csv",

        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "model":
                    "Original RF",

                "test_accuracy":
                    original_acc,
            },

            {
                "model":
                    "Motion RF",

                "test_accuracy":
                    motion_acc,
            },

            {
                "model":
                    "Temporal RF",

                "test_accuracy":
                    temporal_acc,
            },

            {
                "model":
                    "Original + Motion",

                "test_accuracy":
                    ensemble_om_acc,
            },

            {
                "model":
                    "3-Model Ensemble",

                "test_accuracy":
                    ensemble_all_acc,
            },
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "ensemble_results.csv",

        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "  ensemble_predictions.csv"
    )

    print(
        "  ensemble_results.csv"
    )

    print("\n" + "=" * 72)
    print(
        "              ENSEMBLE EXPERIMENT COMPLETE"
    )
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()