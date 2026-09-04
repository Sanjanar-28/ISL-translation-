from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIG
# ============================================================

# Original Random Forest
ORIGINAL_MODEL = Path(
    "data/ml/tuned_random_forest.joblib"
)

# If your original model has a different filename,
# change ONLY the path above.

ORIGINAL_X_TRAIN = Path(
    "data/ml/X_train.npy"
)

ORIGINAL_X_VALIDATION = Path(
    "data/ml/X_validation.npy"
)

ORIGINAL_Y_VALIDATION = Path(
    "data/ml/y_validation.npy"
)

ORIGINAL_X_TEST = Path(
    "data/ml/X_test.npy"
)

ORIGINAL_Y_TEST = Path(
    "data/ml/y_test.npy"
)


# Motion Random Forest
MOTION_MODEL = Path(
    "data/ml_motion/motion_random_forest.joblib"
)

MOTION_X_VALIDATION = Path(
    "data/ml_motion/X_validation.npy"
)

MOTION_X_TEST = Path(
    "data/ml_motion/X_test.npy"
)


# Temporal Random Forest
TEMPORAL_MODEL = Path(
    "data/ml_temporal/temporal_random_forest.joblib"
)

TEMPORAL_X_VALIDATION = Path(
    "data/ml_temporal/X_validation.npy"
)

TEMPORAL_X_TEST = Path(
    "data/ml_temporal/X_test.npy"
)


# Output
OUTPUT_DIR = Path(
    "data/ensemble"
)

# Weight search resolution
WEIGHT_STEP = 0.05


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


def check_all_files():

    required_files = [

        ORIGINAL_MODEL,

        ORIGINAL_X_VALIDATION,
        ORIGINAL_Y_VALIDATION,
        ORIGINAL_X_TEST,
        ORIGINAL_Y_TEST,

        MOTION_MODEL,
        MOTION_X_VALIDATION,
        MOTION_X_TEST,

        TEMPORAL_MODEL,
        TEMPORAL_X_VALIDATION,
        TEMPORAL_X_TEST,
    ]

    for file_path in required_files:

        check_file(
            file_path
        )


def validate_alignment(
    name,
    X,
    expected_samples,
    expected_features=None
):

    if len(X) != expected_samples:

        raise ValueError(
            f"\n{name} sample mismatch!\n"
            f"Expected samples: {expected_samples}\n"
            f"Actual samples:   {len(X)}\n"
        )

    if expected_features is not None:

        if X.shape[1] != expected_features:

            raise ValueError(
                f"\n{name} feature mismatch!\n"
                f"Expected features: "
                f"{expected_features}\n"
                f"Actual features:   "
                f"{X.shape[1]}\n"
            )


def get_aligned_probabilities(
    model,
    X,
    num_classes,
    model_name
):

    print(
        f"Getting probabilities: "
        f"{model_name}..."
    )

    raw_probabilities = (
        model.predict_proba(X)
    )

    model_classes = np.asarray(
        model.classes_
    )

    aligned = np.zeros(
        (
            len(X),
            num_classes
        ),
        dtype=np.float64
    )

    aligned[
        :,
        model_classes
    ] = raw_probabilities

    return aligned


def generate_weight_combinations(
    step
):

    combinations = []

    values = np.arange(
        0.0,
        1.0 + step / 2,
        step
    )

    for original_weight in values:

        for motion_weight in values:

            temporal_weight = (
                1.0
                - original_weight
                - motion_weight
            )

            if temporal_weight < -1e-9:

                continue

            if temporal_weight > 1.0 + 1e-9:

                continue

            if temporal_weight < 0:

                temporal_weight = 0.0

            total = (
                original_weight
                + motion_weight
                + temporal_weight
            )

            if total <= 0:

                continue

            original_weight = (
                original_weight
                / total
            )

            motion_weight = (
                motion_weight
                / total
            )

            temporal_weight = (
                temporal_weight
                / total
            )

            combinations.append(

                (
                    original_weight,
                    motion_weight,
                    temporal_weight,
                )
            )

    return combinations


def evaluate_weights(
    original_probabilities,
    motion_probabilities,
    temporal_probabilities,
    y_true,
    weights
):

    (
        original_weight,
        motion_weight,
        temporal_weight,
    ) = weights

    ensemble_probabilities = (

        original_weight
        * original_probabilities

        +

        motion_weight
        * motion_probabilities

        +

        temporal_weight
        * temporal_probabilities
    )

    predictions = np.argmax(
        ensemble_probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    return (
        accuracy,
        predictions,
        ensemble_probabilities
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 72)

    print(
        "          WEIGHTED ENSEMBLE TUNING"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print(
        "\nChecking required files..."
    )

    check_all_files()

    print(
        "All required files found."
    )


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print(
        "\nLoading models..."
    )

    original_model = joblib.load(
        ORIGINAL_MODEL
    )

    motion_model = joblib.load(
        MOTION_MODEL
    )

    temporal_model = joblib.load(
        TEMPORAL_MODEL
    )

    print(
        "Models loaded."
    )


    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print(
        "\nLoading validation datasets..."
    )

    X_original_validation = np.load(
        ORIGINAL_X_VALIDATION
    )

    y_validation = np.load(
        ORIGINAL_Y_VALIDATION
    )

    X_motion_validation = np.load(
        MOTION_X_VALIDATION
    )

    X_temporal_validation = np.load(
        TEMPORAL_X_VALIDATION
    )


    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print(
        "Loading test datasets..."
    )

    X_original_test = np.load(
        ORIGINAL_X_TEST
    )

    y_test = np.load(
        ORIGINAL_Y_TEST
    )

    X_motion_test = np.load(
        MOTION_X_TEST
    )

    X_temporal_test = np.load(
        TEMPORAL_X_TEST
    )


    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print()

    print(
        "Validation data:"
    )

    print(
        f"  Original: "
        f"{X_original_validation.shape}"
    )

    print(
        f"  Motion:   "
        f"{X_motion_validation.shape}"
    )

    print(
        f"  Temporal: "
        f"{X_temporal_validation.shape}"
    )

    print(
        f"  Labels:   "
        f"{y_validation.shape}"
    )


    print()

    print(
        "Test data:"
    )

    print(
        f"  Original: "
        f"{X_original_test.shape}"
    )

    print(
        f"  Motion:   "
        f"{X_motion_test.shape}"
    )

    print(
        f"  Temporal: "
        f"{X_temporal_test.shape}"
    )

    print(
        f"  Labels:   "
        f"{y_test.shape}"
    )


    # --------------------------------------------------------
    # Check sample alignment
    # --------------------------------------------------------

    print(
        "\nChecking sample alignment..."
    )

    validation_samples = len(
        y_validation
    )

    test_samples = len(
        y_test
    )


    validate_alignment(

        "Original validation",

        X_original_validation,

        validation_samples,

        original_model.n_features_in_
    )


    validate_alignment(

        "Motion validation",

        X_motion_validation,

        validation_samples,

        motion_model.n_features_in_
    )


    validate_alignment(

        "Temporal validation",

        X_temporal_validation,

        validation_samples,

        temporal_model.n_features_in_
    )


    validate_alignment(

        "Original test",

        X_original_test,

        test_samples,

        original_model.n_features_in_
    )


    validate_alignment(

        "Motion test",

        X_motion_test,

        test_samples,

        motion_model.n_features_in_
    )


    validate_alignment(

        "Temporal test",

        X_temporal_test,

        test_samples,

        temporal_model.n_features_in_
    )


    print(
        "Sample and feature alignment confirmed."
    )


    # --------------------------------------------------------
    # Determine number of classes
    # --------------------------------------------------------

    all_class_ids = np.concatenate(

        [

            original_model.classes_,

            motion_model.classes_,

            temporal_model.classes_,

            y_validation,

            y_test,
        ]
    )

    num_classes = (
        int(
            np.max(all_class_ids)
        )
        + 1
    )

    print(
        f"\nNumber of classes: "
        f"{num_classes}"
    )


    # --------------------------------------------------------
    # Get validation probabilities
    # --------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "          VALIDATION PREDICTIONS"
    )

    print(
        "=" * 72
    )


    original_validation_probabilities = (
        get_aligned_probabilities(

            original_model,

            X_original_validation,

            num_classes,

            "Original RF"
        )
    )


    motion_validation_probabilities = (
        get_aligned_probabilities(

            motion_model,

            X_motion_validation,

            num_classes,

            "Motion RF"
        )
    )


    temporal_validation_probabilities = (
        get_aligned_probabilities(

            temporal_model,

            X_temporal_validation,

            num_classes,

            "Temporal RF"
        )
    )


    # --------------------------------------------------------
    # Individual validation accuracies
    # --------------------------------------------------------

    original_validation_predictions = (
        np.argmax(
            original_validation_probabilities,
            axis=1
        )
    )

    motion_validation_predictions = (
        np.argmax(
            motion_validation_probabilities,
            axis=1
        )
    )

    temporal_validation_predictions = (
        np.argmax(
            temporal_validation_probabilities,
            axis=1
        )
    )


    original_validation_accuracy = (
        accuracy_score(

            y_validation,

            original_validation_predictions
        )
    )


    motion_validation_accuracy = (
        accuracy_score(

            y_validation,

            motion_validation_predictions
        )
    )


    temporal_validation_accuracy = (
        accuracy_score(

            y_validation,

            temporal_validation_predictions
        )
    )


    print()

    print(
        "Individual validation accuracy:"
    )

    print(
        f"  Original RF: "
        f"{original_validation_accuracy * 100:.2f}%"
    )

    print(
        f"  Motion RF:   "
        f"{motion_validation_accuracy * 100:.2f}%"
    )

    print(
        f"  Temporal RF: "
        f"{temporal_validation_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Equal-weight validation baseline
    # --------------------------------------------------------

    equal_weights = (

        1 / 3,

        1 / 3,

        1 / 3
    )


    (
        equal_validation_accuracy,

        _,

        _,

    ) = evaluate_weights(

        original_validation_probabilities,

        motion_validation_probabilities,

        temporal_validation_probabilities,

        y_validation,

        equal_weights
    )


    print()

    print(
        "Equal-weight ensemble:"
    )

    print(
        f"  Original: "
        f"{equal_weights[0]:.3f}"
    )

    print(
        f"  Motion:   "
        f"{equal_weights[1]:.3f}"
    )

    print(
        f"  Temporal: "
        f"{equal_weights[2]:.3f}"
    )

    print(
        f"  Validation accuracy: "
        f"{equal_validation_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Generate weight combinations
    # --------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "          SEARCHING WEIGHT COMBINATIONS"
    )

    print(
        "=" * 72
    )


    weight_combinations = (
        generate_weight_combinations(
            WEIGHT_STEP
        )
    )


    print()

    print(
        f"Weight step: "
        f"{WEIGHT_STEP}"
    )

    print(
        f"Combinations: "
        f"{len(weight_combinations)}"
    )


    # --------------------------------------------------------
    # Evaluate validation weights
    # --------------------------------------------------------

    results = []

    best_accuracy = -1.0

    best_weights = None


    for weights in weight_combinations:


        accuracy, _, _ = (

            evaluate_weights(

                original_validation_probabilities,

                motion_validation_probabilities,

                temporal_validation_probabilities,

                y_validation,

                weights
            )
        )


        (
            original_weight,

            motion_weight,

            temporal_weight,
        ) = weights


        results.append(

            {

                "original_weight":
                    original_weight,

                "motion_weight":
                    motion_weight,

                "temporal_weight":
                    temporal_weight,

                "validation_accuracy":
                    accuracy,
            }
        )


        if accuracy > best_accuracy:


            best_accuracy = accuracy

            best_weights = weights


    # --------------------------------------------------------
    # Save all validation results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    results_df = results_df.sort_values(

        by="validation_accuracy",

        ascending=False

    ).reset_index(
        drop=True
    )


    results_path = (

        OUTPUT_DIR

        /

        "ensemble_weight_validation_results.csv"
    )


    results_df.to_csv(

        results_path,

        index=False
    )


    # --------------------------------------------------------
    # Show top 10 weights
    # --------------------------------------------------------

    print()

    print(
        "TOP 10 WEIGHT COMBINATIONS"
    )

    print(
        "-" * 72
    )


    print(

        results_df.head(10).to_string(

            index=False,

            formatters={

                "original_weight":
                    "{:.3f}".format,

                "motion_weight":
                    "{:.3f}".format,

                "temporal_weight":
                    "{:.3f}".format,

                "validation_accuracy":
                    "{:.4f}".format,
            }
        )
    )


    # --------------------------------------------------------
    # Best validation result
    # --------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "          BEST VALIDATION WEIGHTS"
    )

    print(
        "=" * 72
    )


    print()

    print(
        f"Original RF weight: "
        f"{best_weights[0]:.3f}"
    )

    print(
        f"Motion RF weight:   "
        f"{best_weights[1]:.3f}"
    )

    print(
        f"Temporal RF weight: "
        f"{best_weights[2]:.3f}"
    )

    print()

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Get test probabilities
    # --------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "          FINAL TEST EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        "\nUsing the BEST validation weights..."
    )

    print(
        "\nGetting test probabilities..."
    )


    original_test_probabilities = (
        get_aligned_probabilities(

            original_model,

            X_original_test,

            num_classes,

            "Original RF"
        )
    )


    motion_test_probabilities = (
        get_aligned_probabilities(

            motion_model,

            X_motion_test,

            num_classes,

            "Motion RF"
        )
    )


    temporal_test_probabilities = (
        get_aligned_probabilities(

            temporal_model,

            X_temporal_test,

            num_classes,

            "Temporal RF"
        )
    )


    # --------------------------------------------------------
    # Evaluate best weights on test
    # --------------------------------------------------------

    (

        best_test_accuracy,

        best_test_predictions,

        best_test_probabilities,

    ) = evaluate_weights(

        original_test_probabilities,

        motion_test_probabilities,

        temporal_test_probabilities,

        y_test,

        best_weights
    )


    # --------------------------------------------------------
    # Equal-weight test baseline
    # --------------------------------------------------------

    (

        equal_test_accuracy,

        equal_test_predictions,

        _,

    ) = evaluate_weights(

        original_test_probabilities,

        motion_test_probabilities,

        temporal_test_probabilities,

        y_test,

        equal_weights
    )


    # --------------------------------------------------------
    # Individual test accuracies
    # --------------------------------------------------------

    original_test_predictions = (
        np.argmax(
            original_test_probabilities,
            axis=1
        )
    )

    motion_test_predictions = (
        np.argmax(
            motion_test_probabilities,
            axis=1
        )
    )

    temporal_test_predictions = (
        np.argmax(
            temporal_test_probabilities,
            axis=1
        )
    )


    original_test_accuracy = (
        accuracy_score(

            y_test,

            original_test_predictions
        )
    )


    motion_test_accuracy = (
        accuracy_score(

            y_test,

            motion_test_predictions
        )
    )


    temporal_test_accuracy = (
        accuracy_score(

            y_test,

            temporal_test_predictions
        )
    )


    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()

    print(
        "TEST RESULTS"
    )

    print(
        "-" * 72
    )


    print(
        f"Original RF: "
        f"{original_test_accuracy * 100:.2f}%"
    )

    print(
        f"Motion RF:   "
        f"{motion_test_accuracy * 100:.2f}%"
    )

    print(
        f"Temporal RF: "
        f"{temporal_test_accuracy * 100:.2f}%"
    )

    print()

    print(
        f"Equal-weight ensemble: "
        f"{equal_test_accuracy * 100:.2f}%"
    )

    print(
        f"Best weighted ensemble: "
        f"{best_test_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(

        {

            "actual_class_id":

                y_test,

            "original_prediction":

                original_test_predictions,

            "motion_prediction":

                motion_test_predictions,

            "temporal_prediction":

                temporal_test_predictions,

            "equal_weight_prediction":

                equal_test_predictions,

            "weighted_prediction":

                best_test_predictions,
        }
    )


    prediction_path = (

        OUTPUT_DIR

        /

        "weighted_ensemble_test_predictions.csv"
    )


    prediction_df.to_csv(

        prediction_path,

        index=False
    )


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(

        [

            {

                "original_weight":

                    best_weights[0],

                "motion_weight":

                    best_weights[1],

                "temporal_weight":

                    best_weights[2],

                "original_validation_accuracy":

                    original_validation_accuracy,

                "motion_validation_accuracy":

                    motion_validation_accuracy,

                "temporal_validation_accuracy":

                    temporal_validation_accuracy,

                "equal_validation_accuracy":

                    equal_validation_accuracy,

                "best_validation_accuracy":

                    best_accuracy,

                "original_test_accuracy":

                    original_test_accuracy,

                "motion_test_accuracy":

                    motion_test_accuracy,

                "temporal_test_accuracy":

                    temporal_test_accuracy,

                "equal_test_accuracy":

                    equal_test_accuracy,

                "weighted_test_accuracy":

                    best_test_accuracy,
            }
        ]
    )


    summary_path = (

        OUTPUT_DIR

        /

        "best_weighted_ensemble_summary.csv"
    )


    summary_df.to_csv(

        summary_path,

        index=False
    )


    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "                    FINAL COMPARISON"
    )

    print(
        "=" * 72
    )


    print()

    print(
        "Individual models:"
    )

    print(
        f"  Original RF: "
        f"{original_test_accuracy * 100:.2f}%"
    )

    print(
        f"  Motion RF:   "
        f"{motion_test_accuracy * 100:.2f}%"
    )

    print(
        f"  Temporal RF: "
        f"{temporal_test_accuracy * 100:.2f}%"
    )


    print()

    print(
        "Ensemble:"
    )

    print(
        f"  Equal weights:    "
        f"{equal_test_accuracy * 100:.2f}%"
    )

    print(
        f"  Weighted ensemble:"
        f" {best_test_accuracy * 100:.2f}%"
    )


    print()

    print(
        "Best weights:"
    )

    print(
        f"  Original RF: "
        f"{best_weights[0]:.3f}"
    )

    print(
        f"  Motion RF:   "
        f"{best_weights[1]:.3f}"
    )

    print(
        f"  Temporal RF: "
        f"{best_weights[2]:.3f}"
    )


    print()

    print(
        f"Improvement over equal ensemble: "
        f"{(best_test_accuracy - equal_test_accuracy) * 100:+.2f}%"
    )

    print(
        f"Improvement over Original RF: "
        f"{(best_test_accuracy - original_test_accuracy) * 100:+.2f}%"
    )


    print()

    print(
        "=" * 72
    )

    print(
        "          WEIGHTED ENSEMBLE COMPLETE"
    )

    print(
        "=" * 72
    )


    print()

    print(
        "Files created:"
    )

    print(
        f"  {results_path}"
    )

    print(
        f"  {prediction_path}"
    )

    print(
        f"  {summary_path}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()