from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "data/ml_augmented/augmented_random_forest.joblib"
)

ORIGINAL_DATA_DIR = Path(
    "data/ml"
)

OUTPUT_DIR = Path(
    "data/ml_augmented"
)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():

    mapping_path = (
        ORIGINAL_DATA_DIR
        / "label_mapping.csv"
    )

    mapping_df = pd.read_csv(
        mapping_path
    )

    print("\nLabel mapping:")
    print(
        mapping_df.head()
    )

    id_to_label = dict(

        zip(

            mapping_df["label_id"],

            mapping_df["label"]

        )

    )

    return id_to_label


# ============================================================
# EVALUATE
# ============================================================

def evaluate_split(
    model,
    X,
    y,
    split_name,
    id_to_label,
):

    print("\n" + "-" * 70)
    print(
        f"{split_name.upper()} EVALUATION"
    )
    print("-" * 70)

    print(
        f"\nSamples: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "\nPredicting..."
    )

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print(
        f"\n{split_name} accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Create prediction dataframe
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "actual_class_id": y,
            "predicted_class_id": predictions,
        }
    )

    results[
        "actual_label"
    ] = results[
        "actual_class_id"
    ].map(
        id_to_label
    )

    results[
        "predicted_label"
    ] = results[
        "predicted_class_id"
    ].map(
        id_to_label
    )

    results[
        "correct"
    ] = (
        results[
            "actual_class_id"
        ]
        ==
        results[
            "predicted_class_id"
        ]
    )

    return (
        accuracy,
        results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "        AUGMENTED RANDOM FOREST EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print(
        "\nChecking files..."
    )

    if not MODEL_PATH.exists():

        print(
            f"\nERROR: Model not found:"
            f"\n{MODEL_PATH}"
        )

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading augmented model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print(
        "\nLoading original validation data..."
    )

    X_validation = np.load(
        ORIGINAL_DATA_DIR
        / "X_validation.npy"
    )

    y_validation = np.load(
        ORIGINAL_DATA_DIR
        / "y_validation.npy"
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print(
        "Loading original test data..."
    )

    X_test = np.load(
        ORIGINAL_DATA_DIR
        / "X_test.npy"
    )

    y_test = np.load(
        ORIGINAL_DATA_DIR
        / "y_test.npy"
    )

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print(
        "\nDataset shapes:"
    )

    print(
        f"X_validation: "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation: "
        f"{y_validation.shape}"
    )

    print(
        f"X_test:       "
        f"{X_test.shape}"
    )

    print(
        f"y_test:       "
        f"{y_test.shape}"
    )

    # --------------------------------------------------------
    # Feature compatibility check
    # --------------------------------------------------------

    print(
        "\nChecking feature compatibility..."
    )

    model_features = (
        model.n_features_in_
    )

    print(
        f"Model expects: "
        f"{model_features} features"
    )

    print(
        f"Validation has: "
        f"{X_validation.shape[1]} features"
    )

    print(
        f"Test has: "
        f"{X_test.shape[1]} features"
    )

    if (
        X_validation.shape[1]
        != model_features
    ):

        print(
            "\nERROR:"
            " Validation feature count"
            " does not match model."
        )

        return

    if (
        X_test.shape[1]
        != model_features
    ):

        print(
            "\nERROR:"
            " Test feature count"
            " does not match model."
        )

        return

    print(
        "Feature compatibility confirmed."
    )

    # --------------------------------------------------------
    # Load label mapping
    # --------------------------------------------------------

    id_to_label = load_label_mapping()

    # --------------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------------

    (
        validation_accuracy,
        validation_results,
    ) = evaluate_split(
        model=model,
        X=X_validation,
        y=y_validation,
        split_name="Validation",
        id_to_label=id_to_label,
    )

    # --------------------------------------------------------
    # Evaluate test
    # --------------------------------------------------------

    (
        test_accuracy,
        test_results,
    ) = evaluate_split(
        model=model,
        X=X_test,
        y=y_test,
        split_name="Test",
        id_to_label=id_to_label,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_output = (
        OUTPUT_DIR
        / "augmented_validation_predictions.csv"
    )

    test_output = (
        OUTPUT_DIR
        / "augmented_test_predictions.csv"
    )

    validation_results.to_csv(
        validation_output,
        index=False,
    )

    test_results.to_csv(
        test_output,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "                    COMPARISON"
    )
    print("=" * 70)

    print(
        "\nOriginal Random Forest:"
    )

    print(
        "  Validation: 77.17%"
    )

    print(
        "  Test:       75.38%"
    )

    print(
        "\nAugmented Random Forest:"
    )

    print(
        f"  Validation: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"  Test:       "
        f"{test_accuracy * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print(
        "        AUGMENTED RF EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nFiles created:"
    )

    print(
        f"  {validation_output.name}"
    )

    print(
        f"  {test_output.name}"
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()