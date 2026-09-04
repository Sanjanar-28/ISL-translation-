from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "data/ml_normalized/tuned_random_forest.joblib"
)

X_TEST_PATH = Path(
    "data/ml_normalized/X_test.npy"
)

Y_TEST_PATH = Path(
    "data/ml_normalized/y_test.npy"
)

LABEL_MAPPING_PATH = Path(
    "data/ml/label_mapping.csv"
)

OUTPUT_PATH = Path(
    "data/ml_normalized/saved_model_dataset_test.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("             SAVED MODEL DATASET TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        MODEL_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        LABEL_MAPPING_PATH
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print("\nDataset test data:")
    print("  X_test:", X_test.shape)
    print("  y_test:", y_test.shape)

    # --------------------------------------------------------
    # Load label mapping
    # --------------------------------------------------------

    mapping = pd.read_csv(LABEL_MAPPING_PATH)

    # Create:
    # class ID -> label
    id_to_label = dict(
        zip(
            mapping["label_id"].astype(int),
            mapping["label"].astype(str)
        )
    )

    print("\nNumber of classes:", len(id_to_label))

    # --------------------------------------------------------
    # Verify model classes
    # --------------------------------------------------------

    model_classes = np.asarray(model.classes_)

    print("\nModel classes:")
    print(model_classes)

    print("\nY test classes:")
    print(np.unique(y_test))

    if not np.array_equal(
        np.sort(model_classes),
        np.sort(np.unique(y_test))
    ):
        raise ValueError(
            "Model classes and y_test classes do not match."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nRunning predictions...")

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    correct = int(
        np.sum(predictions == y_test)
    )

    total = len(y_test)

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print("\n" + "=" * 70)
    print("                    RESULTS")
    print("=" * 70)

    print(
        f"\nCorrect: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    rows = []

    for i in range(total):

        actual_id = int(y_test[i])
        predicted_id = int(predictions[i])

        actual_label = id_to_label.get(
            actual_id,
            f"UNKNOWN_{actual_id}"
        )

        predicted_label = id_to_label.get(
            predicted_id,
            f"UNKNOWN_{predicted_id}"
        )

        rows.append(
            {
                "index": i,
                "actual_id": actual_id,
                "actual_label": actual_label,
                "predicted_id": predicted_id,
                "predicted_label": predicted_label,
                "correct": (
                    actual_id == predicted_id
                )
            }
        )

    results = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Print first 20
    # --------------------------------------------------------

    print("\nFirst 20 predictions:\n")

    print(
        f"{'Actual':20s}"
        f"{'Predicted':20s}"
        f"Result"
    )

    print("-" * 55)

    for i in range(
        min(20, len(results))
    ):

        row = results.iloc[i]

        result = (
            "OK"
            if row["correct"]
            else "WRONG"
        )

        print(
            f"{row['actual_label']:20s}"
            f"{row['predicted_label']:20s}"
            f"{result}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # Summary by class
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                  CLASS ACCURACY")
    print("=" * 70)

    class_summary = (
        results
        .groupby(
            ["actual_id", "actual_label"]
        )
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum")
        )
        .reset_index()
    )

    class_summary["accuracy"] = (
        class_summary["correct"]
        / class_summary["total"]
        * 100
    )

    for _, row in class_summary.iterrows():

        print(
            f"{row['actual_label']:20s}"
            f"{int(row['correct'])}/"
            f"{int(row['total'])}"
            f"  "
            f"{row['accuracy']:6.2f}%"
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                 TEST COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal accuracy: {accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()