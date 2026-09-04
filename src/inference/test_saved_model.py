from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(
    "data/ml_normalized/tuned_random_forest.joblib"
)

DATASET_PATH = Path(
    "data/processed/full_landmark_dataset.csv"
)

LABEL_MAPPING_PATH = Path(
    "data/ml/label_mapping.csv"
)


def main():

    print("\n")
    print("=" * 70)
    print("             SAVED MODEL DATASET TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    feature_columns = sorted(
        feature_columns,
        key=lambda x: int(
            x.split("_")[1]
        )
    )

    print(
        f"\nDataset rows: "
        f"{len(df)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load label mapping
    # --------------------------------------------------------

    mapping = pd.read_csv(
        LABEL_MAPPING_PATH
    )

    label_map = dict(
        zip(
            mapping["label_id"],
            mapping["label"]
        )
    )

    # --------------------------------------------------------
    # Test all test rows
    # --------------------------------------------------------

    test_df = df[
        df["split"] == "test"
    ].copy()

    X = (
        test_df[
            feature_columns
        ]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # tuned_random_forest was trained on the NORMALIZED
    # features, so normalize the dataset features here in
    # exactly the same way as normalize_landmarks.py.
    #
    # However, the dataset features are already wrist-relative.
    #
    # Therefore DO NOT normalize them again.
    # --------------------------------------------------------

    print(
        f"\nTest samples: "
        f"{len(X)}"
    )

    print(
        f"Input shape: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    # --------------------------------------------------------
    # Convert numeric IDs to labels
    # --------------------------------------------------------

    predicted_labels = [
        label_map.get(
            int(prediction),
            str(prediction)
        )
        for prediction in predictions
    ]

    actual_labels = (
        test_df["label"]
        .tolist()
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    correct = sum(
        predicted == actual
        for predicted, actual
        in zip(
            predicted_labels,
            actual_labels
        )
    )

    total = len(
        actual_labels
    )

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print("\n")
    print("=" * 70)
    print("                    RESULTS")
    print("=" * 70)

    print(
        f"\nCorrect: "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Show predictions
    # --------------------------------------------------------

    print("\nFirst 20 predictions:\n")

    print(
        "Actual             Predicted"
    )

    print(
        "-" * 45
    )

    for actual, predicted in list(
        zip(
            actual_labels,
            predicted_labels
        )
    )[:20]:

        marker = (
            "OK"
            if actual == predicted
            else "WRONG"
        )

        print(
            f"{actual:<20}"
            f"{predicted:<20}"
            f"{marker}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = test_df[
        [
            "class_id",
            "label",
            "video_name",
            "split"
        ]
    ].copy()

    results[
        "predicted_label"
    ] = predicted_labels

    results[
        "correct"
    ] = (
        results["label"]
        ==
        results["predicted_label"]
    )

    output_path = Path(
        "data/ml_normalized/"
        "saved_model_dataset_test.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )

    print("\n")
    print("=" * 70)
    print("                 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()