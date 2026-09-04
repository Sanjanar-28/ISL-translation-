from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "data/processed/full_landmark_dataset_ml.csv"
)

OUTPUT_DIR = Path(
    "data/ml"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("                 ML DATA PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"\nTotal videos: {len(df)}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = {
        "label",
        "split",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        c
        for c in df.columns
        if c.startswith("feature_")
    ]

    print(
        f"Feature columns: {len(feature_columns)}"
    )

    if len(feature_columns) != 2520:
        raise RuntimeError(
            f"Expected 2520 features, "
            f"found {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Check for missing feature values
    # --------------------------------------------------------

    missing_features = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing feature values: "
        f"{missing_features}"
    )

    if missing_features > 0:
        raise RuntimeError(
            "Dataset contains missing feature values."
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = sorted(
        df["label"].unique()
    )

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    print(
        f"Number of classes: {len(labels)}"
    )

    if len(labels) != 49:
        raise RuntimeError(
            f"Expected 49 classes, "
            f"found {len(labels)}"
        )

    # --------------------------------------------------------
    # Convert labels to integers
    # --------------------------------------------------------

    df["label_id"] = (
        df["label"]
        .map(label_to_id)
    )

    if df["label_id"].isna().any():
        raise RuntimeError(
            "Some labels could not be mapped to label IDs."
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create X and y for each split
    # --------------------------------------------------------

    expected_sizes = {
    "train": 595,
    "validation": 127,
    "test": 130,
}

    for split in [
        "train",
        "validation",
        "test"
    ]:

        split_df = df[
            df["split"] == split
        ].copy()

        X = (
            split_df[feature_columns]
            .to_numpy(dtype=np.float32)
        )

        y = (
            split_df["label_id"]
            .to_numpy(dtype=np.int64)
        )

        print(
            f"\n{split.upper()}:"
        )

        print(
            f"  Videos: {len(X)}"
        )

        print(
            f"  X shape: {X.shape}"
        )

        print(
            f"  y shape: {y.shape}"
        )

        # ----------------------------------------------------
        # Validate shape
        # ----------------------------------------------------

        if X.shape[1] != 2520:
            raise RuntimeError(
                f"{split}: expected 2520 features, "
                f"found {X.shape[1]}"
            )

        expected_count = expected_sizes[split]

        if len(X) != expected_count:
            print(
                f"  WARNING: expected "
                f"{expected_count} videos, "
                f"found {len(X)}"
            )

        # ----------------------------------------------------
        # Save X
        # ----------------------------------------------------

        np.save(
            OUTPUT_DIR / f"X_{split}.npy",
            X
        )

        # ----------------------------------------------------
        # Save y
        # ----------------------------------------------------

        np.save(
            OUTPUT_DIR / f"y_{split}.npy",
            y
        )

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    mapping = pd.DataFrame({
        "label_id": list(id_to_label.keys()),
        "label": list(id_to_label.values())
    })

    mapping.to_csv(
        OUTPUT_DIR / "label_mapping.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print label mapping
    # --------------------------------------------------------

    print("\nLabel mapping:")

    print(
        mapping.to_string(index=False)
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                 ML DATA READY")
    print("=" * 70)

    print("\nExpected:")

    print("  Train: 595")
    print("  Validation: 127")
    print("  Test:       130")
    print("  Features:   2520")
    print("  Classes:    49")

    print("\nSaved files:")

    print("  data/ml/X_train.npy")
    print("  data/ml/y_train.npy")
    print("  data/ml/X_validation.npy")
    print("  data/ml/y_validation.npy")
    print("  data/ml/X_test.npy")
    print("  data/ml/y_test.npy")
    print("  data/ml/label_mapping.csv")

    print("\n" + "=" * 70)
    print("                 PREPARATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()