from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "data/ml_normalized/normalized_landmark_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/ml_normalized"
)

EXPECTED_FEATURES = 2520


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("          PREPARE NORMALIZED ML DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Normalized dataset not found:\n{INPUT_CSV}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading normalized dataset...")

    df = pd.read_csv(INPUT_CSV)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Validate required metadata
    # --------------------------------------------------------

    required_columns = {
        "class_id",
        "label",
        "video_name",
        "split",
        "original_frames",
        "left_detections",
        "right_detections",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Find ONLY landmark feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    # Sort numerically:
    # feature_0, feature_1, ..., feature_2519

    feature_columns = sorted(
        feature_columns,
        key=lambda column: int(
            column.split("_")[1]
        )
    )

    print(
        f"\nLandmark features found: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if len(feature_columns) != EXPECTED_FEATURES:

        raise ValueError(
            "\nFEATURE COUNT ERROR!\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Found:    {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Validate exact feature order
    # --------------------------------------------------------

    expected_columns = [
        f"feature_{i}"
        for i in range(
            EXPECTED_FEATURES
        )
    ]

    if feature_columns != expected_columns:

        raise ValueError(
            "\nFEATURE ORDER ERROR!\n"
            "Expected exactly:\n"
            "feature_0 through feature_2519"
        )

    print(
        "\nFeature range:"
    )

    print(
        f"  First: {feature_columns[0]}"
    )

    print(
        f"  Last:  {feature_columns[-1]}"
    )

    # --------------------------------------------------------
    # Check splits
    # --------------------------------------------------------

    required_splits = [
        "train",
        "validation",
        "test",
    ]

    available_splits = set(
        df["split"].unique()
    )

    missing_splits = set(
        required_splits
    ) - available_splits

    if missing_splits:

        raise ValueError(
            "Missing dataset splits:\n"
            + "\n".join(
                sorted(missing_splits)
            )
        )

    print(
        "\nSamples per split:"
    )

    print(
        df.groupby("split")
        .size()
        .reindex(required_splits)
    )

    # --------------------------------------------------------
    # Create label mapping
    #
    # Use alphabetically sorted labels to match the
    # previous experiments.
    # --------------------------------------------------------

    labels = sorted(
        df["label"]
        .unique()
        .tolist()
    )

    print(
        f"\nTotal classes: "
        f"{len(labels)}"
    )

    label_mapping = {
        label: class_id
        for class_id, label
        in enumerate(labels)
    }

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    mapping_df = pd.DataFrame(
        {
            "class_id": list(
                range(len(labels))
            ),
            "label": labels,
        }
    )

    mapping_path = (
        OUTPUT_DIR
        / "normalized_label_mapping.csv"
    )

    mapping_df.to_csv(
        mapping_path,
        index=False
    )

    # --------------------------------------------------------
    # Save feature column information
    # --------------------------------------------------------

    feature_info = pd.DataFrame(
        {
            "feature_index": range(
                EXPECTED_FEATURES
            ),
            "feature_name": feature_columns,
        }
    )

    feature_info.to_csv(
        OUTPUT_DIR
        / "normalized_feature_columns.csv",
        index=False
    )

    # --------------------------------------------------------
    # Create arrays for each split
    # --------------------------------------------------------

    for split in required_splits:

        print(
            f"\nPreparing {split}..."
        )

        split_df = df[
            df["split"] == split
        ].copy()

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        X = (
            split_df[
                feature_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        y = np.array(
            [
                label_mapping[label]
                for label
                in split_df["label"]
            ],
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Validate shapes
        # ----------------------------------------------------

        if X.shape[1] != EXPECTED_FEATURES:

            raise ValueError(
                f"\nWrong feature count "
                f"for {split}!\n"
                f"Expected: {EXPECTED_FEATURES}\n"
                f"Found: {X.shape[1]}"
            )

        if len(X) != len(y):

            raise ValueError(
                f"\nX/y sample mismatch "
                f"for {split}!\n"
                f"X: {len(X)}\n"
                f"y: {len(y)}"
            )

        # ----------------------------------------------------
        # Check for labels
        # ----------------------------------------------------

        print(
            f"  X shape: {X.shape}"
        )

        print(
            f"  y shape: {y.shape}"
        )

        print(
            f"  Classes present: "
            f"{len(np.unique(y))}"
        )

        # ----------------------------------------------------
        # Save arrays
        #
        # Unique filenames prevent overwriting
        # previous experiment files.
        # ----------------------------------------------------

        np.save(
            OUTPUT_DIR
            / f"X_{split}_normalized.npy",
            X
        )

        np.save(
            OUTPUT_DIR
            / f"y_{split}_normalized.npy",
            y
        )

    # --------------------------------------------------------
    # Save complete label mapping
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print(
        "\nReload verification..."
    )

    X_train = np.load(
        OUTPUT_DIR
        / "X_train_normalized.npy"
    )

    y_train = np.load(
        OUTPUT_DIR
        / "y_train_normalized.npy"
    )

    X_validation = np.load(
        OUTPUT_DIR
        / "X_validation_normalized.npy"
    )

    y_validation = np.load(
        OUTPUT_DIR
        / "y_validation_normalized.npy"
    )

    X_test = np.load(
        OUTPUT_DIR
        / "X_test_normalized.npy"
    )

    y_test = np.load(
        OUTPUT_DIR
        / "y_test_normalized.npy"
    )

    print(
        f"  Train:      "
        f"X={X_train.shape}, "
        f"y={y_train.shape}"
    )

    print(
        f"  Validation: "
        f"X={X_validation.shape}, "
        f"y={y_validation.shape}"
    )

    print(
        f"  Test:       "
        f"X={X_test.shape}, "
        f"y={y_test.shape}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "        NORMALIZED DATA PREPARATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nFinal dataset:"
    )

    print(
        f"  Features per video: "
        f"{EXPECTED_FEATURES}"
    )

    print(
        f"  Total classes: "
        f"{len(labels)}"
    )

    print(
        "\nFiles created:"
    )

    print(
        "  X_train_normalized.npy"
    )

    print(
        "  y_train_normalized.npy"
    )

    print(
        "  X_validation_normalized.npy"
    )

    print(
        "  y_validation_normalized.npy"
    )

    print(
        "  X_test_normalized.npy"
    )

    print(
        "  y_test_normalized.npy"
    )

    print(
        "  normalized_label_mapping.csv"
    )

    print(
        "  normalized_feature_columns.csv"
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()