from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "data/processed/full_landmark_dataset_clean.csv"
)

OUTPUT_DIR = Path(
    "data/ml_quality"
)

MIN_DETECTION_RATE = 0.20

EXPECTED_FEATURES = 2520


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("          QUALITY-FILTERED ML DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print(f"\nOriginal rows: {len(df)}")
    print(
        f"Original columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Validate required columns
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
    # Calculate detection rate
    #
    # One frame can contain up to 2 hands.
    #
    # Example:
    # 20 left + 20 right detections
    # 100 frames
    #
    # detection_rate = 40 / 200 = 0.20
    # --------------------------------------------------------

    detection_rate = (
        df["left_detections"]
        + df["right_detections"]
    ) / (
        2 * df["original_frames"]
    )

    # Avoid repeated DataFrame insertion
    df = df.assign(
        detection_rate=detection_rate
    )

    # --------------------------------------------------------
    # Filter poor-quality samples
    # --------------------------------------------------------

    filtered = df[
        df["detection_rate"]
        >= MIN_DETECTION_RATE
    ].copy()

    print(
        f"\nFiltered rows: {len(filtered)}"
    )

    print(
        f"Minimum detection rate: "
        f"{MIN_DETECTION_RATE:.2f}"
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    labels = sorted(
        filtered["label"]
        .unique()
        .tolist()
    )

    print(
        f"\nClasses remaining: "
        f"{len(labels)}"
    )

    # --------------------------------------------------------
    # Check all original classes remain
    # --------------------------------------------------------

    original_labels = set(
        df["label"].unique()
    )

    filtered_labels = set(
        filtered["label"].unique()
    )

    missing_labels = (
        original_labels
        - filtered_labels
    )

    if missing_labels:

        print(
            "\nWARNING: "
            "Some classes were removed."
        )

        for label in sorted(
            missing_labels
        ):
            print(
                f"  {label}"
            )

    else:

        print(
            "\nAll original classes "
            "are preserved."
        )

    # --------------------------------------------------------
    # Samples per split
    # --------------------------------------------------------

    print(
        "\nSamples per split:"
    )

    print(
        filtered
        .groupby("split")
        .size()
    )

    # --------------------------------------------------------
    # Samples per class
    # --------------------------------------------------------

    print(
        "\nSamples per class:"
    )

    print(
        filtered
        .groupby("label")
        .size()
        .to_string()
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Identify EXACT landmark feature columns
    #
    # IMPORTANT:
    # Only feature_0 ... feature_2519 are used.
    #
    # Metadata such as class_id is NEVER included.
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in filtered.columns
        if column.startswith("feature_")
    ]

    # Sort numerically:
    #
    # feature_0
    # feature_1
    # ...
    # feature_2519
    #
    # instead of alphabetically where:
    # feature_100 comes before feature_2
    # --------------------------------------------------------

    feature_columns = sorted(
        feature_columns,
        key=lambda column: int(
            column.split("_")[1]
        )
    )

    print(
        f"\nLandmark feature columns found: "
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
    # Validate exact feature numbering
    # --------------------------------------------------------

    expected_feature_columns = [
        f"feature_{i}"
        for i in range(
            EXPECTED_FEATURES
        )
    ]

    if feature_columns != expected_feature_columns:

        raise ValueError(
            "\nFEATURE ORDER ERROR!\n"
            "The feature columns are not exactly:\n"
            "feature_0 ... feature_2519"
        )

    # --------------------------------------------------------
    # Save feature column names
    #
    # This is important for future inference.
    # --------------------------------------------------------

    feature_columns_df = pd.DataFrame(
        {
            "feature_index": range(
                EXPECTED_FEATURES
            ),
            "feature_name": feature_columns,
        }
    )

    feature_columns_df.to_csv(
        OUTPUT_DIR
        / "feature_columns.csv",
        index=False
    )

    # --------------------------------------------------------
    # Create label mapping
    # --------------------------------------------------------

    label_mapping = {
        label: class_id
        for class_id, label
        in enumerate(labels)
    }

    mapping_df = pd.DataFrame(
        {
            "class_id": list(
                range(len(labels))
            ),
            "label": labels,
        }
    )

    mapping_df.to_csv(
        OUTPUT_DIR
        / "label_mapping.csv",
        index=False
    )

    # --------------------------------------------------------
    # Convert each split
    # --------------------------------------------------------

    required_splits = [
        "train",
        "validation",
        "test",
    ]

    for split in required_splits:

        split_df = filtered[
            filtered["split"] == split
        ].copy()

        if len(split_df) == 0:

            raise ValueError(
                f"\nNo samples found "
                f"for split: {split}"
            )

        # ----------------------------------------------------
        # Create feature matrix
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
        # Create labels
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
                f"\nWrong feature shape "
                f"for {split}!\n"
                f"Expected: "
                f"(*, {EXPECTED_FEATURES})\n"
                f"Found: "
                f"{X.shape}"
            )

        if len(X) != len(y):

            raise ValueError(
                f"\nSample mismatch "
                f"in {split}!\n"
                f"X samples: {len(X)}\n"
                f"y samples: {len(y)}"
            )

        # ----------------------------------------------------
        # Save arrays
        # ----------------------------------------------------

        np.save(
            OUTPUT_DIR
            / f"X_{split}.npy",
            X
        )

        np.save(
            OUTPUT_DIR
            / f"y_{split}.npy",
            y
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(
            f"\n{split}:"
        )

        print(
            f"  X: {X.shape}"
        )

        print(
            f"  y: {y.shape}"
        )

        print(
            f"  Classes present: "
            f"{len(np.unique(y))}"
        )

    # --------------------------------------------------------
    # Save filtered dataset
    # --------------------------------------------------------

    filtered.to_csv(
        OUTPUT_DIR
        / "quality_filtered_dataset.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "             PREPARATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nDetection threshold: "
        f"{MIN_DETECTION_RATE}"
    )

    print(
        f"Final feature count: "
        f"{EXPECTED_FEATURES}"
    )

    print(
        f"Final class count: "
        f"{len(labels)}"
    )

    print(
        "\nFiles created:"
    )

    print(
        "  X_train.npy"
    )

    print(
        "  y_train.npy"
    )

    print(
        "  X_validation.npy"
    )

    print(
        "  y_validation.npy"
    )

    print(
        "  X_test.npy"
    )

    print(
        "  y_test.npy"
    )

    print(
        "  label_mapping.csv"
    )

    print(
        "  feature_columns.csv"
    )

    print(
        "  quality_filtered_dataset.csv"
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()