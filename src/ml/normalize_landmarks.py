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
    "data/ml_normalized"
)

NUM_FRAMES = 20
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES = 3

FEATURES_PER_HAND = (
    LANDMARKS_PER_HAND
    * COORDINATES
)

FEATURES_PER_FRAME = (
    NUM_HANDS
    * FEATURES_PER_HAND
)

EXPECTED_FEATURES = (
    NUM_FRAMES
    * FEATURES_PER_FRAME
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_hand(hand):

    """
    Normalize one hand.

    Input shape:
        (21, 3)

    Landmark 0 = wrist.

    Steps:
        1. Check whether the hand exists.
        2. Move wrist to origin.
        3. Normalize hand size using maximum
           distance from wrist.
    """

    # --------------------------------------------------------
    # If the hand is completely missing,
    # leave it unchanged.
    # --------------------------------------------------------

    if np.all(
        np.isnan(hand)
    ):
        return hand

    # --------------------------------------------------------
    # Copy so original data is not modified
    # --------------------------------------------------------

    hand = hand.copy()

    # --------------------------------------------------------
    # Wrist landmark
    # --------------------------------------------------------

    wrist = hand[0].copy()

    # --------------------------------------------------------
    # Check for invalid wrist
    # --------------------------------------------------------

    if np.any(
        np.isnan(wrist)
    ):
        return hand

    # --------------------------------------------------------
    # Translate landmarks
    #
    # wrist becomes:
    # [0, 0, 0]
    # --------------------------------------------------------

    normalized = (
        hand - wrist
    )

    # --------------------------------------------------------
    # Calculate scale
    #
    # Distance of every landmark
    # from wrist.
    # --------------------------------------------------------

    distances = np.linalg.norm(
        normalized,
        axis=1
    )

    # --------------------------------------------------------
    # Maximum hand distance
    # --------------------------------------------------------

    scale = np.nanmax(
        distances
    )

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    if (
        np.isnan(scale)
        or scale == 0
    ):
        return normalized

    # --------------------------------------------------------
    # Scale normalize
    # --------------------------------------------------------

    normalized = (
        normalized / scale
    )

    return normalized


# ============================================================
# NORMALIZE ONE VIDEO
# ============================================================

def normalize_video_features(
    features
):

    """
    Normalize one video.

    Input:
        Flat array of 2520 features.

    Structure:

        20 frames
            × 2 hands
            × 21 landmarks
            × 3 coordinates

    Shape:

        (20, 2, 21, 3)
    """

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if len(features) != EXPECTED_FEATURES:

        raise ValueError(
            "Wrong feature count!\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Found: {len(features)}"
        )

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    video = features.reshape(
        NUM_FRAMES,
        NUM_HANDS,
        LANDMARKS_PER_HAND,
        COORDINATES
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    normalized_video = np.empty_like(
        video
    )

    # --------------------------------------------------------
    # Normalize every frame
    # --------------------------------------------------------

    for frame_index in range(
        NUM_FRAMES
    ):

        for hand_index in range(
            NUM_HANDS
        ):

            hand = video[
                frame_index,
                hand_index
            ]

            normalized_hand = (
                normalize_hand(hand)
            )

            normalized_video[
                frame_index,
                hand_index
            ] = normalized_hand

    # --------------------------------------------------------
    # Flatten back to 2520 features
    # --------------------------------------------------------

    return normalized_video.reshape(
        EXPECTED_FEATURES
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "        LANDMARK NORMALIZATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{INPUT_CSV}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        f"\nLoading dataset:"
    )

    print(
        f"  {INPUT_CSV}"
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Find feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.startswith(
            "feature_"
        )
    ]

    # --------------------------------------------------------
    # Sort numerically
    # --------------------------------------------------------

    feature_columns = sorted(
        feature_columns,
        key=lambda column: int(
            column.split("_")[1]
        )
    )

    print(
        f"\nLandmark features found:"
        f" {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if (
        len(feature_columns)
        != EXPECTED_FEATURES
    ):

        raise ValueError(
            "\nFEATURE COUNT ERROR!\n"
            f"Expected: "
            f"{EXPECTED_FEATURES}\n"
            f"Found: "
            f"{len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Validate exact names
    # --------------------------------------------------------

    expected_columns = [
        f"feature_{i}"
        for i in range(
            EXPECTED_FEATURES
        )
    ]

    if (
        feature_columns
        != expected_columns
    ):

        raise ValueError(
            "\nFEATURE ORDER ERROR!\n"
            "Expected exactly:\n"
            "feature_0 through "
            "feature_2519"
        )

    print(
        f"Feature range:"
    )

    print(
        f"  First: {feature_columns[0]}"
    )

    print(
        f"  Last:  {feature_columns[-1]}"
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    print(
        "\nExtracting feature matrix..."
    )

    X = df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    print(
        f"Feature matrix: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Normalize all videos
    # --------------------------------------------------------

    print(
        "\nNormalizing landmarks..."
    )

    normalized_X = np.empty_like(
        X
    )

    total_rows = len(X)

    for index in range(
        total_rows
    ):

        normalized_X[index] = (
            normalize_video_features(
                X[index]
            )
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 50 == 0
            or index == total_rows - 1
        ):

            print(
                f"  Processed "
                f"{index + 1}"
                f"/{total_rows}"
            )

    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    print(
        "\nNormalized matrix:"
    )

    print(
        f"  Shape: "
        f"{normalized_X.shape}"
    )

    # --------------------------------------------------------
    # Save normalized feature matrix
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / "normalized_features.npy",
        normalized_X
    )

    # --------------------------------------------------------
    # Create normalized DataFrame
    #
    # IMPORTANT:
    # Keep all metadata unchanged.
    # Replace only feature values.
    # --------------------------------------------------------

    print(
        "\nCreating normalized dataset..."
    )

    metadata_columns = [
        column
        for column in df.columns
        if not column.startswith(
            "feature_"
        )
    ]

    metadata_df = df[
        metadata_columns
    ].copy()

    normalized_feature_df = (
        pd.DataFrame(
            normalized_X,
            columns=feature_columns
        )
    )

    normalized_df = pd.concat(
        [
            metadata_df.reset_index(
                drop=True
            ),
            normalized_feature_df.reset_index(
                drop=True
            ),
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Save normalized CSV
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / "normalized_landmark_dataset.csv"
    )

    normalized_df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # Save feature names
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
        / "feature_columns.csv",
        index=False
    )

    # --------------------------------------------------------
    # Verify splits
    # --------------------------------------------------------

    print(
        "\nSplit distribution:"
    )

    print(
        normalized_df[
            "split"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    print(
        f"\nNumber of classes: "
        f"{normalized_df['label'].nunique()}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "       NORMALIZATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nInput:"
    )

    print(
        f"  {INPUT_CSV}"
    )

    print(
        f"\nOutput directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nFiles created:"
    )

    print(
        "  normalized_features.npy"
    )

    print(
        "  normalized_landmark_dataset.csv"
    )

    print(
        "  feature_columns.csv"
    )

    print(
        f"\nDataset:"
    )

    print(
        f"  Videos: {len(normalized_df)}"
    )

    print(
        f"  Classes: "
        f"{normalized_df['label'].nunique()}"
    )

    print(
        f"  Features per video: "
        f"{EXPECTED_FEATURES}"
    )

    print(
        f"\nNormalization:"
    )

    print(
        "  Wrist translated to origin"
    )

    print(
        "  Hand scale normalized"
    )

    print(
        "  Each hand processed independently"
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()