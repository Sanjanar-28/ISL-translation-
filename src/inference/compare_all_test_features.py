from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"D:\Amrita SLR Dataset\sign project"
)

METADATA_PATH = Path(
    "data/video_metadata_split.csv"
)

DATASET_FEATURES_PATH = Path(
    "data/processed/full_landmark_dataset.csv"
)

OUTPUT_PATH = Path(
    "data/ml_normalized/all_test_feature_comparison.csv"
)

MODEL_PATH = Path(
    "models/hand_landmarker.task"
)

NUM_SAMPLED_FRAMES = 20

FEATURES_PER_FRAME = 126

EXPECTED_FEATURES = (
    NUM_SAMPLED_FRAMES *
    FEATURES_PER_FRAME
)


# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def empty_hand():
    """
    Return 63 missing values.

    21 landmarks × 3 coordinates = 63.
    """

    return [np.nan] * 63


# ------------------------------------------------------------
# EXTRACT 21 LANDMARKS
# ------------------------------------------------------------

def extract_hand(landmarks):
    """
    Extract x, y, z for all 21 landmarks.
    """

    features = []

    for landmark in landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features


# ------------------------------------------------------------
# WRIST RELATIVE NORMALIZATION
# ------------------------------------------------------------

def normalize_hand(features):
    """
    Make all landmarks relative to wrist.

    Landmark 0 = wrist.

    x' = x - wrist_x
    y' = y - wrist_y
    z' = z - wrist_z
    """

    if all(
        pd.isna(value)
        for value in features
    ):
        return features

    wrist_x = features[0]
    wrist_y = features[1]
    wrist_z = features[2]

    normalized = []

    for i in range(21):

        x = features[
            i * 3
        ]

        y = features[
            i * 3 + 1
        ]

        z = features[
            i * 3 + 2
        ]

        if pd.isna(x):

            normalized.extend([
                np.nan,
                np.nan,
                np.nan
            ])

        else:

            normalized.extend([
                x - wrist_x,
                y - wrist_y,
                z - wrist_z
            ])

    return normalized


# ------------------------------------------------------------
# TEMPORAL INTERPOLATION
# ------------------------------------------------------------

def interpolate_features(features):
    """
    Fill missing values using temporal interpolation.
    """

    array = np.array(
        features,
        dtype=float
    )

    for column in range(
        array.shape[1]
    ):

        series = pd.Series(
            array[:, column]
        )

        series = (
            series
            .interpolate(
                method="linear",
                limit_direction="both"
            )
        )

        array[:, column] = (
            series.values
        )

    return array


# ------------------------------------------------------------
# PROCESS ONE VIDEO
# ------------------------------------------------------------

def process_video(
    video_path,
    landmarker
):
    """
    Convert one video into exactly
    2520 features.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    frames = []

    left_detected = []

    right_detected = []

    frame_number = 0

    # ========================================================
    # READ ALL VIDEO FRAMES
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Use mp.Image
        #
        # NOT:
        # vision.Image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # TIMESTAMP
        #
        # Starts from 0 for each video.
        # ----------------------------------------------------

        timestamp_ms = int(
            ((frame_number - 1) / fps)
            * 1000
        )

        # ----------------------------------------------------
        # MEDIA PIPE VIDEO INFERENCE
        # ----------------------------------------------------

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )

        # ----------------------------------------------------
        # DEFAULT EMPTY HANDS
        # ----------------------------------------------------

        left = empty_hand()

        right = empty_hand()

        left_found = False

        right_found = False

        # ----------------------------------------------------
        # PROCESS DETECTED HANDS
        # ----------------------------------------------------

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = (
                result.handedness[
                    hand_index
                ][0]
            )

            hand_label = (
                handedness.category_name
            )

            hand_features = (
                extract_hand(
                    landmarks
                )
            )

            if hand_label == "Left":

                left = hand_features

                left_found = True

            elif hand_label == "Right":

                right = hand_features

                right_found = True

        # ----------------------------------------------------
        # WRIST NORMALIZATION
        # ----------------------------------------------------

        left = normalize_hand(
            left
        )

        right = normalize_hand(
            right
        )

        # ----------------------------------------------------
        # COMBINE:
        #
        # Left  = 63
        # Right = 63
        #
        # Total = 126
        # ----------------------------------------------------

        frame_features = (
            left + right
        )

        if len(frame_features) != 126:

            raise RuntimeError(
                "Frame feature size is not 126"
            )

        frames.append(
            frame_features
        )

        left_detected.append(
            left_found
        )

        right_detected.append(
            right_found
        )

    cap.release()

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(frames) == 0:

        raise RuntimeError(
            f"No frames extracted from: "
            f"{video_path}"
        )

    # ========================================================
    # NUMPY
    # ========================================================

    features = np.array(
        frames,
        dtype=float
    )

    # ========================================================
    # INTERPOLATION
    # ========================================================

    features = interpolate_features(
        features
    )

    # ========================================================
    # TEMPORAL SAMPLING
    # ========================================================

    total_frames = len(
        features
    )

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_SAMPLED_FRAMES,
        dtype=int
    )

    sampled = features[
        indices
    ]

    # ========================================================
    # CHECK SHAPE
    # ========================================================

    expected_shape = (
        NUM_SAMPLED_FRAMES,
        FEATURES_PER_FRAME
    )

    if sampled.shape != expected_shape:

        raise RuntimeError(
            f"Unexpected sampled shape: "
            f"{sampled.shape}; "
            f"expected {expected_shape}"
        )

    # ========================================================
    # FLATTEN
    # ========================================================

    flattened = sampled.flatten()

    # ========================================================
    # CHECK FINAL SIZE
    # ========================================================

    if len(flattened) != EXPECTED_FEATURES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_FEATURES} features, "
            f"got {len(flattened)}"
        )

    return (
        flattened.astype(np.float32),
        total_frames,
        sum(left_detected),
        sum(right_detected)
    )


# ============================================================
# LOAD DATASET FEATURES
# ============================================================

def load_dataset_features():
    """
    Load the original processed landmark dataset.
    """

    print(
        "\nLoading dataset features..."
    )

    if not DATASET_FEATURES_PATH.exists():

        raise FileNotFoundError(
            "Dataset feature file not found:\n"
            f"{DATASET_FEATURES_PATH}"
        )

    df = pd.read_csv(
        DATASET_FEATURES_PATH
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    # Sort numerically.
    feature_columns = sorted(
        feature_columns,
        key=lambda x: int(
            x.split("_")[1]
        )
    )

    print(
        f"Dataset feature columns: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != EXPECTED_FEATURES:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_FEATURES} dataset "
            f"features but found "
            f"{len(feature_columns)}"
        )

    return df, feature_columns


# ============================================================
# FIND VIDEO PATH
# ============================================================

def find_video_path(row):
    """
    Construct video path from metadata.

    Dataset structure:

    sign project/
        1.bird/
        2.black/
        ...
    """

    class_id = int(
        row["class_id"]
    )

    label = str(
        row["label"]
    )

    video_name = str(
        row["video_name"]
    )

    video_path = (
        DATASET_ROOT
        / f"{class_id}.{label}"
        / video_name
    )

    return video_path


# ============================================================
# COMPARE FEATURES
# ============================================================

def compare_features(
    extracted,
    dataset_features
):
    """
    Compare extracted video features
    with dataset features.
    """

    extracted = np.asarray(
        extracted,
        dtype=np.float32
    )

    dataset_features = np.asarray(
        dataset_features,
        dtype=np.float32
    )

    if extracted.shape != dataset_features.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"{extracted.shape} vs "
            f"{dataset_features.shape}"
        )

    difference = np.abs(
        extracted -
        dataset_features
    )

    mae = float(
        np.mean(difference)
    )

    maximum = float(
        np.max(difference)
    )

    # --------------------------------------------------------
    # Exact match
    #
    # Small floating point tolerance.
    # --------------------------------------------------------

    matching = np.isclose(
        extracted,
        dataset_features,
        atol=1e-6,
        rtol=1e-6
    )

    matching_percentage = (
        np.mean(matching)
        * 100.0
    )

    exact_match = bool(
        np.all(matching)
    )

    return (
        mae,
        maximum,
        matching_percentage,
        exact_match
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "          ALL TEST VIDEOS FEATURE COMPARISON"
    )
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n"
            f"{METADATA_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"MediaPipe model not found:\n"
            f"{MODEL_PATH}"
        )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    print(
        "\nLoading metadata..."
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    required_columns = {
        "class_id",
        "label",
        "video_name",
        "split"
    }

    missing = (
        required_columns
        - set(metadata.columns)
    )

    if missing:

        raise ValueError(
            "Metadata is missing columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # ONLY TEST VIDEOS
    # --------------------------------------------------------

    test_metadata = metadata[
        metadata["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    test_metadata = (
        test_metadata
        .reset_index(drop=True)
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    print(
        f"Test videos: {len(test_metadata)}"
    )

    if len(test_metadata) == 0:

        raise RuntimeError(
            "No test videos found."
        )

    # ========================================================
    # LOAD DATASET FEATURES
    # ========================================================

    dataset_df, feature_columns = (
        load_dataset_features()
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    successful = 0

    failed = 0

    exact_matches = 0

    mismatches = 0

    # ========================================================
    # PROCESS EACH TEST VIDEO
    # ========================================================

    for index, row in test_metadata.iterrows():

        label = str(
            row["label"]
        )

        video_name = str(
            row["video_name"]
        )

        print("\n" + "-" * 70)

        print(
            f"[{index + 1}/{len(test_metadata)}] "
            f"{label}/{video_name}"
        )

        # ----------------------------------------------------
        # VIDEO PATH
        # ----------------------------------------------------

        video_path = find_video_path(
            row
        )

        print(
            f"  Path: {video_path}"
        )

        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if not video_path.exists():

            error_message = (
                "Video file not found"
            )

            print(
                f"  ERROR: {error_message}"
            )

            results.append({

                "class_id":
                    row["class_id"],

                "label":
                    label,

                "video_name":
                    video_name,

                "split":
                    row["split"],

                "video_path":
                    str(video_path),

                "status":
                    "FAILED",

                "error":
                    error_message,

                "mae":
                    np.nan,

                "max_difference":
                    np.nan,

                "matching_percentage":
                    np.nan,

                "exact_match":
                    False,

                "frames":
                    np.nan,

                "left_detections":
                    np.nan,

                "right_detections":
                    np.nan,
            })

            failed += 1

            continue

        # ====================================================
        # FIND MATCHING DATASET ROW
        # ====================================================

        dataset_match = dataset_df[
            (
                dataset_df["label"]
                .astype(str)
                == label
            )
            &
            (
                dataset_df["video_name"]
                .astype(str)
                == video_name
            )
        ]

        if dataset_match.empty:

            error_message = (
                "Matching dataset feature row "
                "not found"
            )

            print(
                f"  ERROR: {error_message}"
            )

            results.append({

                "class_id":
                    row["class_id"],

                "label":
                    label,

                "video_name":
                    video_name,

                "split":
                    row["split"],

                "video_path":
                    str(video_path),

                "status":
                    "FAILED",

                "error":
                    error_message,

                "mae":
                    np.nan,

                "max_difference":
                    np.nan,

                "matching_percentage":
                    np.nan,

                "exact_match":
                    False,

                "frames":
                    np.nan,

                "left_detections":
                    np.nan,

                "right_detections":
                    np.nan,
            })

            failed += 1

            continue

        dataset_row = (
            dataset_match.iloc[0]
        )

        dataset_features = (
            dataset_row[
                feature_columns
            ]
            .astype(float)
            .to_numpy()
        )

        # ====================================================
        # CREATE NEW LANDMARKER FOR EVERY VIDEO
        #
        # IMPORTANT:
        # This prevents:
        #
        # Input timestamp must be
        # monotonically increasing
        #
        # because VIDEO mode timestamps
        # start again from zero for each video.
        # ====================================================

        try:

            with HandLandmarker.create_from_options(
                options
            ) as landmarker:

                (
                    extracted_features,
                    frame_count,
                    left_count,
                    right_count
                ) = process_video(
                    video_path,
                    landmarker
                )

            # ------------------------------------------------
            # Compare
            # ------------------------------------------------

            (
                mae,
                maximum_difference,
                matching_percentage,
                exact_match
            ) = compare_features(
                extracted_features,
                dataset_features
            )

            successful += 1

            if exact_match:

                exact_matches += 1

                status = "MATCH"

            else:

                mismatches += 1

                status = "MISMATCH"

            # ------------------------------------------------
            # PRINT
            # ------------------------------------------------

            print(
                f"  Frames: "
                f"{frame_count}"
            )

            print(
                f"  Left detections: "
                f"{left_count}"
            )

            print(
                f"  Right detections: "
                f"{right_count}"
            )

            print(
                f"  Feature size: "
                f"{len(extracted_features)}"
            )

            print(
                f"  MAE: "
                f"{mae:.8f}"
            )

            print(
                f"  Maximum difference: "
                f"{maximum_difference:.8f}"
            )

            print(
                f"  Matching values: "
                f"{matching_percentage:.2f}%"
            )

            print(
                f"  Result: "
                f"{status}"
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            results.append({

                "class_id":
                    row["class_id"],

                "label":
                    label,

                "video_name":
                    video_name,

                "split":
                    row["split"],

                "video_path":
                    str(video_path),

                "status":
                    status,

                "error":
                    "",

                "mae":
                    mae,

                "max_difference":
                    maximum_difference,

                "matching_percentage":
                    matching_percentage,

                "exact_match":
                    exact_match,

                "frames":
                    frame_count,

                "left_detections":
                    left_count,

                "right_detections":
                    right_count,
            })

        except Exception as error:

            failed += 1

            error_message = str(
                error
            )

            print(
                f"  ERROR: "
                f"{error_message}"
            )

            results.append({

                "class_id":
                    row["class_id"],

                "label":
                    label,

                "video_name":
                    video_name,

                "split":
                    row["split"],

                "video_path":
                    str(video_path),

                "status":
                    "FAILED",

                "error":
                    error_message,

                "mae":
                    np.nan,

                "max_difference":
                    np.nan,

                "matching_percentage":
                    np.nan,

                "exact_match":
                    False,

                "frames":
                    np.nan,

                "left_detections":
                    np.nan,

                "right_detections":
                    np.nan,
            })

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "                    SUMMARY"
    )
    print("=" * 70)

    print(
        f"\nTotal test videos:       "
        f"{len(test_metadata)}"
    )

    print(
        f"Successfully compared:   "
        f"{successful}"
    )

    print(
        f"Failed:                  "
        f"{failed}"
    )

    print(
        f"Exact feature matches:   "
        f"{exact_matches}"
    )

    print(
        f"Feature mismatches:      "
        f"{mismatches}"
    )

    # ========================================================
    # MAE STATISTICS
    # ========================================================

    successful_df = results_df[
        results_df["status"].isin([
            "MATCH",
            "MISMATCH"
        ])
    ]

    if not successful_df.empty:

        mae_values = (
            successful_df["mae"]
            .dropna()
        )

        if not mae_values.empty:

            print("\n")

            print(
                f"Average MAE:             "
                f"{mae_values.mean():.8f}"
            )

            print(
                f"Median MAE:              "
                f"{mae_values.median():.8f}"
            )

            print(
                f"Minimum MAE:             "
                f"{mae_values.min():.8f}"
            )

            print(
                f"Maximum MAE:             "
                f"{mae_values.max():.8f}"
            )

            if successful > 0:

                match_rate = (
                    exact_matches
                    / successful
                    * 100
                )

                print(
                    f"\nExact feature-match rate: "
                    f"{match_rate:.2f}%"
                )

    # ========================================================
    # WORST MATCHES
    # ========================================================

    if not successful_df.empty:

        print("\n")
        print("-" * 70)
        print(
            "              WORST FEATURE MATCHES"
        )
        print("-" * 70)

        worst = (
            successful_df
            .sort_values(
                "mae",
                ascending=False
            )
            .head(10)
        )

        for _, item in worst.iterrows():

            print(
                f"{str(item['label']):18s} "
                f"{str(item['video_name']):30s} "
                f"MAE="
                f"{item['mae']:.8f} "
                f"Match="
                f"{item['matching_percentage']:.2f}%"
            )

    # ========================================================
    # BEST MATCHES
    # ========================================================

    if not successful_df.empty:

        print("\n")
        print("-" * 70)
        print(
            "               BEST FEATURE MATCHES"
        )
        print("-" * 70)

        best = (
            successful_df
            .sort_values(
                "mae",
                ascending=True
            )
            .head(10)
        )

        for _, item in best.iterrows():

            print(
                f"{str(item['label']):18s} "
                f"{str(item['video_name']):30s} "
                f"MAE="
                f"{item['mae']:.8f} "
                f"Match="
                f"{item['matching_percentage']:.2f}%"
            )

    # ========================================================
    # FAILED VIDEOS
    # ========================================================

    failed_df = results_df[
        results_df["status"]
        == "FAILED"
    ]

    if not failed_df.empty:

        print("\n")
        print("-" * 70)
        print(
            "                  FAILED VIDEOS"
        )
        print("-" * 70)

        for _, item in failed_df.iterrows():

            print(
                f"{str(item['label']):18s} "
                f"{str(item['video_name']):30s} "
                f"{item['error']}"
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH.resolve()
    )

    print("=" * 70)

    print(
        "\n             FEATURE COMPARISON COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()