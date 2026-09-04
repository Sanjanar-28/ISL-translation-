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

LANDMARK_DATASET = Path(
    "data/processed/full_landmark_dataset.csv"
)

NUM_SAMPLED_FRAMES = 20


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# HAND HELPERS
# ============================================================

def empty_hand():
    return [np.nan] * 63


def extract_hand(landmarks):

    features = []

    for landmark in landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features


def normalize_hand(features):

    if all(pd.isna(value) for value in features):
        return features

    wrist_x = features[0]
    wrist_y = features[1]
    wrist_z = features[2]

    normalized = []

    for i in range(21):

        x = features[i * 3]
        y = features[i * 3 + 1]
        z = features[i * 3 + 2]

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


def interpolate_features(features):

    array = np.array(
        features,
        dtype=float
    )

    for column in range(array.shape[1]):

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

        array[:, column] = series.values

    return array


# ============================================================
# EXTRACT VIDEO FEATURES
# ============================================================

def extract_video_features(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frames = []

    left_detections = 0
    right_detections = 0

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # EXACT timestamp logic from process_dataset.py
        timestamp_ms = int(
            ((frame_number - 1) / fps)
            * 1000
        )

        result = LANDMARKER.detect_for_video(
            mp_image,
            timestamp_ms
        )

        left = empty_hand()
        right = empty_hand()

        left_found = False
        right_found = False

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = (
                result.handedness[
                    hand_index
                ][0]
            )

            label = handedness.category_name

            features = extract_hand(
                landmarks
            )

            if label == "Left":

                left = features
                left_found = True

            elif label == "Right":

                right = features
                right_found = True

        # EXACT normalization from dataset pipeline
        left = normalize_hand(left)
        right = normalize_hand(right)

        frames.append(
            left + right
        )

        if left_found:
            left_detections += 1

        if right_found:
            right_detections += 1

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            "No frames extracted."
        )

    features = np.array(
        frames,
        dtype=float
    )

    # EXACT interpolation
    features = interpolate_features(
        features
    )

    # EXACT sampling
    total_frames = len(features)

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_SAMPLED_FRAMES,
        dtype=int
    )

    sampled = features[
        indices
    ]

    if sampled.shape != (
        NUM_SAMPLED_FRAMES,
        126
    ):

        raise RuntimeError(
            f"Unexpected sampled shape: "
            f"{sampled.shape}"
        )

    flattened = sampled.flatten()

    if len(flattened) != 2520:

        raise RuntimeError(
            f"Expected 2520 features, "
            f"got {len(flattened)}"
        )

    return (
        flattened,
        total_frames,
        left_detections,
        right_detections
    )


# ============================================================
# FIND DATASET ROW
# ============================================================

def find_dataset_row(label, video_name):

    df = pd.read_csv(
        LANDMARK_DATASET
    )

    print(
        f"Dataset rows: {len(df)}"
    )

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    # Make sure features are in numerical order
    feature_columns = sorted(
        feature_columns,
        key=lambda x: int(
            x.split("_")[1]
        )
    )

    match = df[
        (df["label"] == label)
        &
        (df["video_name"] == video_name)
    ]

    if match.empty:

        raise RuntimeError(
            f"Could not find dataset row for:\n"
            f"{label}/{video_name}"
        )

    row = match.iloc[0]

    return row, feature_columns


# ============================================================
# COMPARE
# ============================================================

def compare_features(
    extracted,
    dataset_features
):

    print("\n")
    print("=" * 70)
    print("                  FEATURE COMPARISON")
    print("=" * 70)

    print(
        f"\nExtracted features: "
        f"{len(extracted)}"
    )

    print(
        f"Dataset features:   "
        f"{len(dataset_features)}"
    )

    if len(extracted) != len(dataset_features):

        print(
            "\nERROR: Feature sizes do not match!"
        )

        return False

    extracted = np.asarray(
        extracted,
        dtype=float
    )

    dataset_features = np.asarray(
        dataset_features,
        dtype=float
    )

    # Handle NaN safely
    valid = (
        np.isfinite(extracted)
        &
        np.isfinite(dataset_features)
    )

    if not np.any(valid):

        print(
            "\nERROR: No valid feature values."
        )

        return False

    differences = np.abs(
        extracted[valid]
        -
        dataset_features[valid]
    )

    mae = np.mean(
        differences
    )

    max_difference = np.max(
        differences
    )

    # Relative tolerance
    close = np.isclose(
        extracted[valid],
        dataset_features[valid],
        atol=1e-5,
        rtol=1e-4
    )

    match_percentage = (
        np.mean(close) * 100
    )

    print(
        f"\nMean absolute difference: "
        f"{mae:.8f}"
    )

    print(
        f"Maximum difference:       "
        f"{max_difference:.8f}"
    )

    print(
        f"Matching feature values:  "
        f"{match_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # First 20 values
    # --------------------------------------------------------

    print("\nFirst 20 values:\n")

    print(
        "Index     Extracted        Dataset"
    )

    print(
        "-" * 45
    )

    for i in range(
        min(20, len(extracted))
    ):

        print(
            f"{i:<10}"
            f"{extracted[i]:<17.8f}"
            f"{dataset_features[i]:.8f}"
        )

    # --------------------------------------------------------
    # Largest differences
    # --------------------------------------------------------

    abs_diff_all = np.abs(
        extracted
        -
        dataset_features
    )

    largest = np.argsort(
        abs_diff_all
    )[::-1][:10]

    print(
        "\nLargest differences:"
    )

    print(
        "Index     Extracted        "
        "Dataset          Difference"
    )

    print(
        "-" * 65
    )

    for index in largest:

        print(
            f"{index:<10}"
            f"{extracted[index]:<17.8f}"
            f"{dataset_features[index]:<17.8f}"
            f"{abs_diff_all[index]:.8f}"
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    # We use a reasonably strict threshold.
    if mae < 1e-4:

        print(
            "\n"
            + "-" * 70
        )

        print(
            "RESULT: FEATURES MATCH ✅"
        )

        print(
            "-" * 70
        )

        return True

    else:

        print(
            "\n"
            + "-" * 70
        )

        print(
            "RESULT: FEATURES DO NOT MATCH ❌"
        )

        print(
            "-" * 70
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    import sys

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python src/inference/compare_features.py '
            '"VIDEO_PATH"'
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    if not video_path.exists():

        print(
            f"ERROR: Video does not exist:\n"
            f"{video_path}"
        )

        sys.exit(1)

    print("\n")
    print("=" * 70)
    print("              FEATURE PIPELINE COMPARISON")
    print("=" * 70)

    print(
        f"\nVideo:\n{video_path}"
    )

    # --------------------------------------------------------
    # Determine label/video name
    # --------------------------------------------------------

    video_name = video_path.name

    # Find metadata
    metadata = pd.read_csv(
        METADATA_PATH
    )

    matches = metadata[
        metadata["video_name"]
        == video_name
    ]

    if matches.empty:

        print(
            "\nERROR: Video not found in metadata."
        )

        sys.exit(1)

    row = matches.iloc[0]

    label = row["label"]

    print(
        f"\nLabel: {label}"
    )

    print(
        f"Split: {row['split']}"
    )

    # --------------------------------------------------------
    # Create NEW MediaPipe instance
    # --------------------------------------------------------

    global LANDMARKER

    with HandLandmarker.create_from_options(
        OPTIONS
    ) as LANDMARKER:

        (
            extracted,
            frames,
            left,
            right
        ) = extract_video_features(
            video_path
        )

    # --------------------------------------------------------
    # Extraction summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                 EXTRACTION SUMMARY")
    print("=" * 70)

    print(
        f"\nFrames: "
        f"{frames}"
    )

    print(
        f"Left detections: "
        f"{left}"
    )

    print(
        f"Right detections: "
        f"{right}"
    )

    print(
        f"Feature size: "
        f"{len(extracted)}"
    )

    # --------------------------------------------------------
    # Dataset row
    # --------------------------------------------------------

    dataset_row, feature_columns = (
        find_dataset_row(
            label,
            video_name
        )
    )

    dataset_features = (
        dataset_row[
            feature_columns
        ]
        .astype(float)
        .to_numpy()
    )

    print(
        f"\nDataset label: "
        f"{dataset_row['label']}"
    )

    print(
        f"Dataset split: "
        f"{dataset_row['split']}"
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    compare_features(
        extracted,
        dataset_features
    )


if __name__ == "__main__":
    main()