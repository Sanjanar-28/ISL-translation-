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

OUTPUT_PATH = Path(
    "data/processed/full_landmark_dataset.csv"
)

NUM_SAMPLED_FRAMES = 20


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(
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
# HELPERS
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
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    landmarker
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frames = []

    left_count = 0
    right_count = 0

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

        timestamp_ms = int(
            (frame_number - 1)
            * 1000
            / fps
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # LEFT HAND
        # ----------------------------------------------------

        if (
            result.hand_landmarks
            and len(result.hand_landmarks) > 0
        ):

            left = empty_hand()

            right = empty_hand()

            for hand_index, handedness in enumerate(
                result.handedness
            ):

                if hand_index >= len(
                    result.hand_landmarks
                ):
                    continue

                hand_landmarks = (
                    result.hand_landmarks[
                        hand_index
                    ]
                )

                hand_label = (
                    handedness[0].category_name
                    if handedness
                    else ""
                )

                hand_features = extract_hand(
                    hand_landmarks
                )

                if hand_label.lower() == "left":

                    left = hand_features
                    left_count += 1

                elif hand_label.lower() == "right":

                    right = hand_features
                    right_count += 1

        else:

            left = empty_hand()
            right = empty_hand()

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        left = normalize_hand(left)

        right = normalize_hand(right)

        # ----------------------------------------------------
        # STORE FRAME
        # ----------------------------------------------------

        frames.append(
            left + right
        )

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            "No frames found."
        )

    features = np.array(
        frames,
        dtype=float
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    features = interpolate_features(
        features
    )
    # --------------------------------------------------------
# Fill remaining NaN values
# --------------------------------------------------------

    features = np.nan_to_num(
       features,
       nan=0.0,
       posinf=0.0,
       neginf=0.0
    )
    # --------------------------------------------------------
    # Temporal sampling
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if sampled.shape != (
        20,
        126
    ):

        raise RuntimeError(
            f"Unexpected shape: "
            f"{sampled.shape}"
        )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    flattened = sampled.flatten()

    if len(flattened) != 2520:

        raise RuntimeError(
            f"Expected 2520 features, "
            f"got {len(flattened)}"
        )

    return (
        flattened.tolist(),
        total_frames,
        left_count,
        right_count
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("             PROCESSING ALL VIDEOS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset root not found:\n"
            f"{DATASET_ROOT}"
        )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_PATH}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"\nMetadata videos: "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Existing processed videos
    # --------------------------------------------------------

    processed_videos = set()

    if OUTPUT_PATH.exists():

        try:

            existing = pd.read_csv(
                OUTPUT_PATH,
                usecols=[
                    "class_id",
                    "label",
                    "video_name"
                ]
            )

            for _, row in existing.iterrows():

                processed_videos.add(
                    (
                        int(row["class_id"]),
                        row["label"],
                        row["video_name"]
                    )
                )

            print(
                f"Existing processed videos: "
                f"{len(processed_videos)}"
            )

        except Exception:

            print(
                "Could not read existing output."
            )

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for index, row in metadata.iterrows():

        label = row["label"]

        video_name = row["video_name"]

        class_id = int(
            row["class_id"]
        )

        # ----------------------------------------------------
        # Unique video key
        # ----------------------------------------------------

        key = (
            class_id,
            label,
            video_name
        )

        if key in processed_videos:

            print(
                f"[{index + 1}/{len(metadata)}] "
                f"SKIP: "
                f"{label}/{video_name}"
            )

            continue

        # ----------------------------------------------------
        # Video path
        # ----------------------------------------------------

        video_path = (
            DATASET_ROOT
            / f"{class_id}.{label}"
            / video_name
        )

        print(
            f"\n[{index + 1}/{len(metadata)}] "
            f"Processing: "
            f"{label}/{video_name}"
        )

        if not video_path.exists():

            print(
                "  ERROR: File not found"
            )

            continue

        try:

            # =================================================
            # NEW MEDIAPIPE INSTANCE PER VIDEO
            # =================================================

            with HandLandmarker.create_from_options(
                options
            ) as landmarker:

                (
                    features,
                    frame_count,
                    left_count,
                    right_count
                ) = process_video(
                    video_path,
                    landmarker
                )

            # =================================================
            # STEP 1:
            # REJECT VIDEOS WITH ZERO HAND DETECTIONS
            # =================================================

            if (
                left_count == 0
                and right_count == 0
            ):

                print(
                    f"  WARNING: No hands detected in "
                    f"{label}/{video_name}"
                )

                print(
                    "  SKIPPING invalid feature vector."
                )

                continue

            # ------------------------------------------------
            # Create row
            # ------------------------------------------------

            result = {

                "class_id":
                    class_id,

                "label":
                    label,

                "video_name":
                    video_name,

                "split":
                    row["split"],

                "original_frames":
                    frame_count,

                "left_detections":
                    left_count,

                "right_detections":
                    right_count,
            }

            # ------------------------------------------------
            # Add 2520 features
            # ------------------------------------------------

            for feature_index, value in enumerate(
                features
            ):

                result[
                    f"feature_{feature_index}"
                ] = value

            result_df = pd.DataFrame(
                [result]
            )

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            if OUTPUT_PATH.exists():

                result_df.to_csv(
                    OUTPUT_PATH,
                    mode="a",
                    header=False,
                    index=False
                )

            else:

                result_df.to_csv(
                    OUTPUT_PATH,
                    mode="w",
                    header=True,
                    index=False
                )

            # ------------------------------------------------
            # Add to processed set
            # ------------------------------------------------

            processed_videos.add(
                key
            )

            print(
                f"  SUCCESS: "
                f"{frame_count} frames"
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
                f"  Features: "
                f"{len(features)}"
            )

        except Exception as e:

            print(
                f"  ERROR processing "
                f"{label}/{video_name}:"
            )

            print(
                f"  {e}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                 PROCESSING COMPLETE")
    print("=" * 70)

    if OUTPUT_PATH.exists():

        final_df = pd.read_csv(
            OUTPUT_PATH
        )

        print(
            f"\nProcessed videos: "
            f"{len(final_df)}"
        )

        print(
            f"Features per video: "
            f"{len([c for c in final_df.columns if c.startswith('feature_')])}"
        )

        print(
            f"Output: "
            f"{OUTPUT_PATH}"
        )

    else:

        print(
            "\nNo output dataset was created."
        )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()