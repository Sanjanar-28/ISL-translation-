from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"D:\Amrita SLR Dataset\sign project"
)

METADATA_PATH = Path(
    "data/video_metadata_split.csv"
)

MODEL_PATH = Path(
    "data/ml_normalized/tuned_random_forest.joblib"
)

SAVED_X_PATH = Path(
    "data/ml_normalized/X_test.npy"
)

SAVED_Y_PATH = Path(
    "data/ml_normalized/y_test.npy"
)

OUTPUT_PATH = Path(
    "data/ml_normalized/live_vs_saved_prediction_comparison.csv"
)

LANDMARKER_PATH = Path(
    "models/hand_landmarker.task"
)

NUM_SAMPLED_FRAMES = 20

FEATURES_PER_HAND = 63

FEATURES_PER_FRAME = 126

EXPECTED_FEATURES = 2520


# ============================================================
# MEDIAPIPE
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


# ============================================================
# CREATE IMAGE MODE LANDMARKER
# ============================================================

def create_hand_landmarker():

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                LANDMARKER_PATH
            )
        ),

        # IMPORTANT:
        # IMAGE mode has NO timestamp requirement.
        running_mode=RunningMode.IMAGE,

        num_hands=2,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(
        options
    )


# ============================================================
# EMPTY HAND
# ============================================================

def empty_hand():

    return [np.nan] * 63


# ============================================================
# EXTRACT HAND
# ============================================================

def extract_hand(landmarks):

    features = []

    for landmark in landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(features):

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


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate_features(features):

    array = np.asarray(
        features,
        dtype=np.float64
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

    # Completely missing columns
    # become zero.
    array = np.nan_to_num(
        array,
        nan=0.0
    )

    return array


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def extract_video_features(video_path):

    print(
        f"  Extracting: "
        f"{video_path.name}"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    frames = []

    left_detections = 0
    right_detections = 0

    frame_number = 0

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Create ONE IMAGE detector for this video.
    # No timestamp state exists.
    # --------------------------------------------------------

    with create_hand_landmarker() as landmarker:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:

                break

            frame_number += 1

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb
            )

            # ------------------------------------------------
            # IMAGE MODE
            #
            # NO TIMESTAMP.
            # ------------------------------------------------

            result = (
                landmarker.detect(
                    mp_image
                )
            )

            left = empty_hand()

            right = empty_hand()

            left_found = False
            right_found = False

            # ------------------------------------------------
            # Extract hands
            # ------------------------------------------------

            for (
                hand_index,
                landmarks
            ) in enumerate(
                result.hand_landmarks
            ):

                handedness = (
                    result.handedness[
                        hand_index
                    ][0]
                )

                label = (
                    handedness.category_name
                )

                hand_features = (
                    extract_hand(
                        landmarks
                    )
                )

                if label == "Left":

                    left = hand_features
                    left_found = True

                elif label == "Right":

                    right = hand_features
                    right_found = True

            if left_found:

                left_detections += 1

            if right_found:

                right_detections += 1

            # ------------------------------------------------
            # Wrist normalization
            # ------------------------------------------------

            left = normalize_hand(
                left
            )

            right = normalize_hand(
                right
            )

            # ------------------------------------------------
            # 126 features/frame
            # ------------------------------------------------

            frames.append(
                left + right
            )

    cap.release()

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(frames) == 0:

        raise RuntimeError(
            "No frames extracted."
        )

    # ========================================================
    # NUMPY
    # ========================================================

    features = np.asarray(
        frames,
        dtype=np.float64
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
    # VALIDATE
    # ========================================================

    if sampled.shape != (
        20,
        126
    ):

        raise RuntimeError(
            f"Unexpected shape: "
            f"{sampled.shape}"
        )

    # ========================================================
    # FLATTEN
    # ========================================================

    flattened = (
        sampled.flatten()
    )

    if len(flattened) != 2520:

        raise RuntimeError(
            f"Expected 2520 features, "
            f"got {len(flattened)}"
        )

    return (
        flattened.astype(
            np.float32
        ),
        total_frames,
        left_detections,
        right_detections
    )


# ============================================================
# COMPARE FEATURES
# ============================================================

def compare_features(
    live_features,
    saved_features
):

    live = np.asarray(
        live_features,
        dtype=np.float64
    )

    saved = np.asarray(
        saved_features,
        dtype=np.float64
    )

    if live.shape != saved.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"{live.shape} vs "
            f"{saved.shape}"
        )

    difference = np.abs(
        live - saved
    )

    mae = float(
        np.mean(difference)
    )

    max_difference = float(
        np.max(difference)
    )

    matching_percentage = float(
        np.mean(
            difference <= 1e-7
        ) * 100
    )

    exact_match = (
        max_difference <= 1e-7
    )

    return (
        mae,
        max_difference,
        matching_percentage,
        exact_match
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "       LIVE VS SAVED FEATURE/PREDICTION VERIFICATION"
    )
    print("=" * 70)

    # ========================================================
    # MODEL
    # ========================================================

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded.")

    print(
        f"Model classes: "
        f"{len(model.classes_)}"
    )

    # ========================================================
    # SAVED TEST DATA
    # ========================================================

    print(
        "\nLoading saved test data..."
    )

    X_test = np.load(
        SAVED_X_PATH
    )

    y_test = np.load(
        SAVED_Y_PATH
    )

    print(
        f"X_test shape: "
        f"{X_test.shape}"
    )

    print(
        f"y_test shape: "
        f"{y_test.shape}"
    )

    # ========================================================
    # SAVED PREDICTIONS
    # ========================================================

    saved_predictions = (
        model.predict(
            X_test
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    print(
        "\nLoading metadata..."
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    test_metadata = (
        metadata[
            metadata["split"]
            .astype(str)
            .str.lower()
            == "test"
        ]
        .reset_index(drop=True)
    )

    print(
        f"Test videos: "
        f"{len(test_metadata)}"
    )

    if len(test_metadata) != len(
        X_test
    ):

        raise ValueError(
            "Number of test videos does "
            "not match X_test."
        )

    # ========================================================
    # CLASS MAP
    # ========================================================

    class_map = {}

    for _, row in metadata.iterrows():

        class_map[
            int(row["class_id"])
        ] = str(
            row["label"]
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    live_correct = 0

    saved_correct = 0

    prediction_same = 0

    feature_matches = 0

    total = len(
        test_metadata
    )

    # ========================================================
    # PROCESS ALL TEST VIDEOS
    # ========================================================

    for index, row in (
        test_metadata.iterrows()
    ):

        label = str(
            row["label"]
        )

        video_name = str(
            row["video_name"]
        )

        class_id = int(
            row["class_id"]
        )

        print()
        print("=" * 70)

        print(
            f"[{index + 1}/{total}] "
            f"{label}/{video_name}"
        )

        # ----------------------------------------------------
        # DATASET PATH
        # ----------------------------------------------------

        video_path = (
            DATASET_ROOT
            / f"{class_id}.{label}"
            / video_name
        )

        print(
            f"Path: {video_path}"
        )

        record = {

            "class_id": class_id,

            "label": label,

            "video_name": video_name,

            "split": "test",

            "video_path": str(
                video_path
            ),

            "status": "success",

            "error": "",

            "mae": np.nan,

            "max_difference": np.nan,

            "matching_percentage": np.nan,

            "exact_match": False,

            "frames": 0,

            "left_detections": 0,

            "right_detections": 0,

            "saved_prediction_id": int(
                saved_predictions[index]
            ),

            "live_prediction_id": -1,

            "saved_prediction": class_map.get(
                int(saved_predictions[index]),
                str(saved_predictions[index])
            ),

            "live_prediction": "",

            "predictions_same": False,
        }

        try:

            if not video_path.exists():

                raise FileNotFoundError(
                    f"Video not found: "
                    f"{video_path}"
                )

            # ------------------------------------------------
            # LIVE EXTRACTION
            # ------------------------------------------------

            (
                live_features,
                frames,
                left_detections,
                right_detections
            ) = extract_video_features(
                video_path
            )

            record["frames"] = frames

            record[
                "left_detections"
            ] = left_detections

            record[
                "right_detections"
            ] = right_detections

            # ------------------------------------------------
            # FEATURE COMPARISON
            # ------------------------------------------------

            (
                mae,
                max_difference,
                matching_percentage,
                exact_match
            ) = compare_features(
                live_features,
                X_test[index]
            )

            record["mae"] = mae

            record[
                "max_difference"
            ] = max_difference

            record[
                "matching_percentage"
            ] = matching_percentage

            record[
                "exact_match"
            ] = exact_match

            # ------------------------------------------------
            # LIVE PREDICTION
            # ------------------------------------------------

            live_prediction = (
                model.predict(
                    live_features.reshape(
                        1,
                        -1
                    )
                )[0]
            )

            saved_prediction = (
                saved_predictions[index]
            )

            live_label = class_map.get(
                int(live_prediction),
                str(live_prediction)
            )

            saved_label = class_map.get(
                int(saved_prediction),
                str(saved_prediction)
            )

            same_prediction = (
                int(live_prediction)
                == int(saved_prediction)
            )

            record[
                "live_prediction_id"
            ] = int(
                live_prediction
            )

            record[
                "live_prediction"
            ] = live_label

            record[
                "predictions_same"
            ] = same_prediction

            # ------------------------------------------------
            # ACCURACY
            # ------------------------------------------------

            if (
                int(live_prediction)
                == class_id
            ):

                live_correct += 1

            if (
                int(saved_prediction)
                == class_id
            ):

                saved_correct += 1

            if same_prediction:

                prediction_same += 1

            if exact_match:

                feature_matches += 1

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            print(
                f"  Frames: "
                f"{frames}"
            )

            print(
                f"  Left detections: "
                f"{left_detections}"
            )

            print(
                f"  Right detections: "
                f"{right_detections}"
            )

            print(
                f"  MAE: "
                f"{mae:.15f}"
            )

            print(
                f"  Max difference: "
                f"{max_difference:.15f}"
            )

            print(
                f"  Feature match: "
                f"{matching_percentage:.2f}%"
            )

            print(
                f"  Actual: "
                f"{label}"
            )

            print(
                f"  Saved prediction: "
                f"{saved_label}"
            )

            print(
                f"  Live prediction: "
                f"{live_label}"
            )

            print(
                f"  Predictions same: "
                f"{same_prediction}"
            )

        except Exception as exc:

            record["status"] = "failed"

            record["error"] = str(
                exc
            )

            print(
                f"  ERROR: {exc}"
            )

        results.append(
            record
        )

    # ========================================================
    # SAVE
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

    successful_df = (
        results_df[
            results_df["status"]
            == "success"
        ]
    )

    successful_count = len(
        successful_df
    )

    failed_count = (
        total
        - successful_count
    )

    print()
    print("=" * 70)
    print(
        "                       SUMMARY"
    )
    print("=" * 70)

    print(
        f"\nTotal test videos:       "
        f"{total}"
    )

    print(
        f"Successfully processed:  "
        f"{successful_count}"
    )

    print(
        f"Failed:                  "
        f"{failed_count}"
    )

    if successful_count > 0:

        live_accuracy = (
            live_correct
            / successful_count
            * 100
        )

        saved_accuracy = (
            saved_correct
            / successful_count
            * 100
        )

        same_accuracy = (
            prediction_same
            / successful_count
            * 100
        )

        feature_accuracy = (
            feature_matches
            / successful_count
            * 100
        )

        print(
            f"Feature matches: "
            f"{feature_matches}/"
            f"{successful_count}"
        )

        print(
            f"Feature match rate: "
            f"{feature_accuracy:.2f}%"
        )

        print(
            f"Live prediction accuracy: "
            f"{live_correct}/"
            f"{successful_count}"
            f" = {live_accuracy:.2f}%"
        )

        print(
            f"Saved prediction accuracy: "
            f"{saved_correct}/"
            f"{successful_count}"
            f" = {saved_accuracy:.2f}%"
        )

        print(
            f"Live vs saved prediction "
            f"identical: "
            f"{prediction_same}/"
            f"{successful_count}"
            f" = {same_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        mae_values = (
            successful_df[
                "mae"
            ]
            .astype(float)
        )

        print()
        print(
            "Feature MAE statistics:"
        )

        print(
            f"  Mean:   "
            f"{mae_values.mean():.15f}"
        )

        print(
            f"  Median: "
            f"{mae_values.median():.15f}"
        )

        print(
            f"  Max:    "
            f"{mae_values.max():.15f}"
        )

        # ----------------------------------------------------
        # DISAGREEMENTS
        # ----------------------------------------------------

        disagreements = (
            successful_df[
                successful_df[
                    "predictions_same"
                ] == False
            ]
        )

        print()
        print(
            f"Prediction disagreements: "
            f"{len(disagreements)}"
        )

        if len(disagreements) > 0:

            print()
            print(
                "First disagreements:"
            )

            for _, r in (
                disagreements
                .head(20)
                .iterrows()
            ):

                print(
                    f"{str(r['label']):18}"
                    f"{str(r['video_name']):35}"
                    f"saved={str(r['saved_prediction']):18}"
                    f"live={str(r['live_prediction']):18}"
                    f"MAE={float(r['mae']):.12f}"
                )

    # ========================================================
    # FAILED VIDEOS
    # ========================================================

    if failed_count > 0:

        print()
        print(
            "Failed videos:"
        )

        failed_df = (
            results_df[
                results_df["status"]
                == "failed"
            ]
        )

        for _, r in (
            failed_df.iterrows()
        ):

            print(
                f"{r['label']:18}"
                f"{r['video_name']:35}"
                f"{r['error']}"
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH.resolve()
    )

    print()
    print("=" * 70)
    print(
        "             VERIFICATION COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()