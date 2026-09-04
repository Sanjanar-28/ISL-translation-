from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml_normalized"
    / "tuned_random_forest.joblib"
)

LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "label_mapping.csv"
)

HAND_LANDMARKER_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

NUM_SAMPLED_FRAMES = 20

FEATURES_PER_HAND = 63
FEATURES_PER_FRAME = 126
TOTAL_FEATURES = 2520


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


OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(
            HAND_LANDMARKER_PATH
        )
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
    Return 63 NaN values.

    21 landmarks x 3 coordinates
    = 63 features.
    """

    return [np.nan] * FEATURES_PER_HAND


# ------------------------------------------------------------
# Extract 21 landmarks
# ------------------------------------------------------------

def extract_hand(landmarks):
    """
    Convert MediaPipe hand landmarks into:

        x1,y1,z1,x2,y2,z2,...,x21,y21,z21

    Total = 63 features.
    """

    features = []

    for landmark in landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    return features


# ------------------------------------------------------------
# Wrist-relative normalization
# ------------------------------------------------------------

def normalize_hand(features):
    """
    Normalize all hand landmarks relative to wrist.

    Landmark 0 is the wrist.

    For every landmark:

        x' = x - wrist_x
        y' = y - wrist_y
        z' = z - wrist_z
    """

    # Entire hand missing
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
                np.nan,
            ])

        else:

            normalized.extend([
                x - wrist_x,
                y - wrist_y,
                z - wrist_z,
            ])

    return normalized


# ------------------------------------------------------------
# Temporal interpolation
# ------------------------------------------------------------

def interpolate_features(features):
    """
    Fill missing landmark values using temporal interpolation.

    This is intentionally identical to the dataset
    preprocessing pipeline.
    """

    array = np.asarray(
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
                limit_direction="both",
            )
        )

        array[:, column] = series.values

    return array


# ------------------------------------------------------------
# Final NaN protection
# ------------------------------------------------------------

def fill_remaining_nan(features):
    """
    If an entire coordinate column is NaN, interpolation
    cannot fill it.

    Replace remaining NaN values with zero.

    This should normally only affect completely missing
    hand coordinates.
    """

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# ============================================================
# EXTRACT FEATURES FROM ONE VIDEO
# ============================================================

def extract_video_features(
    video_path,
    landmarker,
):
    """
    Extract the exact 2520-dimensional feature vector.

    Pipeline:

        Video
          ↓
        MediaPipe
          ↓
        Left 63 + Right 63
          ↓
        Wrist normalization
          ↓
        Temporal interpolation
          ↓
        20 frames
          ↓
        20 x 126
          ↓
        Flatten
          ↓
        2520 features
    """

    video_path = Path(video_path)

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
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

    # ========================================================
    # READ VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ----------------------------------------------------
        # Frame number
        # ----------------------------------------------------

        frame_number += 1

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # ----------------------------------------------------
        # Timestamp
        #
        # IMPORTANT:
        # Starts from zero for EVERY video.
        # ----------------------------------------------------

        timestamp_ms = int(
            ((frame_number - 1) / fps)
            * 1000
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # ----------------------------------------------------
        # Default hands
        # ----------------------------------------------------

        left = empty_hand()
        right = empty_hand()

        left_found = False
        right_found = False

        # ----------------------------------------------------
        # Process detected hands
        # ----------------------------------------------------

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            if hand_index >= len(
                result.handedness
            ):
                continue

            handedness = (
                result.handedness[
                    hand_index
                ][0]
            )

            label = (
                handedness.category_name
            )

            features = extract_hand(
                landmarks
            )

            if label == "Left":

                left = features
                left_found = True

            elif label == "Right":

                right = features
                right_found = True

        # ----------------------------------------------------
        # Detection statistics
        # ----------------------------------------------------

        if left_found:
            left_detections += 1

        if right_found:
            right_detections += 1

        # ----------------------------------------------------
        # Wrist normalization
        # ----------------------------------------------------

        left = normalize_hand(
            left
        )

        right = normalize_hand(
            right
        )

        # ----------------------------------------------------
        # One frame = 126 features
        # ----------------------------------------------------

        frame_features = (
            left + right
        )

        if len(frame_features) != FEATURES_PER_FRAME:

            raise RuntimeError(
                "Unexpected feature count per frame: "
                f"{len(frame_features)}"
            )

        frames.append(
            frame_features
        )

    cap.release()

    # ========================================================
    # VALIDATE VIDEO
    # ========================================================

    total_frames = len(frames)

    if total_frames == 0:

        raise RuntimeError(
            f"No frames extracted from: {video_path}"
        )

    # ========================================================
    # NUMPY
    # ========================================================

    features = np.asarray(
        frames,
        dtype=float
    )

    # ========================================================
    # TEMPORAL INTERPOLATION
    # ========================================================

    features = interpolate_features(
        features
    )

    # ========================================================
    # FILL REMAINING NaN
    # ========================================================

    features = fill_remaining_nan(
        features
    )

    # ========================================================
    # TEMPORAL SAMPLING
    # ========================================================

    if total_frames >= NUM_SAMPLED_FRAMES:

        indices = np.linspace(
            0,
            total_frames - 1,
            NUM_SAMPLED_FRAMES,
            dtype=int,
        )

    else:

        # If video contains fewer than 20 frames,
        # repeat the last available frame.

        indices = np.linspace(
            0,
            total_frames - 1,
            NUM_SAMPLED_FRAMES,
        ).round().astype(int)

        indices = np.clip(
            indices,
            0,
            total_frames - 1,
        )

    sampled = features[
        indices
    ]

    # ========================================================
    # VALIDATION
    # ========================================================

    expected_shape = (
        NUM_SAMPLED_FRAMES,
        FEATURES_PER_FRAME,
    )

    if sampled.shape != expected_shape:

        raise RuntimeError(
            "Unexpected sampled shape: "
            f"{sampled.shape}, "
            f"expected {expected_shape}"
        )

    # ========================================================
    # FLATTEN
    # ========================================================

    flattened = sampled.flatten()

    # ========================================================
    # FINAL FEATURE VALIDATION
    # ========================================================

    if len(flattened) != TOTAL_FEATURES:

        raise RuntimeError(
            f"Expected {TOTAL_FEATURES} features, "
            f"got {len(flattened)}"
        )

    return (
        flattened,
        total_frames,
        left_detections,
        right_detections,
    )


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():

    if not LABEL_MAPPING_PATH.exists():

        raise FileNotFoundError(
            "Label mapping not found:\n"
            f"{LABEL_MAPPING_PATH}"
        )

    mapping = pd.read_csv(
        LABEL_MAPPING_PATH
    )

    required_columns = {
        "label_id",
        "label",
    }

    missing = (
        required_columns
        - set(mapping.columns)
    )

    if missing:

        raise RuntimeError(
            "Label mapping missing columns: "
            f"{missing}"
        )

    mapping = mapping.sort_values(
        "label_id"
    )

    label_names = {}

    for _, row in mapping.iterrows():

        label_names[
            int(row["label_id"])
        ] = str(
            row["label"]
        )

    return label_names


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model not found:\n"
            f"{MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# PREDICT
# ============================================================

def predict_video(video_path):

    print("\n")
    print("=" * 70)
    print("                    VIDEO PREDICTION")
    print("=" * 70)

    print(
        f"\nVideo:\n{video_path}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    label_names = load_label_mapping()

    # --------------------------------------------------------
    # Create NEW landmarker for this video
    # --------------------------------------------------------

    with HandLandmarker.create_from_options(
        OPTIONS
    ) as landmarker:

        (
            features,
            total_frames,
            left_detections,
            right_detections,
        ) = extract_video_features(
            video_path,
            landmarker,
        )

    # --------------------------------------------------------
    # Print extraction information
    # --------------------------------------------------------

    print(
        f"\nFrames: {total_frames}"
    )

    print(
        f"Left detections: "
        f"{left_detections}"
    )

    print(
        f"Right detections: "
        f"{right_detections}"
    )

    print(
        f"Feature size: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X = features.reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X
    )

    predicted_id = int(
        prediction[0]
    )

    predicted_label = label_names.get(
        predicted_id,
        f"class_{predicted_id}",
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(X)[0]
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = None

    if probabilities is not None:

        classes = model.classes_

        predicted_position = np.where(
            classes == predicted_id
        )[0]

        if len(predicted_position) > 0:

            confidence = float(
                probabilities[
                    predicted_position[0]
                ]
            )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"\nPredicted class ID: "
        f"{predicted_id}"
    )

    print(
        f"Predicted sign: "
        f"{predicted_label}"
    )

    if confidence is not None:

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    if probabilities is not None:

        classes = model.classes_

        ranked = np.argsort(
            probabilities
        )[::-1]

        print(
            "\nTop predictions:"
        )

        for rank, position in enumerate(
            ranked[:5],
            start=1,
        ):

            class_id = int(
                classes[position]
            )

            label = label_names.get(
                class_id,
                f"class_{class_id}",
            )

            probability = float(
                probabilities[position]
            )

            print(
                f"  {rank}. "
                f"{label:<20s} "
                f"{probability * 100:.2f}%"
            )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "\nFINAL PREDICTION:"
    )

    print(
        f"  Sign: {predicted_label}"
    )

    print(
        f"  Class ID: {predicted_id}"
    )

    if confidence is not None:

        print(
            f"  Confidence: "
            f"{confidence * 100:.2f}%"
        )

    print(
        "\n" + "=" * 70
    )

    return {
        "class_id": predicted_id,
        "label": predicted_label,
        "confidence": confidence,
        "features": features,
        "frames": total_frames,
        "left_detections": left_detections,
        "right_detections": right_detections,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    import sys

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "  python -m src.ml.predict_video "
            "<video_path>"
        )

        print(
            "\nExample:"
        )

        print(
            r'  python -m src.ml.predict_video "D:\Amrita SLR Dataset\sign project\50.power\Power 3(1).mp4"'
        )

        return

    video_path = Path(
        sys.argv[1]
    )

    predict_video(
        video_path
    )


if __name__ == "__main__":
    main()