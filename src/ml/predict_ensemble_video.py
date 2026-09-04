from pathlib import Path

import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

ORIGINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "tuned_random_forest.joblib"
)

MOTION_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml_motion"
    / "motion_random_forest.joblib"
)

TEMPORAL_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml_temporal"
    / "temporal_random_forest.joblib"
)

# ------------------------------------------------------------
# LABEL MAPPING
# ------------------------------------------------------------

LABEL_MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "label_mapping.csv"
)

# ------------------------------------------------------------
# MEDIAPIPE
# ------------------------------------------------------------

HAND_LANDMARKER_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

# ------------------------------------------------------------
# VIDEO / FEATURE CONSTANTS
# ------------------------------------------------------------

NUM_SAMPLED_FRAMES = 20

FEATURES_PER_HAND = 63
FEATURES_PER_FRAME = 126

ORIGINAL_FEATURES = 2520
MOTION_FEATURES = 5040
TEMPORAL_FEATURES = 1008


# ============================================================
# ENSEMBLE WEIGHTS
#
# Best weights found during validation tuning
# ============================================================

WEIGHT_ORIGINAL = 0.25
WEIGHT_MOTION = 0.45
WEIGHT_TEMPORAL = 0.30

# ============================================================
# PREDICTION RELIABILITY THRESHOLDS
# ============================================================

MIN_CONFIDENCE_ACCEPT = 0.15
MIN_MARGIN_ACCEPT = 0.05
MIN_MODEL_AGREEMENT = 2


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
# HAND HELPERS
# ============================================================

def empty_hand():
    """
    21 landmarks x 3 coordinates = 63 values.
    """

    return [np.nan] * FEATURES_PER_HAND


def extract_hand(landmarks):

    features = []

    for landmark in landmarks:

        features.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
        )

    return features


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(features):
    """
    Translate each hand so the wrist becomes the origin.

    Landmark 0 = wrist.
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

        x = features[i * 3]
        y = features[i * 3 + 1]
        z = features[i * 3 + 2]

        if pd.isna(x):

            normalized.extend(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )

        else:

            normalized.extend(
                [
                    x - wrist_x,
                    y - wrist_y,
                    z - wrist_z,
                ]
            )

    return normalized


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate_features(features):

    array = np.asarray(
        features,
        dtype=float,
    )

    for column in range(
        array.shape[1]
    ):

        series = pd.Series(
            array[:, column]
        )

        series = series.interpolate(
            method="linear",
            limit_direction="both",
        )

        array[:, column] = (
            series.values
        )

    return array


def fill_remaining_nan(features):

    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# ============================================================
# EXTRACT ORIGINAL FEATURES
# ============================================================

def extract_video_features(
    video_path,
    landmarker,
):

    video_path = Path(
        video_path
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
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

    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:

            break

        frame_number += 1

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb,
        )

        timestamp_ms = int(
            (
                (frame_number - 1)
                / fps
            )
            * 1000
        )

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        # ----------------------------------------------------
        # EMPTY HANDS
        # ----------------------------------------------------

        left = empty_hand()
        right = empty_hand()

        left_found = False
        right_found = False

        # ----------------------------------------------------
        # DETECT HANDS
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

            hand_features = extract_hand(
                landmarks
            )

            if label == "Left":

                left = hand_features
                left_found = True

            elif label == "Right":

                right = hand_features
                right_found = True

        # ----------------------------------------------------
        # DETECTION COUNTS
        # ----------------------------------------------------

        if left_found:

            left_detections += 1

        if right_found:

            right_detections += 1

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        left = normalize_hand(
            left
        )

        right = normalize_hand(
            right
        )

        frame_features = (
            left + right
        )

        if (
            len(frame_features)
            != FEATURES_PER_FRAME
        ):

            raise RuntimeError(
                "Unexpected frame feature "
                f"count: {len(frame_features)}"
            )

        frames.append(
            frame_features
        )

    cap.release()

    # ========================================================
    # VALIDATE VIDEO
    # ========================================================

    total_frames = len(
        frames
    )

    if total_frames == 0:

        raise RuntimeError(
            "No frames extracted from "
            f"{video_path}"
        )

    features = np.asarray(
        frames,
        dtype=float,
    )

    # ========================================================
    # INTERPOLATE
    # ========================================================

    features = interpolate_features(
        features
    )

    features = fill_remaining_nan(
        features
    )

    # ========================================================
    # SAMPLE 20 FRAMES
    # ========================================================

    if (
        total_frames
        >= NUM_SAMPLED_FRAMES
    ):

        indices = np.linspace(
            0,
            total_frames - 1,
            NUM_SAMPLED_FRAMES,
            dtype=int,
        )

    else:

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

    expected_shape = (
        NUM_SAMPLED_FRAMES,
        FEATURES_PER_FRAME,
    )

    if (
        sampled.shape
        != expected_shape
    ):

        raise RuntimeError(
            "Unexpected sampled shape:\n"
            f"Got: {sampled.shape}\n"
            f"Expected: {expected_shape}"
        )

    flattened = sampled.flatten()

    if (
        len(flattened)
        != ORIGINAL_FEATURES
    ):

        raise RuntimeError(
            f"Expected {ORIGINAL_FEATURES} "
            f"features, got "
            f"{len(flattened)}"
        )

    return (
        flattened.astype(
            np.float32
        ),
        total_frames,
        left_detections,
        right_detections,
    )


# ============================================================
# BUILD MOTION FEATURES
# ============================================================

def build_motion_features(X):

    """
    Convert:

        (samples, 2520)

    into:

        (samples, 5040)

    Expected motion representation:

        original landmark features
        +
        frame-to-frame motion differences

    2520 + 2520 = 5040
    """

    if X.ndim != 2:

        raise ValueError(
            f"Expected 2D input, "
            f"got {X.shape}"
        )

    if (
        X.shape[1]
        != ORIGINAL_FEATURES
    ):

        raise ValueError(
            f"Expected {ORIGINAL_FEATURES} "
            f"features, got "
            f"{X.shape[1]}"
        )

    # --------------------------------------------------------
    # Reshape
    #
    # (samples, 2520)
    #
    # ->
    #
    # (samples, 20, 126)
    # --------------------------------------------------------

    sequence = X.reshape(
        -1,
        NUM_SAMPLED_FRAMES,
        FEATURES_PER_FRAME,
    )

    # --------------------------------------------------------
    # Frame-to-frame difference
    #
    # First frame has no previous frame.
    # Therefore first motion frame = zeros.
    # --------------------------------------------------------

    motion = np.diff(
        sequence,
        axis=1,
    )

    first_frame = np.zeros(
        (
            len(sequence),
            1,
            FEATURES_PER_FRAME,
        ),
        dtype=sequence.dtype,
    )

    motion = np.concatenate(
        [
            first_frame,
            motion,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Flatten original
    # --------------------------------------------------------

    original_flat = sequence.reshape(
        len(sequence),
        -1,
    )

    # --------------------------------------------------------
    # Flatten motion
    # --------------------------------------------------------

    motion_flat = motion.reshape(
        len(sequence),
        -1,
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = np.concatenate(
        [
            original_flat,
            motion_flat,
        ],
        axis=1,
    )

    if (
        combined.shape[1]
        != MOTION_FEATURES
    ):

        raise RuntimeError(
            f"Expected {MOTION_FEATURES} "
            f"motion features, got "
            f"{combined.shape[1]}"
        )

    return combined.astype(
        np.float32
    )


# ============================================================
# BUILD TEMPORAL FEATURES
#
# EXACT SAME TRANSFORMATION AS:
# train_temporal_features_rf.py
# ============================================================

def build_temporal_features(X):

    if X.ndim != 2:

        raise ValueError(
            f"Expected 2D input, "
            f"got {X.shape}"
        )

    if (
        X.shape[1]
        != ORIGINAL_FEATURES
    ):

        raise ValueError(
            f"Expected {ORIGINAL_FEATURES} "
            f"features, got "
            f"{X.shape[1]}"
        )

    X = X.reshape(
        -1,
        NUM_SAMPLED_FRAMES,
        FEATURES_PER_FRAME,
    )

    # --------------------------------------------------------
    # FRAME STATISTICS
    # --------------------------------------------------------

    mean = X.mean(
        axis=1
    )

    std = X.std(
        axis=1
    )

    minimum = X.min(
        axis=1
    )

    maximum = X.max(
        axis=1
    )

    value_range = (
        maximum
        - minimum
    )

    # --------------------------------------------------------
    # MOTION STATISTICS
    # --------------------------------------------------------

    diff = np.diff(
        X,
        axis=1,
    )

    diff_mean = diff.mean(
        axis=1
    )

    diff_std = diff.std(
        axis=1
    )

    diff_abs_mean = (
        np.abs(diff)
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            value_range,
            diff_mean,
            diff_std,
            diff_abs_mean,
        ],
        axis=1,
    )

    if (
        features.shape[1]
        != TEMPORAL_FEATURES
    ):

        raise RuntimeError(
            f"Expected {TEMPORAL_FEATURES} "
            f"temporal features, got "
            f"{features.shape[1]}"
        )

    return features.astype(
        np.float32
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    paths = {
        "Original RF":
            ORIGINAL_MODEL_PATH,

        "Motion RF":
            MOTION_MODEL_PATH,

        "Temporal RF":
            TEMPORAL_MODEL_PATH,
    }

    for name, path in paths.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} model not found:\n"
                f"{path}"
            )

    print(
        "\nLoading models..."
    )

    original_model = joblib.load(
        ORIGINAL_MODEL_PATH
    )

    motion_model = joblib.load(
        MOTION_MODEL_PATH
    )

    temporal_model = joblib.load(
        TEMPORAL_MODEL_PATH
    )

    print(
        "Models loaded successfully."
    )

    return (
        original_model,
        motion_model,
        temporal_model,
    )


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():

    if not (
        LABEL_MAPPING_PATH.exists()
    ):

        raise FileNotFoundError(
            "Label mapping not found:\n"
            f"{LABEL_MAPPING_PATH}"
        )

    mapping = pd.read_csv(
        LABEL_MAPPING_PATH
    )

    required = {
        "label_id",
        "label",
    }

    missing = (
        required
        - set(mapping.columns)
    )

    if missing:

        raise RuntimeError(
            f"Missing columns: "
            f"{missing}"
        )

    mapping = mapping.sort_values(
        "label_id"
    )

    id_to_label = dict(
        zip(
            mapping["label_id"],
            mapping["label"],
        )
    )

    return {
        int(k): str(v)
        for k, v
        in id_to_label.items()
    }


# ============================================================
# ALIGN MODEL PROBABILITIES
# ============================================================

def align_probabilities(
    model,
    probabilities,
    all_classes,
):

    aligned = np.zeros(
        len(all_classes),
        dtype=np.float64,
    )

    class_to_position = {
        int(class_id): index
        for index, class_id
        in enumerate(all_classes)
    }

    for position, class_id in enumerate(
        model.classes_
    ):

        class_id = int(
            class_id
        )

        if (
            class_id
            in class_to_position
        ):

            target_position = (
                class_to_position[
                    class_id
                ]
            )

            aligned[
                target_position
            ] = probabilities[
                position
            ]

    return aligned


# ============================================================
# PREDICT VIDEO
# ============================================================

def predict_video(video_path):

    print("\n" + "=" * 72)
    print(
        "       WEIGHTED ENSEMBLE "
        "VIDEO PREDICTION"
    )
    print("=" * 72)

    print(
        f"\nVideo:\n{video_path}"
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    (
        original_model,
        motion_model,
        temporal_model,
    ) = load_models()

    label_names = (
        load_label_mapping()
    )

    # ========================================================
    # EXTRACT ORIGINAL FEATURES
    # ========================================================

    print(
        "\nExtracting MediaPipe landmarks..."
    )

    with HandLandmarker.create_from_options(
        OPTIONS
    ) as landmarker:

        (
            original_features,
            total_frames,
            left_detections,
            right_detections,
        ) = extract_video_features(
            video_path,
            landmarker,
        )

    print(
        "\nFeature extraction complete:"
    )

    print(
        f"  Frames:           "
        f"{total_frames}"
    )

    print(
        f"  Left detections:  "
        f"{left_detections}"
    )

    print(
        f"  Right detections: "
        f"{right_detections}"
    )

    print(
        f"  Original features:"
        f" {len(original_features)}"
    )

    # ========================================================
    # PREPARE ORIGINAL INPUT
    # ========================================================

    X_original = (
        original_features
        .reshape(1, -1)
        .astype(np.float32)
    )

    # ========================================================
    # BUILD MOTION INPUT
    # ========================================================

    X_motion = (
        build_motion_features(
            X_original
        )
    )

    # ========================================================
    # BUILD TEMPORAL INPUT
    # ========================================================

    X_temporal = (
        build_temporal_features(
            X_original
        )
    )

    print(
        "\nModel feature shapes:"
    )

    print(
        f"  Original: "
        f"{X_original.shape}"
    )

    print(
        f"  Motion:   "
        f"{X_motion.shape}"
    )

    print(
        f"  Temporal: "
        f"{X_temporal.shape}"
    )

    # ========================================================
    # VALIDATE MODEL INPUT SIZES
    # ========================================================

    expected_original = getattr(
        original_model,
        "n_features_in_",
        None,
    )

    expected_motion = getattr(
        motion_model,
        "n_features_in_",
        None,
    )

    expected_temporal = getattr(
        temporal_model,
        "n_features_in_",
        None,
    )

    if (
        expected_original is not None
        and X_original.shape[1]
        != expected_original
    ):

        raise RuntimeError(
            "Original model feature mismatch:\n"
            f"Model expects: "
            f"{expected_original}\n"
            f"Input has: "
            f"{X_original.shape[1]}"
        )

    if (
        expected_motion is not None
        and X_motion.shape[1]
        != expected_motion
    ):

        raise RuntimeError(
            "Motion model feature mismatch:\n"
            f"Model expects: "
            f"{expected_motion}\n"
            f"Input has: "
            f"{X_motion.shape[1]}"
        )

    if (
        expected_temporal is not None
        and X_temporal.shape[1]
        != expected_temporal
    ):

        raise RuntimeError(
            "Temporal model feature mismatch:\n"
            f"Model expects: "
            f"{expected_temporal}\n"
            f"Input has: "
            f"{X_temporal.shape[1]}"
        )

    # ========================================================
    # PREDICT PROBABILITIES
    # ========================================================

    print(
        "\nRunning ensemble..."
    )

    original_proba = (
        original_model
        .predict_proba(
            X_original
        )[0]
    )

    motion_proba = (
        motion_model
        .predict_proba(
            X_motion
        )[0]
    )

    temporal_proba = (
        temporal_model
        .predict_proba(
            X_temporal
        )[0]
    )

    # ========================================================
    # GLOBAL CLASS ORDER
    # ========================================================

    all_classes = np.array(
        sorted(
            set(
                original_model.classes_
            )
            | set(
                motion_model.classes_
            )
            | set(
                temporal_model.classes_
            )
        )
    )

    # ========================================================
    # ALIGN PROBABILITIES
    # ========================================================

    original_aligned = (
        align_probabilities(
            original_model,
            original_proba,
            all_classes,
        )
    )

    motion_aligned = (
        align_probabilities(
            motion_model,
            motion_proba,
            all_classes,
        )
    )

    temporal_aligned = (
        align_probabilities(
            temporal_model,
            temporal_proba,
            all_classes,
        )
    )

    # ========================================================
    # WEIGHTED ENSEMBLE
    # ========================================================

    ensemble_probabilities = (
        WEIGHT_ORIGINAL
        * original_aligned

        + WEIGHT_MOTION
        * motion_aligned

        + WEIGHT_TEMPORAL
        * temporal_aligned
    )

    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    best_position = int(
        np.argmax(
            ensemble_probabilities
        )
    )

    predicted_id = int(
        all_classes[
            best_position
        ]
    )

    predicted_label = (
        label_names.get(
            predicted_id,
            f"class_{predicted_id}",
        )
    )

    confidence = float(
        ensemble_probabilities[
            best_position
        ]
    )

    # ========================================================
    # INDIVIDUAL MODEL PREDICTIONS
    # ========================================================

    original_id = int(
        original_model.predict(
            X_original
        )[0]
    )

    motion_id = int(
        motion_model.predict(
            X_motion
        )[0]
    )

    temporal_id = int(
        temporal_model.predict(
            X_temporal
        )[0]
    )

    # ========================================================
    # CONFIDENCE ANALYSIS
    # ========================================================

    ranked_positions = np.argsort(
        ensemble_probabilities
    )[::-1]

    second_best_position = int(
        ranked_positions[1]
    )

    second_best_id = int(
        all_classes[
            second_best_position
        ]
    )

    second_best_label = label_names.get(
        second_best_id,
        f"class_{second_best_id}",
    )

    second_best_probability = float(
        ensemble_probabilities[
            second_best_position
        ]
    )

    confidence_margin = float(
        confidence - second_best_probability
    )

    model_predictions = [
        original_id,
        motion_id,
        temporal_id,
    ]

    agreement_count = sum(
        prediction == predicted_id
        for prediction in model_predictions
    )

    agreement_status = (
        f"{agreement_count}/3 models agree"
    )

    if confidence >= 0.60:
        confidence_level = "High"
    elif confidence >= 0.30:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    # ========================================================
    # PREDICTION RELIABILITY
    # ========================================================

    is_confident = (
        confidence >= MIN_CONFIDENCE_ACCEPT
    )

    has_clear_margin = (
        confidence_margin >= MIN_MARGIN_ACCEPT
    )

    has_model_agreement = (
        agreement_count >= MIN_MODEL_AGREEMENT
    )

    is_reliable = (
        is_confident
        and (
            has_clear_margin
            or has_model_agreement
        )
    )

    if is_reliable:

        if (
            confidence >= 0.50
            and agreement_count == 3
        ):
            reliability_level = "High"

        elif (
            agreement_count >= 2
            or confidence_margin >= 0.15
        ):
            reliability_level = "Medium"

        else:
            reliability_level = "Low"

    else:
        reliability_level = "Uncertain"

    # ========================================================
    # PRINT INDIVIDUAL RESULTS
    # ========================================================

    print(
        "\nIndividual model predictions:"
    )

    print(
        f"  Original RF: "
        f"{label_names.get(original_id, original_id)}"
    )

    print(
        f"  Motion RF:   "
        f"{label_names.get(motion_id, motion_id)}"
    )

    print(
        f"  Temporal RF: "
        f"{label_names.get(temporal_id, temporal_id)}"
    )

    # ========================================================
    # TOP 5 ENSEMBLE PREDICTIONS
    # ========================================================

    print(
        "\nTop 5 ensemble predictions:"
    )

    for rank, position in enumerate(
        ranked_positions[:5],
        start=1,
    ):

        class_id = int(
            all_classes[position]
        )

        label = label_names.get(
            class_id,
            f"class_{class_id}",
        )

        probability = float(
            ensemble_probabilities[
                position
            ]
        )

        print(
            f"  {rank}. "
            f"{label:<20s} "
            f"{probability * 100:.2f}%"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "                  FINAL PREDICTION"
    )

    print(
        "=" * 72
    )

    print(
        f"\n  Sign:       "
        f"{predicted_label}"
    )

    print(
        f"  Class ID:   "
        f"{predicted_id}"
    )

    print(
        f"  Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        "\n"
        + "=" * 72
    )

    return {
      "class_id": predicted_id,
      "label": predicted_label,
      "confidence": confidence,
    }
    


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "  python -m "
            "src.ml.predict_ensemble_video "
            "<video_path>"
        )

        print(
            "\nExample:"
        )

        print(
            r'  python -m src.ml.predict_ensemble_video "D:\video.mp4"'
        )

        return

    video_path = Path(
        sys.argv[1]
    )

    predict_video(
        video_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()