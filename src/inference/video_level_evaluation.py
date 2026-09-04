from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "data/ml_normalized/tuned_random_forest.joblib"
)

METADATA_PATH = Path(
    "data/video_metadata_split.csv"
)

LABEL_MAPPING_PATH = Path(
    "data/ml/label_mapping.csv"
)

HAND_MODEL_PATH = Path(
    "models/hand_landmarker.task"
)

DATASET_ROOT = Path(
    r"D:\Amrita SLR Dataset\sign project"
)

OUTPUT_PATH = Path(
    "data/ml_normalized/true_video_level_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

N_FRAMES = 20

LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3

FEATURES_PER_HAND = 21 * 3
FEATURES_PER_FRAME = 126

TOTAL_FEATURES = 20 * 126


# ============================================================
# CREATE MEDIAPIPE HAND LANDMARKER
# ============================================================

def create_hand_landmarker():

    if not HAND_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Hand model not found:\n"
            f"{HAND_MODEL_PATH.resolve()}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(
            HAND_MODEL_PATH.resolve()
        )
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


# ============================================================
# EXTRACT ONE FRAME
# ============================================================

def extract_frame_features(
    detector,
    frame
):
    """
    Return exactly 126 values.

    Layout:

    0   - 62   = left hand
    63  - 125  = right hand

    Each hand:
        21 landmarks × XYZ
    """

    left = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )

    right = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe Tasks image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(
        mp_image
    )

    if result.hand_landmarks:

        for hand_idx, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            if hand_idx >= len(
                result.handedness
            ):
                continue

            handedness = (
                result.handedness[
                    hand_idx
                ][0].category_name
            )

            values = []

            for landmark in hand_landmarks:

                values.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            values = np.asarray(
                values,
                dtype=np.float32
            )

            if handedness.lower() == "left":

                left[:] = values

            elif handedness.lower() == "right":

                right[:] = values

    return np.concatenate(
        [
            left,
            right
        ]
    )


# ============================================================
# EXTRACT VIDEO FEATURES
# ============================================================

def extract_video_features(
    detector,
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if frame_count <= 0:

        cap.release()

        raise RuntimeError(
            "Video has zero frames."
        )

    # --------------------------------------------------------
    # Same sampling strategy:
    # exactly 20 frames across entire video
    # --------------------------------------------------------

    frame_indices = np.linspace(
        0,
        frame_count - 1,
        N_FRAMES
    ).astype(int)

    features = []

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:

            print(
                f"    Warning: "
                f"could not read frame "
                f"{frame_index}"
            )

            features.append(
                np.zeros(
                    FEATURES_PER_FRAME,
                    dtype=np.float32
                )
            )

            continue

        frame_features = extract_frame_features(
            detector,
            frame
        )

        features.append(
            frame_features
        )

    cap.release()

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.shape != (
        N_FRAMES,
        FEATURES_PER_FRAME
    ):

        raise RuntimeError(
            f"Unexpected feature shape: "
            f"{features.shape}"
        )

    return features


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(
    features
):
    """
    EXACT normalization used during training.

    Input:
        (20, 126)

    Reshape:
        (20, 42, 3)

    For each frame:

        origin = landmark 0

        relative = landmarks - origin

        scale = maximum landmark distance

        normalized = relative / scale
    """

    X = features.copy()

    X = X.reshape(
        N_FRAMES,
        42,
        3
    )

    normalized = np.zeros_like(
        X,
        dtype=np.float32
    )

    for frame_idx in range(
        N_FRAMES
    ):

        landmarks = X[
            frame_idx
        ]

        origin = landmarks[
            0
        ].copy()

        relative = (
            landmarks - origin
        )

        distances = np.linalg.norm(
            relative,
            axis=1
        )

        scale = np.max(
            distances
        )

        if scale > 1e-8:

            relative = (
                relative / scale
            )

        normalized[
            frame_idx
        ] = relative

    return normalized.reshape(
        N_FRAMES,
        FEATURES_PER_FRAME
    )


# ============================================================
# FIND VIDEO
# ============================================================

def find_video(
    label,
    video_name
):

    # --------------------------------------------------------
    # Search recursively.
    # This handles folders such as:
    #
    # 1.bird
    # 2.black
    # ...
    # --------------------------------------------------------

    matches = list(
        DATASET_ROOT.rglob(
            video_name
        )
    )

    if not matches:

        return None

    # Prefer a matching label folder.
    for path in matches:

        parent = path.parent.name.lower()

        if label.lower() in parent:

            return path

    return matches[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("             TRUE VIDEO-LEVEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded."
    )

    print(
        f"Model classes: "
        f"{len(model.classes_)}"
    )

    # --------------------------------------------------------
    # LABEL MAPPING
    # --------------------------------------------------------

    label_mapping = pd.read_csv(
        LABEL_MAPPING_PATH
    )

    id_to_label = {}

    for _, row in label_mapping.iterrows():

        id_to_label[
            int(row["label_id"])
        ] = str(
            row["label"]
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    print(
        "\nLoading metadata..."
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata rows: "
        f"{len(metadata)}"
    )

    print(
        "Metadata columns:"
    )

    print(
        list(metadata.columns)
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
            f"Missing metadata columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # TEST SET
    # --------------------------------------------------------

    test_metadata = metadata[
        metadata["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    test_metadata = test_metadata.reset_index(
        drop=True
    )

    print(
        f"\nTest videos: "
        f"{len(test_metadata)}"
    )

    if len(test_metadata) != 130:

        print(
            "\nWARNING:"
        )

        print(
            "Expected 130 test videos."
        )

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    print(
        "\nLoading hand landmark detector..."
    )

    detector = create_hand_landmarker()

    print(
        "Hand landmark detector loaded."
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    records = []

    correct = 0
    successful = 0
    failed = 0

    total = len(
        test_metadata
    )

    print(
        "\n" + "-" * 70
    )

    for idx, row in test_metadata.iterrows():

        number = idx + 1

        label = str(
            row["label"]
        )

        video_name = str(
            row["video_name"]
        )

        print(
            f"\n[{number}/{total}] "
            f"{label}/{video_name}"
        )

        # ----------------------------------------------------
        # Find actual video
        # ----------------------------------------------------

        video_path = find_video(
            label,
            video_name
        )

        if video_path is None:

            print(
                "  ERROR: Video not found"
            )

            failed += 1

            records.append({

                "video_name":
                    video_name,

                "video_path":
                    "",

                "actual_label":
                    label,

                "actual_class_id":
                    int(row["class_id"]),

                "predicted_label":
                    "",

                "predicted_class_id":
                    "",

                "confidence":
                    "",

                "correct":
                    False,

                "status":
                    "VIDEO_NOT_FOUND"
            })

            continue

        print(
            f"  Path: {video_path}"
        )

        try:

            # ------------------------------------------------
            # Extract
            # ------------------------------------------------

            features = extract_video_features(
                detector,
                video_path
            )

            print(
                f"  Raw feature shape: "
                f"{features.shape}"
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            normalized = normalize_features(
                features
            )

            print(
                f"  Normalized shape: "
                f"{normalized.shape}"
            )

            # ------------------------------------------------
            # Flatten
            # ------------------------------------------------

            X = normalized.reshape(
                1,
                TOTAL_FEATURES
            )

            print(
                f"  Final shape: "
                f"{X.shape}"
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                X
            )[0]

            predicted_class_id = int(
                prediction
            )

            predicted_label = (
                id_to_label.get(
                    predicted_class_id,
                    str(predicted_class_id)
                )
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            probabilities = (
                model.predict_proba(X)[0]
            )

            confidence = float(
                np.max(
                    probabilities
                )
            )

            # ------------------------------------------------
            # Actual
            # ------------------------------------------------

            actual_class_id = int(
                row["class_id"]
            )

            # IMPORTANT:
            #
            # label_mapping uses:
            # 0 = Music
            # 1 = NAME
            # ...
            #
            # metadata class_id appears to be
            # 1 = bird, etc.
            #
            # Therefore compare LABELS, not raw
            # metadata class_id values.
            # ------------------------------------------------

            actual_label = label

            is_correct = (
                predicted_label.lower()
                ==
                actual_label.lower()
            )

            if is_correct:

                correct += 1

            successful += 1

            print(
                f"  Actual:      "
                f"{actual_label}"
            )

            print(
                f"  Predicted:   "
                f"{predicted_label}"
            )

            print(
                f"  Confidence:  "
                f"{confidence * 100:.2f}%"
            )

            print(
                f"  Result:      "
                f"{'CORRECT' if is_correct else 'WRONG'}"
            )

            records.append({

                "video_name":
                    video_name,

                "video_path":
                    str(video_path),

                "actual_label":
                    actual_label,

                "actual_class_id":
                    actual_class_id,

                "predicted_label":
                    predicted_label,

                "predicted_class_id":
                    predicted_class_id,

                "confidence":
                    confidence,

                "correct":
                    is_correct,

                "status":
                    "SUCCESS"
            })

        except Exception as e:

            failed += 1

            print(
                f"  ERROR: {e}"
            )

            records.append({

                "video_name":
                    video_name,

                "video_path":
                    str(video_path),

                "actual_label":
                    label,

                "actual_class_id":
                    int(row["class_id"]),

                "predicted_label":
                    "",

                "predicted_class_id":
                    "",

                "confidence":
                    "",

                "correct":
                    False,

                "status":
                    f"ERROR: {e}"
            })

    # --------------------------------------------------------
    # Close detector
    # --------------------------------------------------------

    detector.close()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        records
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal test videos:       "
        f"{total}"
    )

    print(
        f"Successfully processed:  "
        f"{successful}"
    )

    print(
        f"Failed:                  "
        f"{failed}"
    )

    print(
        f"Correct predictions:     "
        f"{correct}"
    )

    if successful > 0:

        accuracy = (
            correct /
            successful *
            100
        )

        print(
            f"Inference accuracy:      "
            f"{accuracy:.2f}%"
        )

        print(
            f"\nWrong predictions:       "
            f"{successful - correct}"
        )

    else:

        print(
            "\nInference accuracy:      N/A"
        )

    # --------------------------------------------------------
    # CLASS ACCURACY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASS ACCURACY"
    )

    print(
        "=" * 70
    )

    successful_df = results_df[
        results_df["status"]
        == "SUCCESS"
    ]

    if len(successful_df) > 0:

        for class_label in sorted(
            successful_df[
                "actual_label"
            ].unique()
        ):

            class_df = successful_df[
                successful_df[
                    "actual_label"
                ] == class_label
            ]

            class_correct = int(
                class_df[
                    "correct"
                ].sum()
            )

            class_total = len(
                class_df
            )

            class_accuracy = (
                class_correct
                /
                class_total
                *
                100
            )

            print(
                f"{class_label:<18}"
                f"{class_correct}/{class_total}"
                f" {class_accuracy:.2f}%"
            )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH.resolve()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "VIDEO-LEVEL EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()