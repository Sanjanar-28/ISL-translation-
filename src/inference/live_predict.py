from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "data/ml_normalized/normalized_random_forest.joblib"
)

LABEL_PATH = Path(
    "data/ml_normalized/normalized_label_mapping.csv"
)

MEDIAPIPE_MODEL = Path(
    "models/hand_landmarker.task"
)

TARGET_FRAMES = 20

LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3

FEATURES_PER_HAND = (
    LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK
)

FEATURES_PER_FRAME = FEATURES_PER_HAND * 2

EXPECTED_FEATURES = (
    TARGET_FRAMES * FEATURES_PER_FRAME
)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    MODEL_PATH,
    LABEL_PATH,
    MEDIAPIPE_MODEL
]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Random Forest...")

model = joblib.load(MODEL_PATH)

print(
    f"Model expects {model.n_features_in_} features"
)

if model.n_features_in_ != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Model expects {model.n_features_in_}, "
        f"but this program creates {EXPECTED_FEATURES}"
    )


# ============================================================
# LOAD LABELS
# ============================================================

label_df = pd.read_csv(LABEL_PATH)

label_mapping = dict(
    zip(
        label_df["class_id"].astype(int),
        label_df["label"].astype(str)
    )
)

print(
    f"Loaded {len(label_mapping)} labels"
)


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

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MEDIAPIPE_MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# EMPTY HAND
# ============================================================

def empty_hand():

    return np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )


# ============================================================
# EXTRACT HAND
# ============================================================

def extract_hand_features(hand_landmarks):

    features = []

    for landmark in hand_landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# EXTRACT ONE FRAME
# ============================================================

def extract_frame_features(result):

    left_hand = empty_hand()
    right_hand = empty_hand()

    for hand_index, landmarks in enumerate(
        result.hand_landmarks
    ):

        handedness = (
            result.handedness[hand_index][0]
        )

        hand_label = handedness.category_name

        features = extract_hand_features(
            landmarks
        )

        if hand_label == "Left":

            left_hand = features

        elif hand_label == "Right":

            right_hand = features

    return np.concatenate([
        left_hand,
        right_hand
    ])


# ============================================================
# NORMALIZE FRAME
# ============================================================

def normalize_frame(frame_features):

    landmarks = frame_features.reshape(
        42,
        3
    )

    origin = landmarks[0].copy()

    relative = landmarks - origin

    distances = np.linalg.norm(
        relative,
        axis=1
    )

    scale = np.max(distances)

    if scale > 1e-8:

        relative = relative / scale

    return relative.reshape(
        -1
    ).astype(
        np.float32
    )


# ============================================================
# SAMPLE FRAMES
# ============================================================

def sample_frames(frames, target_count=20):

    total = len(frames)

    if total < target_count:

        raise ValueError(
            f"Only {total} frames captured. "
            f"Need at least {target_count}."
        )

    # Uniformly select frames across
    # the entire gesture.

    indices = np.linspace(
        0,
        total - 1,
        target_count
    ).astype(int)

    return [
        frames[i]
        for i in indices
    ]


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(frames):

    selected_frames = sample_frames(
        frames,
        TARGET_FRAMES
    )

    normalized_frames = []

    for frame in selected_frames:

        normalized = normalize_frame(
            frame
        )

        normalized_frames.append(
            normalized
        )

    features = np.concatenate(
        normalized_frames
    )

    if len(features) != EXPECTED_FEATURES:

        raise RuntimeError(
            f"Generated {len(features)} features "
            f"instead of {EXPECTED_FEATURES}"
        )

    return features.reshape(
        1,
        EXPECTED_FEATURES
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_gesture(frames):

    model_input = prepare_model_input(
        frames
    )

    prediction = model.predict(
        model_input
    )[0]

    class_id = int(prediction)

    label = label_mapping.get(
        class_id,
        f"class_{class_id}"
    )

    # Random Forest probability
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            model_input
        )[0]

        confidence = float(
            np.max(probabilities)
        ) * 100

    return label, confidence


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# STATE
# ============================================================

recording = False

captured_frames = []

frame_number = 0

result_text = "Press SPACE to start"

confidence_text = ""

print()
print("=" * 60)
print("             ISL GESTURE PREDICTOR")
print("=" * 60)
print()
print("SPACE = Start / Stop recording")
print("Q     = Quit")
print()
print("Perform ONE sign after pressing SPACE.")
print("Press SPACE again when finished.")
print()


# ============================================================
# MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:

            print("Could not read webcam.")
            break

        frame_number += 1

        display_frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (frame_number / 30.0) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # Extract current frame
        # ----------------------------------------------------

        frame_features = extract_frame_features(
            result
        )

        # ----------------------------------------------------
        # RECORDING
        # ----------------------------------------------------

        if recording:

            captured_frames.append(
                frame_features
            )

            cv2.putText(
                display_frame,
                "RECORDING...",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            cv2.putText(
                display_frame,
                f"Frames: {len(captured_frames)}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                display_frame,
                result_text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            if confidence_text:

                cv2.putText(
                    display_frame,
                    confidence_text,
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

        # ----------------------------------------------------
        # INSTRUCTIONS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "SPACE: Start/Stop    Q: Quit",
            (20, 700),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ISL Gesture Predictor",
            display_frame
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # SPACE
        # ====================================================

        if key == ord(" "):

            # -----------------------------------------------
            # START
            # -----------------------------------------------

            if not recording:

                captured_frames = []

                result_text = "Perform your sign..."

                confidence_text = ""

                recording = True

                print()
                print(
                    "Recording started..."
                )

            # -----------------------------------------------
            # STOP
            # -----------------------------------------------

            else:

                recording = False

                total_frames = len(
                    captured_frames
                )

                print()
                print(
                    f"Recording stopped."
                )

                print(
                    f"Captured frames: "
                    f"{total_frames}"
                )

                if total_frames < TARGET_FRAMES:

                    result_text = (
                        "Too short - try again"
                    )

                    confidence_text = ""

                    print(
                        f"Need at least "
                        f"{TARGET_FRAMES} frames."
                    )

                else:

                    try:

                        print(
                            "Running prediction..."
                        )

                        label, confidence = (
                            predict_gesture(
                                captured_frames
                            )
                        )

                        result_text = (
                            f"Prediction: {label}"
                        )

                        if confidence is not None:

                            confidence_text = (
                                f"Confidence: "
                                f"{confidence:.1f}%"
                            )

                        else:

                            confidence_text = ""

                        print(
                            f"Prediction: {label}"
                        )

                        if confidence is not None:

                            print(
                                f"Confidence: "
                                f"{confidence:.1f}%"
                            )

                    except Exception as error:

                        result_text = (
                            "Prediction failed"
                        )

                        confidence_text = ""

                        print(
                            "\nPrediction error:"
                        )

                        print(error)

        # ====================================================
        # QUIT
        # ====================================================

        elif key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("=" * 60)
print("              CAMERA STOPPED")
print("=" * 60)