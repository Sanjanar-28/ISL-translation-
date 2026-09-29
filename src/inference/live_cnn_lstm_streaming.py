import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd

from pathlib import Path
from collections import deque, Counter


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "data/ml_cnn_lstm/cnn_lstm_40frames_augmented.keras"
)

LABEL_MAPPING_PATH = Path(
    "data/ml_cnn_lstm/label_mapping.csv"
)

MEDIAPIPE_MODEL = Path(
    "models/hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 40

# Predict every N webcam frames
PREDICT_EVERY = 5

# Minimum confidence accepted
CONFIDENCE_THRESHOLD = 0.45

# Number of recent predictions used for stability
STABILITY_WINDOW = 5

# Require this many identical predictions
MIN_STABLE_COUNT = 3


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("       CONTINUOUS CNN + LSTM ISL PREDICTOR")
print("=" * 65)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")
print("Input shape:", model.input_shape)


# ============================================================
# LOAD LABELS
# ============================================================

label_df = pd.read_csv(
    LABEL_MAPPING_PATH
)

label_map = dict(
    zip(
        label_df["class_id"].astype(int),
        label_df["label"].astype(str)
    )
)

print(
    "Classes:",
    len(label_map)
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

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


# ============================================================
# EMPTY HAND
# ============================================================

def empty_hand():

    return np.zeros(
        63,
        dtype=np.float32
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(result):

    left_hand = empty_hand()

    right_hand = empty_hand()

    if result.hand_landmarks:

        for i, landmarks in enumerate(
            result.hand_landmarks
        ):

            coords = []

            for landmark in landmarks:

                coords.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            coords = np.asarray(
                coords,
                dtype=np.float32
            )

            if (
                result.handedness
                and i < len(result.handedness)
            ):

                hand_type = (
                    result
                    .handedness[i][0]
                    .category_name
                )

                if hand_type == "Left":

                    left_hand = coords

                elif hand_type == "Right":

                    right_hand = coords

    return np.concatenate([
        left_hand,
        right_hand
    ])


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sequence(sequence):

    # sequence:
    # (40,126)

    sequence = sequence.reshape(
        WINDOW_SIZE,
        42,
        3
    )

    normalized = np.zeros_like(
        sequence,
        dtype=np.float32
    )

    for i in range(WINDOW_SIZE):

        landmarks = sequence[i]

        origin = landmarks[0].copy()

        relative = landmarks - origin

        distances = np.linalg.norm(
            relative,
            axis=1
        )

        scale = np.max(
            distances
        )

        if scale > 1e-8:

            relative = relative / scale

        normalized[i] = relative

    # IMPORTANT:
    # return exactly (40,126)

    return normalized.reshape(
        WINDOW_SIZE,
        126
    )


# ============================================================
# CREATE LANDMARKER
# ============================================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(
            MEDIAPIPE_MODEL
        )
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)

landmarker = (
    HandLandmarker
    .create_from_options(options)
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    landmarker.close()

    raise SystemExit


# ============================================================
# BUFFERS
# ============================================================

frame_buffer = deque(
    maxlen=WINDOW_SIZE
)

prediction_history = deque(
    maxlen=STABILITY_WINDOW
)

frame_counter = 0

current_prediction = "Waiting..."

current_confidence = 0.0

stable_prediction = None


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 65)
print("CONTINUOUS MODE STARTED")
print("=" * 65)
print()
print("Perform a sign in front of the camera.")
print("The system continuously analyzes the latest 40 frames.")
print()
print("Q = Quit")
print("=" * 65)


try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not read webcam frame."
            )

            break

        # ----------------------------------------------------
        # ORIGINAL FRAME FOR MEDIAPIPE
        # ----------------------------------------------------

        original_frame = frame.copy()

        # ----------------------------------------------------
        # MIRRORED DISPLAY
        # ----------------------------------------------------

        display_frame = cv2.flip(
            original_frame,
            1
        )

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            original_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = (
            frame_counter * 33
        )

        result = (
            landmarker
            .detect_for_video(
                mp_image,
                timestamp_ms
            )
        )

        features = extract_features(
            result
        )

        frame_buffer.append(
            features
        )

        frame_counter += 1

        # ----------------------------------------------------
        # PREDICT WHEN BUFFER IS FULL
        # ----------------------------------------------------

        if (
            len(frame_buffer)
            >= WINDOW_SIZE
            and
            frame_counter % PREDICT_EVERY == 0
        ):

            sequence = np.asarray(
                frame_buffer,
                dtype=np.float32
            )

            sequence = normalize_sequence(
                sequence
            )

            model_input = np.expand_dims(
                sequence,
                axis=0
            )

            probabilities = (
                model.predict(
                    model_input,
                    verbose=0
                )[0]
            )

            class_id = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = float(
                probabilities[class_id]
            )

            label = label_map.get(
                class_id,
                str(class_id)
            )

            current_prediction = label

            current_confidence = (
                confidence * 100
            )

            # ------------------------------------------------
            # STABILITY
            # ------------------------------------------------

            if (
                confidence
                >= CONFIDENCE_THRESHOLD
            ):

                prediction_history.append(
                    label
                )

                counts = Counter(
                    prediction_history
                )

                most_common_label, count = (
                    counts.most_common(1)[0]
                )

                if count >= MIN_STABLE_COUNT:

                    stable_prediction = (
                        most_common_label
                    )

            # ------------------------------------------------
            # PRINT OCCASIONAL RESULT
            # ------------------------------------------------

            if (
                confidence
                >= CONFIDENCE_THRESHOLD
            ):

                print(
                    f"Prediction: "
                    f"{label:<18} "
                    f"{confidence * 100:5.1f}%"
                )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            display_frame,
            "CONTINUOUS ISL",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Frames: {len(frame_buffer)}/40",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Current: {current_prediction}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Confidence: {current_confidence:.1f}%",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        if stable_prediction:

            cv2.putText(
                display_frame,
                f"FINAL: {stable_prediction}",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (0, 255, 0),
                3
            )

        cv2.imshow(
            "Continuous CNN + LSTM ISL",
            display_frame
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


finally:

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()

    print()
    print("Webcam closed.")


print("Done.")