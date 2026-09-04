from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

VIDEO_PATH = Path(
    r"D:\Amrita SLR Dataset\sign project\1.bird\0403(77).mp4"
)

MODEL_PATH = Path(
    "models/hand_landmarker.task"
)

OUTPUT_PATH = Path(
    "data/test_landmarks.csv"
)


# ==================================================
# CHECK FILES
# ==================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ==================================================
# MEDIAPIPE SETUP
# ==================================================

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


# ==================================================
# HELPER FUNCTION
# ==================================================

def empty_hand_features():

    """
    Creates 63 empty values for one hand.

    21 landmarks × 3 coordinates
    = 63 values
    """

    return [None] * 63


def extract_hand_features(hand_landmarks):

    """
    Extract x, y, z coordinates
    from 21 MediaPipe landmarks.
    """

    features = []

    for landmark in hand_landmarks:

        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return features


# ==================================================
# OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


# ==================================================
# STORAGE
# ==================================================

records = []

frame_number = 0


# ==================================================
# PROCESS VIDEO
# ==================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ------------------------------------------
        # Convert BGR → RGB
        # ------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ------------------------------------------
        # MediaPipe Image
        # ------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ------------------------------------------
        # Timestamp
        # ------------------------------------------

        timestamp_ms = int(
            ((frame_number - 1) / fps) * 1000
        )

        # ------------------------------------------
        # Detection
        # ------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ------------------------------------------
        # Initialize both hands
        # ------------------------------------------

        left_hand = empty_hand_features()
        right_hand = empty_hand_features()

        left_detected = False
        right_detected = False

        # ------------------------------------------
        # Extract landmarks
        # ------------------------------------------

        for hand_index, landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = (
                result.handedness[hand_index][0]
            )

            hand_label = (
                handedness.category_name
            )

            features = extract_hand_features(
                landmarks
            )

            if hand_label == "Left":

                left_hand = features
                left_detected = True

            elif hand_label == "Right":

                right_hand = features
                right_detected = True

        # ------------------------------------------
        # Create record
        # ------------------------------------------

        record = {
            "video": VIDEO_PATH.name,
            "frame": frame_number,
            "left_detected": left_detected,
            "right_detected": right_detected,
        }

        # ------------------------------------------
        # Add left-hand features
        # ------------------------------------------

        for i, value in enumerate(left_hand):

            record[
                f"left_{i}"
            ] = value

        # ------------------------------------------
        # Add right-hand features
        # ------------------------------------------

        for i, value in enumerate(right_hand):

            record[
                f"right_{i}"
            ] = value

        records.append(record)


# ==================================================
# CLEANUP
# ==================================================

cap.release()


# ==================================================
# SAVE CSV
# ==================================================

df = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==================================================
# REPORT
# ==================================================

print("\n")
print("=" * 60)
print("          LANDMARK EXTRACTION TEST")
print("=" * 60)

print(
    f"\nVideo: {VIDEO_PATH.name}"
)

print(
    f"Frames extracted: {len(df)}"
)

print(
    f"Left-hand detections: "
    f"{df['left_detected'].sum()}"
)

print(
    f"Right-hand detections: "
    f"{df['right_detected'].sum()}"
)

print(
    f"\nFeatures per frame: "
    f"{len(df.columns) - 4}"
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)

print("\n" + "=" * 60)