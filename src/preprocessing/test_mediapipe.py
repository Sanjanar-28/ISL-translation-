from pathlib import Path

import cv2
import mediapipe as mp


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = Path(
    r"D:\Amrita SLR Dataset\sign project\2.black\krishnarajblack.mp4"
)

MODEL_PATH = Path(
    "models/hand_landmarker.task"
)

DEBUG_DIR = Path("debug_frames")
DEBUG_DIR.mkdir(exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    # IMPORTANT:
    # IMAGE mode = each frame is detected independently
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,

    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2,
)


# ============================================================
# CHECK FILES
# ============================================================

if not VIDEO_PATH.exists():
    raise RuntimeError(
        f"Video does not exist:\n{VIDEO_PATH}"
    )

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model does not exist:\n{MODEL_PATH}"
    )


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0


reported_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


# ============================================================
# STATISTICS
# ============================================================

frames_processed = 0
frames_with_hands = 0
frames_without_hands = 0

total_hands = 0

left_detections = 0
right_detections = 0

missing_frames = []


# ============================================================
# IMAGE MODE PROCESSING
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_index = frames_processed
        frames_processed += 1

        # ----------------------------------------------------
        # IMPORTANT:
        # NO RESIZING
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # IMAGE MODE
        # Each frame is processed independently.
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )

        number_of_hands = len(
            result.hand_landmarks
        )

        total_hands += number_of_hands

        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

        if number_of_hands == 0:

            frames_without_hands += 1

            missing_frames.append(
                frame_index + 1
            )

        else:

            frames_with_hands += 1

        # ----------------------------------------------------
        # HANDEDNESS
        # ----------------------------------------------------

        for hand_index in range(
            number_of_hands
        ):

            if (
                hand_index
                >= len(result.handedness)
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

            if label == "Left":

                left_detections += 1

            elif label == "Right":

                right_detections += 1

        # ----------------------------------------------------
        # SAVE DEBUG FRAMES
        # ----------------------------------------------------

        if frame_index in (
            0,
            reported_frames // 2,
            reported_frames - 1,
        ):

            debug_path = (
                DEBUG_DIR
                / f"black_image_mode_{frame_index}.jpg"
            )

            cv2.imwrite(
                str(debug_path),
                frame
            )

            print(
                f"Saved: {debug_path}"
            )


# ============================================================
# RELEASE
# ============================================================

cap.release()


# ============================================================
# RESULTS
# ============================================================

if frames_processed == 0:

    raise RuntimeError(
        "No frames were processed."
    )


detection_rate = (
    frames_with_hands
    / frames_processed
    * 100
)

average_hands = (
    total_hands
    / frames_processed
)


# ============================================================
# REPORT
# ============================================================

print("\n")

print("=" * 55)
print("       MEDIAPIPE IMAGE MODE TEST")
print("=" * 55)

print("\nVideo:")
print(f"  {VIDEO_PATH}")

print("\nProcessing:")
print("  Running mode: IMAGE")
print("  Resolution: ORIGINAL")
print("  Detection confidence: 0.2")
print("  Presence confidence:  0.2")
print("  Tracking confidence:  0.2")

print("\nVideo information:")
print(f"  FPS: {fps:.2f}")
print(f"  Frames reported: {reported_frames}")

print("\nFrame statistics:")
print(f"  Frames processed:       {frames_processed}")
print(f"  Frames with hands:      {frames_with_hands}")
print(f"  Frames without hands:   {frames_without_hands}")

print("\nDetection:")
print(f"  Hand detection rate:    {detection_rate:.2f}%")

print("\nHand statistics:")
print(f"  Total hands detected:   {total_hands}")
print(f"  Left detections:        {left_detections}")
print(f"  Right detections:       {right_detections}")
print(f"  Average hands/frame:    {average_hands:.2f}")

print("\nMissing frame numbers:")

if missing_frames:
    print(f"  {missing_frames}")
else:
    print("  None")

print("\n")
print("=" * 55)
print("             TEST COMPLETE")
print("=" * 55)