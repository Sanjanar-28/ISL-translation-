from pathlib import Path
import cv2
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = Path(
    r"D:\Amrita SLR Dataset\sign project\1.bird\0403.mp4"
)

MODEL_PATH = Path(
    "models/hand_landmarker.task"
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


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


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print()
print("=" * 55)
print("              MEDIAPIPE HAND TEST")
print("=" * 55)

print()
print("Video:")
print(f"  {VIDEO_PATH}")

print()
print("Video information:")
print(f"  FPS:          {fps:.2f}")
print(f"  Total frames: {total_frames}")


# ============================================================
# PROCESS VIDEO
# ============================================================

frames_processed = 0
frames_with_hands = 0
frames_without_hands = 0

total_hands = 0
max_hands = 0

missing_frames = []


with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames_processed += 1

        # ----------------------------------------------------
        # Convert OpenCV BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Convert frame to MediaPipe Image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ----------------------------------------------------
        # Timestamp
        #
        # MediaPipe VIDEO mode requires timestamps
        # in milliseconds and they must increase.
        # ----------------------------------------------------

        timestamp_ms = int(
            (frames_processed - 1) * 1000 / fps
        )

        # ----------------------------------------------------
        # Run hand detection
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # Count detected hands
        # ----------------------------------------------------

        hands_detected = len(
            result.hand_landmarks
        )

        total_hands += hands_detected

        max_hands = max(
            max_hands,
            hands_detected
        )

        if hands_detected > 0:

            frames_with_hands += 1

        else:

            frames_without_hands += 1

            missing_frames.append(
                frames_processed
            )


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()


# ============================================================
# CALCULATE STATISTICS
# ============================================================

if frames_processed > 0:

    detection_rate = (
        frames_with_hands /
        frames_processed
    ) * 100

    average_hands = (
        total_hands /
        frames_processed
    )

else:

    detection_rate = 0.0
    average_hands = 0.0


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 55)
print("              MEDIAPIPE TEST REPORT")
print("=" * 55)

print()
print("Video:")
print(f"  {VIDEO_PATH}")

print()
print("Video FPS:")
print(f"  {fps:.2f}")

print()
print("Frame statistics:")
print(f"  Frames processed:       {frames_processed}")
print(f"  Frames with hands:      {frames_with_hands}")
print(f"  Frames without hands:   {frames_without_hands}")

print()
print("Detection:")
print(
    f"  Hand detection rate:    "
    f"{detection_rate:.2f}%"
)

print()
print("Hand statistics:")
print(f"  Total hands detected:   {total_hands}")
print(f"  Average hands/frame:    {average_hands:.2f}")
print(f"  Maximum hands/frame:    {max_hands}")

print()
print("Missing frame numbers:")

if missing_frames:

    print(
        f"  {missing_frames}"
    )

else:

    print("  None")


print()
print("=" * 55)