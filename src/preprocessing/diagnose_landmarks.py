from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"D:\Amrita SLR Dataset\sign project"
)

MODEL_PATH = Path(
    "models/hand_landmarker.task"
)

NUM_SAMPLED_FRAMES = 20


# ============================================================
# TEST VIDEOS
# ============================================================
#
# Format:
# (dataset_folder, label, video_name)
#
# These are the lowest-detection videos from the TEST split.
# ============================================================

VIDEOS = [
    ("2.black", "black", "krishnarajblack.mp4"),
    ("29.nose", "nose", "manideepnose-1.mp4"),
    ("4.cow", "cow", "krishnarajcow_2.mp4"),
    ("16.eat", "eat", "manideepeat-1.mp4"),
    ("25.mother", "mother", "manideepmother-1.mp4"),
]


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

    # Allow up to two hands
    num_hands=2,

    # CURRENT PROJECT SETTINGS
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2,
)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, label):

    print("\n")
    print("=" * 70)
    print("VIDEO DIAGNOSTIC")
    print("=" * 70)

    print("\nLabel:")
    print(f"  {label}")

    print("\nVideo:")
    print(f"  {video_path}")

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print("\nERROR:")
        print("  Could not open video.")

        return

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    total_frames_property = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("\nVideo information:")
    print(f"  FPS: {fps:.2f}")
    print(
        f"  Frame count reported by OpenCV: "
        f"{total_frames_property}"
    )

    # --------------------------------------------------------
    # PER-FRAME DETECTION STORAGE
    # --------------------------------------------------------

    detected_per_frame = []

    left_per_frame = []

    right_per_frame = []

    frame_number = 0

    # --------------------------------------------------------
    # CREATE MEDIAPIPE LANDMARKER
    # --------------------------------------------------------

    try:

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                # ------------------------------------------------
                # BGR → RGB
                # ------------------------------------------------

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # ------------------------------------------------
                # VIDEO TIMESTAMP
                # ------------------------------------------------

                timestamp_ms = int(
                    ((frame_number - 1) / fps)
                    * 1000
                )

                # ------------------------------------------------
                # MEDIAPIPE
                # ------------------------------------------------

                result = (
                    landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                left_found = False
                right_found = False

                # ------------------------------------------------
                # CHECK HANDS
                # ------------------------------------------------

                for hand_index, _ in enumerate(
                    result.hand_landmarks
                ):

                    handedness = (
                        result.handedness[
                            hand_index
                        ][0]
                    )

                    hand_label = (
                        handedness.category_name
                    )

                    if hand_label == "Left":

                        left_found = True

                    elif hand_label == "Right":

                        right_found = True

                # ------------------------------------------------
                # STORE FRAME RESULT
                # ------------------------------------------------

                left_per_frame.append(
                    left_found
                )

                right_per_frame.append(
                    right_found
                )

                detected_per_frame.append(
                    left_found or right_found
                )

    finally:

        cap.release()

    # ============================================================
    # RAW DETECTION STATISTICS
    # ============================================================

    actual_frames = len(
        detected_per_frame
    )

    if actual_frames == 0:

        print("\nERROR:")
        print("  No frames were read.")

        return

    detected_frames = sum(
        detected_per_frame
    )

    left_count = sum(
        left_per_frame
    )

    right_count = sum(
        right_per_frame
    )

    frames_without_hands = (
        actual_frames
        - detected_frames
    )

    detection_rate = (
        detected_frames
        / actual_frames
        * 100
    )

    # ============================================================
    # 20-FRAME SAMPLING
    # ============================================================

    indices = np.linspace(
        0,
        actual_frames - 1,
        NUM_SAMPLED_FRAMES,
        dtype=int
    )

    sampled_detection = [
        detected_per_frame[i]
        for i in indices
    ]

    sampled_left = [
        left_per_frame[i]
        for i in indices
    ]

    sampled_right = [
        right_per_frame[i]
        for i in indices
    ]

    sampled_detected_count = sum(
        sampled_detection
    )

    sampled_left_count = sum(
        sampled_left
    )

    sampled_right_count = sum(
        sampled_right
    )

    sampled_missing_count = (
        NUM_SAMPLED_FRAMES
        - sampled_detected_count
    )

    sampled_detection_rate = (
        sampled_detected_count
        / NUM_SAMPLED_FRAMES
        * 100
    )

    # ============================================================
    # PRINT RAW RESULTS
    # ============================================================

    print("\n")
    print("RAW MEDIAPIPE DETECTION:")

    print(
        f"  Frames processed:       "
        f"{actual_frames}"
    )

    print(
        f"  Frames with hands:      "
        f"{detected_frames}"
    )

    print(
        f"  Frames without hands:   "
        f"{frames_without_hands}"
    )

    print(
        f"  Detection rate:         "
        f"{detection_rate:.2f}%"
    )

    # ============================================================
    # HAND STATISTICS
    # ============================================================

    print("\n")
    print("HAND DETECTION:")

    print(
        f"  Left detections:        "
        f"{left_count}"
    )

    print(
        f"  Right detections:       "
        f"{right_count}"
    )

    print(
        f"  Total hand detections:  "
        f"{left_count + right_count}"
    )

    # ============================================================
    # SAMPLING RESULTS
    # ============================================================

    print("\n")
    print("20-FRAME SAMPLING:")

    print(
        f"  Sampled frames:         "
        f"{NUM_SAMPLED_FRAMES}"
    )

    print(
        f"  Sampled frames with hands:"
        f" {sampled_detected_count}"
    )

    print(
        f"  Sampled frames without hands:"
        f" {sampled_missing_count}"
    )

    print(
        f"  Sampled detection rate:  "
        f"{sampled_detection_rate:.2f}%"
    )

    print(
        f"  Sampled left detections: "
        f"{sampled_left_count}"
    )

    print(
        f"  Sampled right detections:"
        f" {sampled_right_count}"
    )

    # ============================================================
    # SAMPLED FRAME DETAILS
    # ============================================================

    print("\n")
    print("SAMPLED FRAME INDICES:")

    for sample_number, frame_index in enumerate(
        indices
    ):

        status = (
            "HAND"
            if sampled_detection[sample_number]
            else "NO HAND"
        )

        left_status = (
            "L"
            if sampled_left[sample_number]
            else "-"
        )

        right_status = (
            "R"
            if sampled_right[sample_number]
            else "-"
        )

        print(
            f"  sample {sample_number + 1:02d}: "
            f"frame {frame_index + 1:03d} "
            f"-> {status} "
            f"[{left_status}{right_status}]"
        )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("       LANDMARK EXTRACTION DIAGNOSTIC")
    print("=" * 70)

    print("\nMediaPipe thresholds:")
    print("  Detection confidence: 0.2")
    print("  Presence confidence:  0.2")
    print("  Tracking confidence:  0.2")

    print("\nVideos to test:")

    for folder, label, video_name in VIDEOS:

        print(
            f"  {label:15s} "
            f"{folder}\\{video_name}"
        )

    # ========================================================
    # PROCESS EACH VIDEO
    # ========================================================

    for folder, label, video_name in VIDEOS:

        video_path = (
            DATASET_ROOT
            / folder
            / video_name
        )

        if not video_path.exists():

            print("\n")
            print("=" * 70)
            print("VIDEO NOT FOUND")
            print("=" * 70)

            print(
                f"\n{video_path}"
            )

            continue

        process_video(
            video_path,
            label
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("              DIAGNOSTIC COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()