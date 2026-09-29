import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(
    r"D:\Amrita SLR Dataset\sign project"
)

OUTPUT_DIR = Path(
    "data/ml_cnn_lstm"
)

MEDIAPIPE_MODEL = Path(
    "models/hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_FRAMES = 40

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "1.bird",
    "10.youareperfect",
    "11.monsoon",
    "12.afternoon",
    "13.angry",
    "14.bad",
    "15.boy",
    "16.eat",
    "17.friend",
    "18.drinking",
    "19.girl",
    "2.black",
    "20.brother",
    "21.good",
    "22.father",
    "23.evening",
    "24.help",
    "25.mother",
    "26.NAME",
    "27.Night",
    "28.Music",
    "29.nose",
    "3.cat",
    "30.sleep",
    "31.sit",
    "32.sorry",
    "33.stand",
    "34.stop",
    "35.student",
    "36.study",
    "37.teacher",
    "38.thankyou",
    "39.today",
    "4.cow",
    "40.tommorow",
    "41.welcome",
    "42.work",
    "43.yesterday",
    "44.teeth",
    "45.hand",
    "46.write",
    "47.umberlla",
    "48.ring",
    "5.dog",
    "50.power",
    "6.fish",
    "7.goodmorning",
    "8.grey",
    "9.hello"
]


# ============================================================
# MEDIAPIPE SETUP
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
# EXTRACT LANDMARKS
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

                handedness = (
                    result
                    .handedness[i][0]
                    .category_name
                )

                if handedness == "Left":

                    left_hand = coords

                elif handedness == "Right":

                    right_hand = coords

    return np.concatenate([
        left_hand,
        right_hand
    ])


# ============================================================
# NORMALIZE 40 FRAMES
# ============================================================

def normalize_sequence(sequence):

    # sequence shape:
    #
    # (40, 126)
    #
    # Convert:
    #
    # (40, 42, 3)

    sequence = sequence.reshape(
        TARGET_FRAMES,
        42,
        3
    )

    normalized = np.zeros_like(
        sequence,
        dtype=np.float32
    )

    for f in range(TARGET_FRAMES):

        landmarks = sequence[f]

        origin = landmarks[0].copy()

        relative = landmarks - origin

        distances = np.linalg.norm(
            relative,
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-8:

            relative = relative / scale

        normalized[f] = relative

    return normalized.reshape(
        TARGET_FRAMES,
        126
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    # IMPORTANT:
    # Create a NEW landmarker for every video
    # so timestamps restart correctly.

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
        mp.tasks.vision.HandLandmarker
        .create_from_options(options)
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        landmarker.close()

        return None


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0


    frames = []

    frame_index = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                break

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

            timestamp_ms = int(
                frame_index *
                1000 /
                fps
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

            frames.append(
                features
            )

            frame_index += 1

    finally:

        cap.release()

        landmarker.close()


    if len(frames) < TARGET_FRAMES:

        return None


    frames = np.asarray(
        frames,
        dtype=np.float32
    )


    # ========================================================
    # UNIFORM 40 FRAME SAMPLING
    # ========================================================

    indices = np.linspace(
        0,
        len(frames) - 1,
        TARGET_FRAMES
    ).astype(int)

    sampled = frames[
        indices
    ]


    # ========================================================
    # NORMALIZE
    # ========================================================

    normalized = normalize_sequence(
        sampled
    )


    return normalized


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "       EXTRACTING 40-FRAME CNN-LSTM DATASET"
    )

    print("=" * 70)

    X = []

    y = []

    metadata = []

    total_videos = 0

    successful = 0

    failed = 0


    # ========================================================
    # PROCESS CLASSES
    # ========================================================

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        class_dir = (
            DATASET_DIR /
            class_name
        )

        if not class_dir.exists():

            print(
                f"\nWARNING: Missing folder: "
                f"{class_name}"
            )

            continue


        videos = sorted(
            class_dir.glob("*.mp4")
        )

        print()
        print(
            f"[{class_id + 1:02d}/"
            f"{len(CLASS_NAMES)}] "
            f"{class_name}: "
            f"{len(videos)} videos"
        )


        for video_path in videos:

            total_videos += 1

            try:

                sequence = process_video(
                    video_path
                )

                if sequence is None:

                    failed += 1

                    print(
                        f"  FAILED: "
                        f"{video_path.name}"
                    )

                    continue


                X.append(
                    sequence
                )

                y.append(
                    class_id
                )

                metadata.append({

                    "class_id":
                        class_id,

                    "label":
                        class_name,

                    "video":
                        video_path.name
                })


                successful += 1


            except Exception as e:

                failed += 1

                print(
                    f"  ERROR: "
                    f"{video_path.name} "
                    f"-> {e}"
                )


    # ========================================================
    # CONVERT TO ARRAYS
    # ========================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )


    print()
    print("=" * 70)

    print(
        "                 DATASET CREATED"
    )

    print("=" * 70)

    print()

    print(
        "Total videos found:",
        total_videos
    )

    print(
        "Successfully processed:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Expected sequence:",
        "(40, 126)"
    )


    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )


    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_df.to_csv(
        OUTPUT_DIR /
        "metadata.csv",
        index=False
    )


    import re

    clean_labels = [
        re.sub(r"^\d+\.", "", name)
        for name in CLASS_NAMES
    ]

    label_df = pd.DataFrame({

    "class_id":
        list(range(
            len(CLASS_NAMES)
        )),

    "label":
        clean_labels
})

    label_df.to_csv(
        OUTPUT_DIR /
        "label_mapping.csv",
        index=False
    )


    print()
    print(
        "Saved to:",
        OUTPUT_DIR
    )

    print()
    print("=" * 70)

    print(
        "              EXTRACTION COMPLETE"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()