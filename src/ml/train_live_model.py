from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    r"D:\Amrita SLR Dataset\sign project"
)

OUTPUT_DIR = Path(
    "data/ml_live"
)

MEDIAPIPE_MODEL = Path(
    "models/hand_landmarker.task"
)

TARGET_FRAMES = 20

RANDOM_STATE = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 49 CLASS LABELS
# ============================================================

CLASS_NAMES = [
    "bird",
    "youareperfect",
    "monsoon",
    "afternoon",
    "angry",
    "bad",
    "boy",
    "eat",
    "friend",
    "drinking",
    "girl",
    "black",
    "brother",
    "good",
    "father",
    "evening",
    "help",
    "mother",
    "NAME",
    "Night",
    "Music",
    "nose",
    "cat",
    "sleep",
    "sit",
    "sorry",
    "stand",
    "stop",
    "student",
    "study",
    "teacher",
    "thankyou",
    "today",
    "cow",
    "tommorow",
    "welcome",
    "work",
    "yesterday",
    "teeth",
    "hand",
    "write",
    "umberlla",
    "ring",
    "dog",
    "power",
    "fish",
    "goodmorning",
    "grey",
    "hello",
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

    """
    21 landmarks × 3 coordinates
    = 63 values.
    """

    return np.zeros(
        63,
        dtype=np.float32
    )


# ============================================================
# EXTRACT HAND LANDMARKS
# ============================================================

def extract_hand_features(
    landmarks
):

    features = []

    for landmark in landmarks:

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

def extract_frame_features(
    result
):

    """
    One frame contains:

        Left hand  = 63
        Right hand = 63

        Total = 126 features
    """

    left_hand = empty_hand()

    right_hand = empty_hand()

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

        elif hand_label == "Right":

            right_hand = features

    return np.concatenate([
        left_hand,
        right_hand
    ])


# ============================================================
# NORMALIZE ONE FRAME
# ============================================================

def normalize_frame(
    frame_features
):

    """
    Same normalization logic used
    in normalize_landmarks.py.

    42 landmarks × 3 coordinates.

    1. First landmark = origin
    2. Translate all landmarks
    3. Calculate maximum distance
    4. Scale coordinates
    """

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

    scale = np.max(
        distances
    )

    if scale > 1e-8:

        relative = relative / scale

    return relative.reshape(
        -1
    ).astype(
        np.float32
    )


# ============================================================
# SAMPLE 20 FRAMES
# ============================================================

def sample_frames(
    frames
):

    if len(frames) < TARGET_FRAMES:

        return None

    indices = np.linspace(
        0,
        len(frames) - 1,
        TARGET_FRAMES
    ).astype(int)

    return [
        frames[index]
        for index in indices
    ]


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path
):

    """
    Process one video independently.

    IMPORTANT:
    A fresh MediaPipe landmarker is created
    for every video.

    This prevents the
    'timestamp must be monotonically increasing'
    error when moving to the next video.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"Could not open: "
            f"{video_path.name}"
        )

        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    frames = []

    frame_number = 0


    # ========================================================
    # FRESH MEDIAPIPE INSTANCE FOR THIS VIDEO
    # ========================================================

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


            # ------------------------------------------------
            # MediaPipe image
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp_ms = int(
                ((frame_number - 1) / fps)
                * 1000
            )


            # ------------------------------------------------
            # Detect
            # ------------------------------------------------

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )


            # ------------------------------------------------
            # Extract 126 features
            # ------------------------------------------------

            features = extract_frame_features(
                result
            )

            frames.append(
                features
            )


    cap.release()


    # ========================================================
    # CHECK FRAME COUNT
    # ========================================================

    if len(frames) < TARGET_FRAMES:

        print(
            f"SKIPPED - only "
            f"{len(frames)} frames"
        )

        return None


    # ========================================================
    # SAMPLE 20 FRAMES
    # ========================================================

    selected_frames = sample_frames(
        frames
    )

    if selected_frames is None:

        return None


    # ========================================================
    # NORMALIZE
    # ========================================================

    normalized_frames = []

    for frame in selected_frames:

        normalized = normalize_frame(
            frame
        )

        normalized_frames.append(
            normalized
        )


    # ========================================================
    # CONCATENATE
    # ========================================================

    features = np.concatenate(
        normalized_frames
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(features) != 2520:

        print(
            f"ERROR: {video_path.name} "
            f"generated {len(features)} features"
        )

        return None


    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("             BUILDING LIVE ISL DATASET")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_DIR}"
        )

    if not MEDIAPIPE_MODEL.exists():

        raise FileNotFoundError(
            f"MediaPipe model not found:\n"
            f"{MEDIAPIPE_MODEL}"
        )


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    X = []

    y = []

    metadata = []


    # ========================================================
    # PROCESS ALL 49 CLASSES
    # ========================================================

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        # ----------------------------------------------------
        # Find matching folder
        # ----------------------------------------------------

        matching_folders = [
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
            and folder.name.split(".", 1)[-1]
            == class_name
        ]


        if not matching_folders:

            print(
                f"\nWARNING: "
                f"Folder not found for "
                f"{class_name}"
            )

            continue


        class_dir = matching_folders[0]


        # ----------------------------------------------------
        # Get videos
        # ----------------------------------------------------

        videos = sorted(
            class_dir.glob("*.mp4")
        )


        print()
        print(
            f"[{class_id:02d}] "
            f"{class_name}: "
            f"{len(videos)} videos"
        )


        # ----------------------------------------------------
        # Process videos
        # ----------------------------------------------------

        for video_path in videos:

            print(
                f"    {video_path.name} ... ",
                end="",
                flush=True
            )


            try:

                features = process_video(
                    video_path
                )


            except Exception as error:

                print(
                    f"ERROR: {error}"
                )

                continue


            if features is None:

                print(
                    "SKIPPED"
                )

                continue


            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            X.append(
                features
            )

            y.append(
                class_id
            )

            metadata.append({
                "class_id": class_id,
                "label": class_name,
                "video": video_path.name
            })


            print(
                "OK"
            )


    # ========================================================
    # CREATE ARRAYS
    # ========================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    metadata_df = pd.DataFrame(
        metadata
    )


    # ========================================================
    # DATASET REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("                 DATASET CREATED")
    print("=" * 70)

    print(
        f"Samples: {len(X)}"
    )

    if len(X) > 0:

        print(
            f"Features: {X.shape[1]}"
        )

    print(
        f"Classes: {len(np.unique(y))}"
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(X) == 0:

        raise RuntimeError(
            "No training samples were created."
        )

    if X.shape[1] != 2520:

        raise RuntimeError(
            f"Expected 2520 features, "
            f"got {X.shape[1]}"
        )


    # ========================================================
    # SAVE DATASET
    # ========================================================

    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )

    metadata_df.to_csv(
        OUTPUT_DIR / "metadata.csv",
        index=False
    )

    pd.DataFrame({
        "class_id": range(
            len(CLASS_NAMES)
        ),
        "label": CLASS_NAMES
    }).to_csv(
        OUTPUT_DIR / "label_mapping.csv",
        index=False
    )


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print()
    print("=" * 70)
    print("                 TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )


    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )


    # ========================================================
    # TRAIN RANDOM FOREST
    # ========================================================

    print()
    print("=" * 70)
    print("              TRAINING RANDOM FOREST")
    print("=" * 70)
    print()

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )


    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # TEST
    # ========================================================

    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print()
    print("=" * 70)
    print("                    RESULTS")
    print("=" * 70)

    print()

    print(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        OUTPUT_DIR /
        "live_random_forest.joblib"
    )


    joblib.dump(
        model,
        model_path
    )


    # ========================================================
    # SAVE TEST DATA
    # ========================================================

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("                 TRAINING COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Model saved to:"
    )

    print(
        model_path
    )

    print()

    print(
        f"Dataset saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()