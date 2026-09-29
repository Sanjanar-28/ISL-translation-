import cv2
import numpy as np
import joblib
import mediapipe as mp
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("data/ml_live/augmented_live_random_forest.joblib")
LABEL_PATH = Path("data/ml_live/label_mapping.csv")
MEDIAPIPE_MODEL = Path("models/hand_landmarker.task")

TARGET_FRAMES = 20


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("             ISL LIVE PREDICTOR")
print("=" * 60)

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")
print("Model expects:", model.n_features_in_, "features")


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

label_mapping = pd.read_csv(LABEL_PATH)

# Expected:
# class_id,label

label_dict = {}

for _, row in label_mapping.iterrows():

    class_id = int(row["class_id"])
    label = str(row["label"])

    label_dict[class_id] = label


print("Labels loaded:", len(label_dict))


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# EMPTY HAND
# ============================================================

def empty_hand():
    return np.zeros(63, dtype=np.float32)


# ============================================================
# EXTRACT 126 FEATURES FROM ONE FRAME
# ============================================================

def extract_frame_features(result):

    left_hand = empty_hand()
    right_hand = empty_hand()

    if result.hand_landmarks:

        for i, landmarks in enumerate(result.hand_landmarks):

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

            # ------------------------------------------------
            # Handedness
            # ------------------------------------------------

            if result.handedness and i < len(result.handedness):

                handedness = (
                    result.handedness[i][0].category_name
                )

                if handedness == "Left":

                    left_hand = coords

                elif handedness == "Right":

                    right_hand = coords

    # EXACT ORDER:
    #
    # Left  = 63
    # Right = 63
    #
    # Total = 126

    return np.concatenate(
        [left_hand, right_hand]
    )


# ============================================================
# NORMALIZATION
# SAME AS TRAINING
# ============================================================

def normalize_frames(frames):

    frames = np.asarray(
        frames,
        dtype=np.float32
    )

    if frames.shape != (TARGET_FRAMES, 126):

        raise ValueError(
            f"Unexpected frame shape: {frames.shape}"
        )

    # --------------------------------------------------------
    # 20 x 126
    #      ↓
    # 20 x 42 x 3
    # --------------------------------------------------------

    frames = frames.reshape(
        TARGET_FRAMES,
        42,
        3
    )

    normalized = np.zeros_like(frames)

    for f in range(TARGET_FRAMES):

        landmarks = frames[f]

        # First landmark = origin
        origin = landmarks[0].copy()

        relative = landmarks - origin

        # Scale
        distances = np.linalg.norm(
            relative,
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-8:

            relative = relative / scale

        normalized[f] = relative

    return normalized.reshape(-1)


# ============================================================
# SAMPLE 20 FRAMES
# ============================================================

def sample_frames(frames):

    total = len(frames)

    if total < TARGET_FRAMES:

        return None

    indices = np.linspace(
        0,
        total - 1,
        TARGET_FRAMES
    ).astype(int)

    return [
        frames[i]
        for i in indices
    ]


# ============================================================
# CREATE MEDIAPIPE LANDMARKER
# ============================================================

def create_landmarker():

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

    return HandLandmarker.create_from_options(
        options
    )


# ============================================================
# PREDICT
# ============================================================

def predict_sign(frames):

    print("\nProcessing captured frames...")

    sampled = sample_frames(frames)

    if sampled is None:

        print(
            f"ERROR: Need at least "
            f"{TARGET_FRAMES} frames."
        )

        return None, None

    print(
        f"Using {len(sampled)} sampled frames."
    )

    # --------------------------------------------------------
    # CREATE FRESH LANDMARKER
    # --------------------------------------------------------

    landmarker = create_landmarker()

    feature_frames = []

    try:

        for i, frame in enumerate(sampled):

            # IMPORTANT:
            #
            # DO NOT FLIP FRAME HERE.
            #
            # MediaPipe receives the original
            # webcam orientation.

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = i * 100

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            features = extract_frame_features(
                result
            )

            feature_frames.append(
                features
            )

    finally:

        landmarker.close()

    # --------------------------------------------------------
    # CHECK FEATURE SHAPE
    # --------------------------------------------------------

    feature_frames = np.asarray(
        feature_frames,
        dtype=np.float32
    )

    print(
        "Raw feature shape:",
        feature_frames.shape
    )

    if feature_frames.shape != (20, 126):

        print(
            "ERROR: Feature shape is incorrect."
        )

        return None, None

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    features = normalize_frames(
        feature_frames
    )

    X = features.reshape(
        1,
        -1
    )

    print(
        "Final feature shape:",
        X.shape
    )

    # --------------------------------------------------------
    # VERIFY MODEL INPUT
    # --------------------------------------------------------

    if X.shape[1] != model.n_features_in_:

        print(
            "ERROR: Model expects",
            model.n_features_in_,
            "features but received",
            X.shape[1]
        )

        return None, None

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[0]

    prediction_id = int(
        model.classes_[
            np.argmax(probabilities)
        ]
    )

    confidence = (
        np.max(probabilities) * 100
    )

    prediction_label = label_dict.get(
        prediction_id,
        str(prediction_id)
    )

    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    print("\n" + "=" * 60)
    print("                  PREDICTION")
    print("=" * 60)

    print(
        f"Prediction : {prediction_label}"
    )

    print(
        f"Class ID   : {prediction_id}"
    )

    print(
        f"Confidence : {confidence:.1f}%"
    )

    print("\nTop 5 predictions:")

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        class_id = int(
            model.classes_[idx]
        )

        label = label_dict.get(
            class_id,
            str(class_id)
        )

        prob = probabilities[idx] * 100

        print(
            f"{rank}. "
            f"{label:<20} "
            f"{prob:.1f}%"
        )

    print("=" * 60)

    return prediction_label, confidence


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nControls:")
    print("SPACE = Start / Stop recording")
    print("Q     = Quit")
    print()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        return

    recording = False

    captured_frames = []

    prediction = None

    confidence = None

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read webcam."
            )

            break

        # ----------------------------------------------------
        # ORIGINAL FRAME FOR MEDIAPIPE
        # ----------------------------------------------------

        original_frame = frame.copy()

        # ----------------------------------------------------
        # MIRRORED FRAME ONLY FOR DISPLAY
        # ----------------------------------------------------

        display_frame = cv2.flip(
            original_frame,
            1
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        if recording:

            # IMPORTANT:
            # Store ORIGINAL frame.
            #
            # Do NOT store the mirrored display frame.

            captured_frames.append(
                original_frame.copy()
            )

            status = "RECORDING"

        else:

            status = "READY"

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Frames: {len(captured_frames)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if prediction is not None:

            cv2.putText(
                display_frame,
                f"Prediction: {prediction}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Confidence: {confidence:.1f}%",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            "ISL Live Predictor",
            display_frame
        )

        # ----------------------------------------------------
        # KEY
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # SPACE
        # ====================================================

        if key == 32:

            # ------------------------------------------------
            # START
            # ------------------------------------------------

            if not recording:

                captured_frames = []

                prediction = None

                confidence = None

                recording = True

                print("\n")
                print("=" * 60)
                print("RECORDING STARTED")
                print("Perform ONE sign now.")
                print("Press SPACE when finished.")
                print("=" * 60)

            # ------------------------------------------------
            # STOP
            # ------------------------------------------------

            else:

                recording = False

                print("\n")
                print(
                    "Recording stopped."
                )

                print(
                    "Captured frames:",
                    len(captured_frames)
                )

                if len(captured_frames) >= TARGET_FRAMES:

                    prediction, confidence = predict_sign(
                        captured_frames
                    )

                else:

                    print(
                        f"Not enough frames."
                    )

                    print(
                        f"Captured: "
                        f"{len(captured_frames)}"
                    )

                    print(
                        f"Required: "
                        f"{TARGET_FRAMES}"
                    )

        # ====================================================
        # Q
        # ====================================================

        elif key == ord("q"):

            break

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print("\nWebcam closed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()