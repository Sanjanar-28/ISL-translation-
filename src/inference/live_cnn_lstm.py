import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import urllib.request
import urllib.error
import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "data/ml_cnn_lstm/cnn_lstm_40frames_augmented.keras"
)

LABEL_PATH = Path(
    "data/ml_cnn_lstm/label_classes_augmented.npy"
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

# CNN + LSTM expects exactly 40 frames

TARGET_FRAMES = 40

# User can record anywhere from 40 to 100 frames

MAX_RECORDING_FRAMES = 100


# ============================================================
# BACKEND
# ============================================================

BACKEND_URL = (
    "http://127.0.0.1:8000/live-prediction"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("             CNN + LSTM LIVE ISL")
print("=" * 65)

print("\nLoading CNN + LSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD LABEL MAPPING
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


print("Model loaded.")
print("Classes:", len(label_map))
print("Input shape:", model.input_shape)


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
# EXTRACT 126 LANDMARK FEATURES
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
# NORMALIZE
# ============================================================

def normalize_sequence(sequence):

    sequence = sequence.reshape(
        TARGET_FRAMES,
        42,
        3
    )

    normalized = np.zeros_like(
        sequence,
        dtype=np.float32
    )

    for f in range(
        TARGET_FRAMES
    ):

        landmarks = sequence[f]

        origin = landmarks[0].copy()

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

        normalized[f] = relative

    return normalized.reshape(
        TARGET_FRAMES,
        126
    )


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
# SEND PREDICTION TO BACKEND
# ============================================================

def send_prediction_to_backend(
    prediction,
    confidence
):

    # The model gives confidence as:
    #
    # 0 - 100
    #
    # Backend expects:
    #
    # 0 - 1

    confidence_decimal = (
        float(confidence) / 100.0
    )

    payload = {
        "label": str(prediction),
        "confidence": confidence_decimal
    }

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        BACKEND_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=5
        ) as response:

            response_data = (
                response.read()
                .decode("utf-8")
            )

            result = json.loads(
                response_data
            )

            if result.get("success"):

                print()
                print(
                    "Backend connection: SUCCESS"
                )

                print(
                    "Backend received:",
                    result.get("label")
                )

                print(
                    "Backend confidence:",
                    f"{result.get('confidence', 0) * 100:.1f}%"
                )

                return True

            else:

                print()
                print(
                    "Backend returned an unsuccessful response."
                )

                return False

    except urllib.error.URLError as error:

        print()
        print(
            "Backend connection: FAILED"
        )

        print(
            "Make sure FastAPI is running on:"
        )

        print(
            "http://127.0.0.1:8000"
        )

        print(
            "Error:",
            error
        )

        return False

    except Exception as error:

        print()
        print(
            "Backend connection error:",
            error
        )

        return False


# ============================================================
# PREDICT SIGN
# ============================================================

def predict_sign(frames):

    print()
    print("=" * 65)
    print("Processing captured frames...")
    print("=" * 65)

    total_frames = len(frames)

    # --------------------------------------------------------
    # CHECK MINIMUM
    # --------------------------------------------------------

    if total_frames < TARGET_FRAMES:

        print(
            f"Need at least {TARGET_FRAMES} frames."
        )

        print(
            f"Only received {total_frames} frames."
        )

        return None, 0.0

    # --------------------------------------------------------
    # SAMPLE EXACTLY 40 FRAMES
    # --------------------------------------------------------

    indices = np.linspace(
        0,
        total_frames - 1,
        TARGET_FRAMES
    ).astype(int)

    sampled = [
        frames[i]
        for i in indices
    ]

    print(
        f"Captured frames : {total_frames}"
    )

    print(
        f"Using frames    : {TARGET_FRAMES}"
    )

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    landmarker = create_landmarker()

    feature_frames = []

    try:

        for i, frame in enumerate(
            sampled
        ):

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

            timestamp_ms = i * 100

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

            feature_frames.append(
                features
            )

    finally:

        landmarker.close()

    # --------------------------------------------------------
    # CONVERT TO ARRAY
    # --------------------------------------------------------

    feature_frames = np.asarray(
        feature_frames,
        dtype=np.float32
    )

    print(
        "Raw feature shape:",
        feature_frames.shape
    )

    if feature_frames.shape != (
        TARGET_FRAMES,
        126
    ):

        print(
            "ERROR: Wrong feature shape."
        )

        return None, 0.0

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    X = normalize_sequence(
        feature_frames
    )

    # --------------------------------------------------------
    # FINAL MODEL INPUT
    # --------------------------------------------------------

    X = np.expand_dims(
        X,
        axis=0
    )

    print(
        "Model input shape:",
        X.shape
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        X,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    best_index = top_indices[0]

    class_id = int(
        best_index
    )

    prediction = label_map.get(
        class_id,
        str(class_id)
    )

    confidence = (
        probabilities[best_index] * 100
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("                  PREDICTION")
    print("=" * 65)

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence:.1f}%"
    )

    print()
    print("Top 5 predictions:")

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        label = label_map.get(
            int(idx),
            str(idx)
        )

        score = (
            probabilities[idx] * 100
        )

        print(
            f"{rank}. "
            f"{label:<20} "
            f"{score:.1f}%"
        )

    print("=" * 65)

    # ========================================================
    # SEND TO BACKEND
    # ========================================================

    send_prediction_to_backend(
        prediction,
        confidence
    )

    return prediction, confidence


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("Controls:")
    print("SPACE = Start / Stop recording")
    print("Q     = Quit")
    print()
    print(
        f"Maximum recording: "
        f"{MAX_RECORDING_FRAMES} frames"
    )

    print(
        f"Model input: "
        f"{TARGET_FRAMES} frames"
    )

    print()

    # --------------------------------------------------------
    # WEBCAM
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    # --------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------

    recording = False

    captured_frames = []

    prediction = None

    confidence = 0.0

    processing = False

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read webcam."
            )

            break

        # ----------------------------------------------------
        # ORIGINAL FRAME
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
        # RECORDING
        # ----------------------------------------------------

        if recording:

            captured_frames.append(
                original_frame.copy()
            )

            status = "RECORDING"

            # ------------------------------------------------
            # AUTOMATIC 100-FRAME LIMIT
            # ------------------------------------------------

            if (
                len(captured_frames)
                >= MAX_RECORDING_FRAMES
            ):

                recording = False

                processing = True

                print()
                print("=" * 65)
                print(
                    "100 FRAME LIMIT REACHED"
                )
                print(
                    "Processing automatically..."
                )
                print("=" * 65)

                prediction, confidence = (
                    predict_sign(
                        captured_frames
                    )
                )

                processing = False

        else:

            if processing:

                status = "PROCESSING"

            else:

                status = "READY"

        # ----------------------------------------------------
        # DISPLAY STATUS
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

        # ----------------------------------------------------
        # FRAME COUNT
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            (
                f"Frames: "
                f"{len(captured_frames)}/"
                f"{MAX_RECORDING_FRAMES}"
            ),
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

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
                (
                    f"Confidence: "
                    f"{confidence:.1f}%"
                ),
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # ----------------------------------------------------
        # INSTRUCTIONS
        # ----------------------------------------------------

        if not recording:

            cv2.putText(
                display_frame,
                "SPACE = Start",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "SPACE = Stop",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.putText(
            display_frame,
            "Q = Quit",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # SHOW
        # ----------------------------------------------------

        cv2.imshow(
            "CNN + LSTM ISL Predictor",
            display_frame
        )

        # ----------------------------------------------------
        # KEYBOARD
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

                confidence = 0.0

                recording = True

                print()
                print("=" * 65)
                print("RECORDING STARTED")
                print(
                    "Perform ONE sign."
                )
                print(
                    "Press SPACE when finished."
                )
                print(
                    f"Maximum: "
                    f"{MAX_RECORDING_FRAMES} frames"
                )
                print("=" * 65)

            # ------------------------------------------------
            # STOP
            # ------------------------------------------------

            else:

                recording = False

                print()
                print(
                    "Recording stopped."
                )

                print(
                    "Captured frames:",
                    len(captured_frames)
                )

                # --------------------------------------------
                # PREDICT
                # --------------------------------------------

                if (
                    len(captured_frames)
                    >= TARGET_FRAMES
                ):

                    prediction, confidence = (
                        predict_sign(
                            captured_frames
                        )
                    )

                else:

                    print(
                        "Not enough frames."
                    )

                    print(
                        f"Minimum required: "
                        f"{TARGET_FRAMES}"
                    )

        # ====================================================
        # Q
        # ====================================================

        elif key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("Webcam closed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()