from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware

from src.ml.predict_ensemble_video import predict_video


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ISL Translator API",
    description=(
        "Backend API for Indian Sign Language "
        "video recognition."
    ),
    version="1.0.0",
)
class TTSRequest(BaseModel):
    text: str
    language: str


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "ISL Translator API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# ============================================================
# TEXT TO SPEECH
# ============================================================

@app.post("/tts")
async def text_to_speech(request: TTSRequest):

    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
    }

    if request.language not in language_codes:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language."
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    temp_dir = Path(tempfile.mkdtemp())
    audio_path = temp_dir / "speech.mp3"

    try:

        language_code = language_codes[
            request.language
        ]

        tts = gTTS(
            text=request.text,
            lang=language_code,
            slow=False
        )

        tts.save(str(audio_path))

        return FileResponse(
            path=str(audio_path),
            media_type="audio/mpeg",
            filename="speech.mp3",
            background=None,
        )

    except Exception as error:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
# ============================================================
# PREDICT VIDEO
# ============================================================

@app.post("/predict")
async def predict_sign(
    video: UploadFile = File(...)
):

    temp_dir = None

    try:

        # ====================================================
        # VALIDATE FILE
        # ====================================================

        if not video.filename:

            raise HTTPException(
                status_code=400,
                detail="No video file selected.",
            )

        # ====================================================
        # CHECK VIDEO EXTENSION
        # ====================================================

        allowed_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
        }

        file_extension = (
            Path(video.filename)
            .suffix
            .lower()
        )

        if file_extension not in allowed_extensions:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported video format. "
                    "Please upload MP4, AVI, MOV, or MKV."
                ),
            )

        # ====================================================
        # CREATE TEMPORARY DIRECTORY
        # ====================================================

        temp_dir = Path(
            tempfile.mkdtemp()
        )

        temp_path = (
            temp_dir
            / f"uploaded_video{file_extension}"
        )

        # ====================================================
        # SAVE UPLOADED VIDEO
        # ====================================================

        with temp_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                video.file,
                buffer,
            )

        # ====================================================
        # RUN ML PREDICTION
        # ====================================================

        result = predict_video(
            temp_path
        )

        # ====================================================
        # RETURN CLEAN RESULT
        # ====================================================

        return {

            "success": True,

            "class_id": result[
                "class_id"
            ],

            "label": result[
                "label"
            ],

            "confidence": result[
                "confidence"
            ],

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "Prediction error:",
            str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:

        # ====================================================
        # CLOSE UPLOADED FILE
        # ====================================================

        try:

            await video.close()

        except Exception:

            pass

        # ====================================================
        # REMOVE TEMPORARY FILES
        # ====================================================

        if temp_dir:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


# ============================================================
# CNN-LSTM LIVE VIDEO PREDICTION
# ============================================================

CNN_MODEL_PATH = Path(
    "data/ml_cnn_lstm/cnn_lstm_40frames_augmented.keras"
)

CNN_LABEL_MAPPING_PATH = Path(
    "data/ml_cnn_lstm/label_mapping.csv"
)

CNN_MEDIAPIPE_MODEL = Path(
    "models/hand_landmarker.task"
)

CNN_MODEL = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

cnn_label_df = pd.read_csv(
    CNN_LABEL_MAPPING_PATH
)

CNN_LABEL_MAP = dict(
    zip(
        cnn_label_df["class_id"].astype(int),
        cnn_label_df["label"].astype(str)
    )
)


def cnn_empty_hand():
    return np.zeros(63, dtype=np.float32)


def cnn_extract_features(result):

    left_hand = cnn_empty_hand()
    right_hand = cnn_empty_hand()

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
                    result.handedness[i][0].category_name
                )

                if handedness == "Left":
                    left_hand = coords

                elif handedness == "Right":
                    right_hand = coords

    return np.concatenate([
        left_hand,
        right_hand
    ])


def cnn_normalize_sequence(sequence):

    sequence = sequence.reshape(
        40,
        42,
        3
    )

    normalized = np.zeros_like(
        sequence,
        dtype=np.float32
    )

    for f in range(40):

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
        40,
        126
    )


@app.post("/predict-live-video")
async def predict_live_video(
    video: UploadFile = File(...)
):

    temp_dir = Path(
        tempfile.mkdtemp()
    )

    try:

        temp_path = temp_dir / "live.webm"

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(
                video.file,
                buffer
            )

        cap = cv2.VideoCapture(
            str(temp_path)
        )

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open recorded video."
            )

        frames = []

        while True:

            success, frame = cap.read()

            if not success:
                break

            frames.append(frame)

        cap.release()

        if len(frames) < 40:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Need at least 40 frames. "
                    f"Received {len(frames)}."
                )
            )

        indices = np.linspace(
            0,
            len(frames) - 1,
            40
        ).astype(int)

        sampled = [
            frames[i]
            for i in indices
        ]

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
                model_asset_path=str(
                    CNN_MEDIAPIPE_MODEL
                )
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        feature_frames = []

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            for i, frame in enumerate(sampled):

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    i * 100
                )

                feature_frames.append(
                    cnn_extract_features(result)
                )

        feature_frames = np.asarray(
            feature_frames,
            dtype=np.float32
        )

        X = cnn_normalize_sequence(
            feature_frames
        )

        X = np.expand_dims(
            X,
            axis=0
        )

        probabilities = CNN_MODEL.predict(
            X,
            verbose=0
        )[0]

        class_id = int(
            np.argmax(probabilities)
        )

        label = CNN_LABEL_MAP.get(
            class_id,
            str(class_id)
        )

        confidence = float(
            probabilities[class_id]
        )

        return {
            "success": True,
            "class_id": class_id,
            "label": label,
            "confidence": confidence
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Live CNN-LSTM prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        await video.close()

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )
# ============================================================
# LIVE WEBCAM PREDICTION
# ============================================================

class LivePredictionRequest(BaseModel):
    label: str
    confidence: float


latest_live_prediction = {
    "success": False,
    "label": None,
    "confidence": 0.0,
}


@app.post("/live-prediction")
async def live_prediction(request: LivePredictionRequest):

    global latest_live_prediction

    if not request.label.strip():
        raise HTTPException(
            status_code=400,
            detail="Prediction label cannot be empty."
        )

    if not 0 <= request.confidence <= 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1."
        )

    latest_live_prediction = {
        "success": True,
        "label": request.label,
        "confidence": request.confidence,
    }

    print(
        f"Live prediction received: "
        f"{request.label} "
        f"({request.confidence * 100:.2f}%)"
    )

    return latest_live_prediction


# ============================\================================
# GET LATEST LIVE PREDICTION
# ============================================================

@app.get("/live-prediction")
async def get_live_prediction():

    return latest_live_prediction