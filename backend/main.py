from pathlib import Path
import shutil
import tempfile
import io

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from datetime import datetime, timezone

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)

from pydantic import BaseModel

from gtts import gTTS

from fastapi.middleware.cors import CORSMiddleware

from src.ml.predict_ensemble_video import predict_video


# ============================================================
# PATHS
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

SIGN_DATASET_ROOT = Path(
    r"D:\Amrita SLR Dataset\sign project"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".mkv",
}


# ============================================================
# MONGODB
# ============================================================

MONGO_URI = "mongodb://localhost:27017/"

mongo_client = MongoClient(
    MONGO_URI
)

db = mongo_client[
    "ISL_ASSISTANT_DB"
]

users_collection = db[
    "users"
]

history_collection = db[
    "history"
]


# ============================================================
# EMAIL NORMALIZATION
# ============================================================

def normalize_email(email: str):

    return email.strip().lower()


# ============================================================
# UNIQUE EMAIL INDEX
# ============================================================

try:

    users_collection.create_index(
        "email",
        unique=True,
        name="unique_email"
    )

    print(
        "MongoDB unique email index ready."
    )

except Exception as error:

    print(
        "WARNING: Could not create unique email index."
    )

    print(
        "If duplicate emails already exist, "
        "remove duplicate test users in MongoDB Compass."
    )

    print(
        "MongoDB error:",
        error
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ISL Translator API",
    description=(
        "Backend API for Indian Sign Language "
        "video recognition."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST MODELS
# ============================================================

class TTSRequest(BaseModel):

    text: str
    language: str


class LivePredictionRequest(BaseModel):

    label: str
    confidence: float


class ProfileRequest(BaseModel):

    userId: str | None = None
    name: str
    email: str
    phone: str = ""
    preferredLanguage: str = "English"


class HistoryRequest(BaseModel):

    userId: str
    sign: str
    confidence: float


class TranslateSignRequest(BaseModel):

    sign: str
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
        "message":
            "ISL Translator API is running"
    }


# ============================================================
# HEALTH
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
async def text_to_speech(
    request: TTSRequest
):

    language_codes = {

        "English": "en",

        "Hindi": "hi",

        "Kannada": "kn",
    }

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    if request.language not in language_codes:

        raise HTTPException(
            status_code=400,
            detail="Unsupported language."
        )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if not request.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    try:

        language_code = (
            language_codes[
                request.language
            ]
        )

        # ----------------------------------------------------
        # GENERATE TTS
        # ----------------------------------------------------

        tts = gTTS(
            text=request.text,
            lang=language_code,
            slow=False
        )

        # ----------------------------------------------------
        # MEMORY BUFFER
        # ----------------------------------------------------

        audio_buffer = io.BytesIO()

        tts.write_to_fp(
            audio_buffer
        )

        audio_buffer.seek(0)

        # ----------------------------------------------------
        # RETURN MP3
        # ----------------------------------------------------

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition":
                    'inline; filename="speech.mp3"'
            }
        )

    except Exception as error:

        print(
            "TTS error:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# PREDICT UPLOADED VIDEO
# ============================================================

@app.post("/predict")
async def predict_sign(
    video: UploadFile = File(...)
):

    temp_dir = None

    try:

        if not video.filename:

            raise HTTPException(
                status_code=400,
                detail="No video file selected."
            )

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

        if (
            file_extension
            not in allowed_extensions
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported video format. "
                    "Please upload MP4, AVI, MOV, "
                    "MKV, or WEBM."
                ),
            )

        temp_dir = Path(
            tempfile.mkdtemp()
        )

        temp_path = (
            temp_dir
            / f"uploaded_video{file_extension}"
        )

        with temp_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                video.file,
                buffer
            )

        result = predict_video(
            temp_path
        )

        return {

            "success": True,

            "class_id":
                result["class_id"],

            "label":
                result["label"],

            "confidence":
                result["confidence"],
        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "Prediction error:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        try:
            await video.close()
        except Exception:
            pass

        if temp_dir:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


# ============================================================
# LOAD CNN + LSTM
# ============================================================

print()
print("=" * 65)
print("Loading CNN + LSTM model...")
print("=" * 65)

CNN_MODEL = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

cnn_label_df = pd.read_csv(
    CNN_LABEL_MAPPING_PATH
)

CNN_LABEL_MAP = dict(
    zip(
        cnn_label_df[
            "class_id"
        ].astype(int),

        cnn_label_df[
            "label"
        ].astype(str)
    )
)

print(
    "CNN + LSTM model loaded."
)

print(
    "Number of classes:",
    len(CNN_LABEL_MAP)
)


# ============================================================
# CNN HELPERS
# ============================================================

def cnn_empty_hand():

    return np.zeros(
        63,
        dtype=np.float32
    )


def cnn_extract_features(
    result
):

    left_hand = (
        cnn_empty_hand()
    )

    right_hand = (
        cnn_empty_hand()
    )

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
                and i <
                len(result.handedness)
            ):

                handedness = (
                    result
                    .handedness[i][0]
                    .category_name
                )

                if (
                    handedness
                    == "Left"
                ):

                    left_hand = coords

                elif (
                    handedness
                    == "Right"
                ):

                    right_hand = coords

    return np.concatenate([
        left_hand,
        right_hand
    ])


def cnn_normalize_sequence(
    sequence
):

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

        landmarks = (
            sequence[f]
        )

        origin = (
            landmarks[0]
            .copy()
        )

        relative = (
            landmarks - origin
        )

        distances = (
            np.linalg.norm(
                relative,
                axis=1
            )
        )

        scale = np.max(
            distances
        )

        if scale > 1e-8:

            relative = (
                relative / scale
            )

        normalized[f] = (
            relative
        )

    return normalized.reshape(
        40,
        126
    )


# ============================================================
# CNN-LSTM LIVE VIDEO
# ============================================================

@app.post("/predict-live-video")
async def predict_live_video(
    video: UploadFile = File(...)
):

    temp_dir = Path(
        tempfile.mkdtemp()
    )

    try:

        temp_path = (
            temp_dir / "live.webm"
        )

        with temp_path.open(
            "wb"
        ) as buffer:

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
                detail=(
                    "Could not open recorded video."
                )
            )

        frames = []

        while True:

            success, frame = (
                cap.read()
            )

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

        BaseOptions = (
            mp.tasks.BaseOptions
        )

        HandLandmarker = (
            mp.tasks.vision.HandLandmarker
        )

        HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = (
            HandLandmarkerOptions(

                base_options=BaseOptions(
                    model_asset_path=str(
                        CNN_MEDIAPIPE_MODEL
                    )
                ),

                running_mode=(
                    RunningMode.VIDEO
                ),

                num_hands=2,

                min_hand_detection_confidence=0.5,

                min_hand_presence_confidence=0.5,

                min_tracking_confidence=0.5,
            )
        )

        feature_frames = []

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

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

                result = (
                    landmarker
                    .detect_for_video(
                        mp_image,
                        i * 100
                    )
                )

                feature_frames.append(
                    cnn_extract_features(
                        result
                    )
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

        probabilities = (
            CNN_MODEL.predict(
                X,
                verbose=0
            )[0]
        )

        class_id = int(
            np.argmax(
                probabilities
            )
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

            "class_id":
                class_id,

            "label":
                label,

            "confidence":
                confidence,
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

        try:
            await video.close()
        except Exception:
            pass

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# LIVE WEBCAM PREDICTION
# ============================================================

latest_live_prediction = {

    "success": False,

    "label": None,

    "confidence": 0.0,
}


class LivePredictionRequest(BaseModel):

    label: str
    confidence: float


@app.post("/live-prediction")
async def live_prediction(
    request: LivePredictionRequest
):

    global latest_live_prediction

    if not request.label.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction label "
                "cannot be empty."
            )
        )

    if not (
        0
        <= request.confidence
        <= 1
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Confidence must be "
                "between 0 and 1."
            )
        )

    latest_live_prediction = {

        "success": True,

        "label":
            request.label,

        "confidence":
            request.confidence,
    }

    return latest_live_prediction


@app.get("/live-prediction")
async def get_live_prediction():

    return latest_live_prediction


# ============================================================
# MULTILINGUAL SIGN TRANSLATION
# ============================================================

SIGN_TRANSLATIONS = {

    "bird": {
        "English": "bird",
        "Hindi": "पक्षी",
        "Kannada": "ಪಕ್ಷಿ",
    },

    "black": {
        "English": "black",
        "Hindi": "काला",
        "Kannada": "ಕಪ್ಪು",
    },

    "cat": {
        "English": "cat",
        "Hindi": "बिल्ली",
        "Kannada": "ಬೆಕ್ಕು",
    },

    "cow": {
        "English": "cow",
        "Hindi": "गाय",
        "Kannada": "ಹಸು",
    },

    "dog": {
        "English": "dog",
        "Hindi": "कुत्ता",
        "Kannada": "ನಾಯಿ",
    },

    "fish": {
        "English": "fish",
        "Hindi": "मछली",
        "Kannada": "ಮೀನು",
    },

    "goodmorning": {
        "English": "good morning",
        "Hindi": "सुप्रभात",
        "Kannada": "ಶುಭೋದಯ",
    },

    "grey": {
        "English": "grey",
        "Hindi": "धूसर",
        "Kannada": "ಬೂದು",
    },

    "hello": {
        "English": "hello",
        "Hindi": "नमस्ते",
        "Kannada": "ನಮಸ್ಕಾರ",
    },

    "youareperfect": {
        "English": "you are perfect",
        "Hindi": "आप बिल्कुल सही हैं",
        "Kannada": "ನೀವು ಪರಿಪೂರ್ಣರಾಗಿದ್ದೀರಿ",
    },

    "monsoon": {
        "English": "monsoon",
        "Hindi": "मानसून",
        "Kannada": "ಮುಂಗಾರು",
    },

    "afternoon": {
        "English": "afternoon",
        "Hindi": "दोपहर",
        "Kannada": "ಮಧ್ಯಾಹ್ನ",
    },

    "angry": {
        "English": "angry",
        "Hindi": "गुस्सा",
        "Kannada": "ಕೋಪ",
    },

    "bad": {
        "English": "bad",
        "Hindi": "बुरा",
        "Kannada": "ಕೆಟ್ಟದ್ದು",
    },

    "boy": {
        "English": "boy",
        "Hindi": "लड़का",
        "Kannada": "ಹುಡುಗ",
    },

    "eat": {
        "English": "eat",
        "Hindi": "खाना",
        "Kannada": "ತಿನ್ನು",
    },

    "friend": {
        "English": "friend",
        "Hindi": "दोस्त",
        "Kannada": "ಸ್ನೇಹಿತ",
    },

    "drinking": {
        "English": "drinking",
        "Hindi": "पीना",
        "Kannada": "ಕುಡಿಯುವುದು",
    },

    "girl": {
        "English": "girl",
        "Hindi": "लड़की",
        "Kannada": "ಹುಡುಗಿ",
    },

    "brother": {
        "English": "brother",
        "Hindi": "भाई",
        "Kannada": "ಅಣ್ಣ / ತಮ್ಮ",
    },

    "good": {
        "English": "good",
        "Hindi": "अच्छा",
        "Kannada": "ಒಳ್ಳೆಯದು",
    },

    "father": {
        "English": "father",
        "Hindi": "पिता",
        "Kannada": "ತಂದೆ",
    },

    "evening": {
        "English": "evening",
        "Hindi": "शाम",
        "Kannada": "ಸಂಜೆ",
    },

    "help": {
        "English": "help",
        "Hindi": "मदद",
        "Kannada": "ಸಹಾಯ",
    },

    "mother": {
        "English": "mother",
        "Hindi": "माँ",
        "Kannada": "ತಾಯಿ",
    },

    "NAME": {
        "English": "name",
        "Hindi": "नाम",
        "Kannada": "ಹೆಸರು",
    },

    "Night": {
        "English": "night",
        "Hindi": "रात",
        "Kannada": "ರಾತ್ರಿ",
    },

    "Music": {
        "English": "music",
        "Hindi": "संगीत",
        "Kannada": "ಸಂಗೀತ",
    },

    "nose": {
        "English": "nose",
        "Hindi": "नाक",
        "Kannada": "ಮೂಗು",
    },

    "sleep": {
        "English": "sleep",
        "Hindi": "सोना",
        "Kannada": "ನಿದ್ರೆ",
    },

    "sit": {
        "English": "sit",
        "Hindi": "बैठो",
        "Kannada": "ಕುಳಿತುಕೊಳ್ಳಿ",
    },

    "sorry": {
        "English": "sorry",
        "Hindi": "माफ़ कीजिए",
        "Kannada": "ಕ್ಷಮಿಸಿ",
    },

    "stand": {
        "English": "stand",
        "Hindi": "खड़े हो जाओ",
        "Kannada": "ನಿಲ್ಲಿ",
    },

    "stop": {
        "English": "stop",
        "Hindi": "रुको",
        "Kannada": "ನಿಲ್ಲಿಸಿ",
    },

    "student": {
        "English": "student",
        "Hindi": "छात्र",
        "Kannada": "ವಿದ್ಯಾರ್ಥಿ",
    },

    "study": {
        "English": "study",
        "Hindi": "पढ़ाई",
        "Kannada": "ಅಧ್ಯಯನ",
    },

    "teacher": {
        "English": "teacher",
        "Hindi": "शिक्षक",
        "Kannada": "ಶಿಕ್ಷಕ",
    },

    "thankyou": {
        "English": "thank you",
        "Hindi": "धन्यवाद",
        "Kannada": "ಧನ್ಯವಾದಗಳು",
    },

    "today": {
        "English": "today",
        "Hindi": "आज",
        "Kannada": "ಇಂದು",
    },

    "tommorow": {
        "English": "tomorrow",
        "Hindi": "कल",
        "Kannada": "ನಾಳೆ",
    },

    "welcome": {
        "English": "welcome",
        "Hindi": "स्वागत है",
        "Kannada": "ಸ್ವಾಗತ",
    },

    "work": {
        "English": "work",
        "Hindi": "काम",
        "Kannada": "ಕೆಲಸ",
    },

    "yesterday": {
        "English": "yesterday",
        "Hindi": "कल",
        "Kannada": "ನಿನ್ನೆ",
    },

    "teeth": {
        "English": "teeth",
        "Hindi": "दाँत",
        "Kannada": "ಹಲ್ಲುಗಳು",
    },

    "hand": {
        "English": "hand",
        "Hindi": "हाथ",
        "Kannada": "ಕೈ",
    },

    "write": {
        "English": "write",
        "Hindi": "लिखना",
        "Kannada": "ಬರೆಯಿರಿ",
    },

    "umberlla": {
        "English": "umbrella",
        "Hindi": "छाता",
        "Kannada": "ಛತ್ರಿ",
    },

    "ring": {
        "English": "ring",
        "Hindi": "अंगूठी",
        "Kannada": "ಉಂಗುರ",
    },

    "power": {
        "English": "power",
        "Hindi": "शक्ति",
        "Kannada": "ಶಕ್ತಿ",
    },
}


@app.post("/translate-sign")
async def translate_sign(
    request: TranslateSignRequest
):

    sign = request.sign.strip()

    language = request.language.strip()

    if language not in {
        "English",
        "Hindi",
        "Kannada",
    }:

        raise HTTPException(
            status_code=400,
            detail="Unsupported language."
        )

    translation = (
        SIGN_TRANSLATIONS.get(
            sign
        )
    )

    if translation is None:

        for key, value in (
            SIGN_TRANSLATIONS.items()
        ):

            if (
                key.lower()
                == sign.lower()
            ):

                translation = value

                break

    if translation is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No translation found "
                f"for sign '{sign}'."
            )
        )

    return {

        "success": True,

        "sign": sign,

        "language": language,

        "translatedText":
            translation[language],
    }


# ============================================================
# PROFILE
# ============================================================

@app.post("/profile")
async def save_profile(
    request: ProfileRequest
):

    email = normalize_email(
        request.email
    )

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    if not request.name.strip():

        raise HTTPException(
            status_code=400,
            detail="Name is required."
        )

    existing_by_email = (
        users_collection.find_one(
            {
                "email": email
            }
        )
    )

    if request.userId:

        try:

            requested_object_id = (
                ObjectId(
                    request.userId
                )
            )

        except Exception:

            requested_object_id = None

        if requested_object_id:

            current_user = (
                users_collection.find_one(
                    {
                        "_id":
                            requested_object_id
                    }
                )
            )

            if current_user:

                if (
                    existing_by_email
                    and
                    existing_by_email[
                        "_id"
                    ]
                    != requested_object_id
                ):

                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "This email is already "
                            "registered to another user."
                        )
                    )

                users_collection.update_one(

                    {
                        "_id":
                            requested_object_id
                    },

                    {
                        "$set": {

                            "name":
                                request.name.strip(),

                            "email":
                                email,

                            "phone":
                                request.phone.strip(),

                            "preferredLanguage":
                                request.preferredLanguage,

                            "updatedAt":
                                datetime.now(
                                    timezone.utc
                                ),
                        }
                    }
                )

                return {

                    "success": True,

                    "message":
                        "Profile updated successfully.",

                    "userId":
                        str(
                            requested_object_id
                        ),
                }

    if existing_by_email:

        existing_id = (
            existing_by_email[
                "_id"
            ]
        )

        users_collection.update_one(

            {
                "_id":
                    existing_id
            },

            {
                "$set": {

                    "name":
                        request.name.strip(),

                    "phone":
                        request.phone.strip(),

                    "preferredLanguage":
                        request.preferredLanguage,

                    "updatedAt":
                        datetime.now(
                            timezone.utc
                        ),
                }
            }
        )

        return {

            "success": True,

            "message":
                "Existing user found. Profile updated.",

            "userId":
                str(existing_id),
        }

    user_document = {

        "name":
            request.name.strip(),

        "email":
            email,

        "phone":
            request.phone.strip(),

        "preferredLanguage":
            request.preferredLanguage,

        "createdAt":
            datetime.now(
                timezone.utc
            ),

        "updatedAt":
            datetime.now(
                timezone.utc
            ),
    }

    try:

        result = (
            users_collection.insert_one(
                user_document
            )
        )

    except DuplicateKeyError:

        existing_user = (
            users_collection.find_one(
                {
                    "email": email
                }
            )
        )

        if not existing_user:

            raise HTTPException(
                status_code=409,
                detail=(
                    "Email already exists."
                )
            )

        return {

            "success": True,

            "message":
                "Existing user found.",

            "userId":
                str(
                    existing_user[
                        "_id"
                    ]
                ),
        }

    return {

        "success": True,

        "message":
            "New user created successfully.",

        "userId":
            str(
                result.inserted_id
            ),
    }


# ============================================================
# GET PROFILE
# ============================================================

@app.get("/profile/{user_id}")
async def get_profile(
    user_id: str
):

    try:

        object_id = ObjectId(
            user_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid user ID."
        )

    user = (
        users_collection.find_one(
            {
                "_id": object_id
            }
        )
    )

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found."
        )

    return {

        "userId":
            str(user["_id"]),

        "name":
            user.get(
                "name",
                ""
            ),

        "email":
            user.get(
                "email",
                ""
            ),

        "phone":
            user.get(
                "phone",
                ""
            ),

        "preferredLanguage":
            user.get(
                "preferredLanguage",
                "English"
            ),
    }


# ============================================================
# SAVE HISTORY
# ============================================================

@app.post("/history")
async def save_history(
    request: HistoryRequest
):

    if not request.userId:

        raise HTTPException(
            status_code=400,
            detail="User ID is required."
        )

    try:

        user_object_id = ObjectId(
            request.userId
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid user ID."
        )

    user = (
        users_collection.find_one(
            {
                "_id":
                    user_object_id
            }
        )
    )

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found."
        )

    if not request.sign.strip():

        raise HTTPException(
            status_code=400,
            detail="Sign cannot be empty."
        )

    if not (
        0
        <= request.confidence
        <= 1
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Confidence must be "
                "between 0 and 1."
            )
        )

    history_document = {

        "userId":
            str(user_object_id),

        "sign":
            request.sign.strip(),

        "confidence":
            float(request.confidence),

        "timestamp":
            datetime.now(
                timezone.utc
            ),
    }

    result = (
        history_collection.insert_one(
            history_document
        )
    )

    return {

        "success": True,

        "message":
            "History saved successfully.",

        "historyId":
            str(
                result.inserted_id
            ),
    }


# ============================================================
# GET HISTORY
# ============================================================

@app.get("/history/{user_id}")
async def get_history(
    user_id: str
):

    try:

        object_id = ObjectId(
            user_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid user ID."
        )

    user = (
        users_collection.find_one(
            {
                "_id":
                    object_id
            }
        )
    )

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found."
        )

    records = (
        history_collection
        .find(
            {
                "userId":
                    str(object_id)
            }
        )
        .sort(
            "timestamp",
            -1
        )
    )

    history = []

    for record in records:

        history.append({

            "_id":
                str(
                    record["_id"]
                ),

            "userId":
                record.get(
                    "userId"
                ),

            "sign":
                record.get(
                    "sign",
                    ""
                ),

            "confidence":
                float(
                    record.get(
                        "confidence",
                        0
                    )
                ),

            "timestamp":
                record.get(
                    "timestamp"
                ),
        })

    return history


# ============================================================
# SIGN VIDEO DATASET
# ============================================================

def find_sign_folder(
    sign_name: str
):

    target = (
        sign_name
        .strip()
        .lower()
    )

    if not SIGN_DATASET_ROOT.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "ISL sign dataset folder not found."
            )
        )

    for folder in (
        SIGN_DATASET_ROOT.iterdir()
    ):

        if not folder.is_dir():
            continue

        folder_name = folder.name

        if "." not in folder_name:
            continue

        folder_sign = (
            folder_name
            .split(".", 1)[1]
            .strip()
            .lower()
        )

        if folder_sign == target:

            return folder

    return None


@app.get(
    "/sign-video/{sign_name}"
)
async def get_sign_video(
    sign_name: str
):

    folder = find_sign_folder(
        sign_name
    )

    if folder is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Sign '{sign_name}' not found."
            )
        )

    videos = sorted(
        [
            file
            for file in folder.rglob("*")
            if (
                file.is_file()
                and
                file.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        ]
    )

    if not videos:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No video found for sign "
                f"'{sign_name}'."
            )
        )

    video_path = videos[0]

    media_types = {

        ".mp4":
            "video/mp4",

        ".webm":
            "video/webm",

        ".mov":
            "video/quicktime",

        ".avi":
            "video/x-msvideo",

        ".mkv":
            "video/x-matroska",
    }

    return FileResponse(

        path=str(
            video_path
        ),

        media_type=(
            media_types.get(
                video_path.suffix.lower(),
                "video/mp4"
            )
        ),

        filename=
            video_path.name,
    )