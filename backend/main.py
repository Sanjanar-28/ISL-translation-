from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
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