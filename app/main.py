import io
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import boto3
import mlflow
import mlflow.tensorflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from sqlalchemy import create_engine, text

# ============================================================================
# CONFIG
# ============================================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_URI = os.getenv("MODEL_URI", "models:/cats_dogs_classifier@champion")

DATABASE_URL = os.getenv("DATABASE_URL")

R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_UPLOAD_PREFIX = os.getenv("R2_UPLOAD_PREFIX", "production-uploads")
R2_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "auto")

IMG_SIZE = tuple(
    int(x) for x in os.getenv("IMG_SIZE", "299,299").split(",")
)
NUM_CHANNELS = int(os.getenv("NUM_CHANNELS", "3"))
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# ============================================================================
# APP INIT
# ============================================================================

app = FastAPI(title="Cats vs Dogs Inference API")
templates = Jinja2Templates(directory="app/templates")

model = None
engine = None
s3_client = None


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
def startup_event():
    global model, engine, s3_client

    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("Missing MLFLOW_TRACKING_URI")
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL")
    if not R2_BUCKET_NAME:
        raise RuntimeError("Missing R2_BUCKET_NAME")
    if not R2_ENDPOINT_URL:
        raise RuntimeError("Missing MLFLOW_S3_ENDPOINT_URL")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load model một lần khi app start, không load lại mỗi request
    model = mlflow.tensorflow.load_model(MODEL_URI)

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )


# ============================================================================
# UTILS
# ============================================================================

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image).astype("float32")
    array = np.expand_dims(array, axis=0)
    return array


def upload_image_to_r2(file_bytes: bytes, filename: str, content_type: str) -> str:
    safe_ext = filename.split(".")[-1].lower() if "." in filename else "jpg"
    object_key = (
        f"{R2_UPLOAD_PREFIX}/"
        f"{datetime.utcnow().strftime('%Y/%m/%d')}/"
        f"{uuid.uuid4()}.{safe_ext}"
    )

    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=object_key,
        Body=file_bytes,
        ContentType=content_type,
    )

    return f"s3://{R2_BUCKET_NAME}/{object_key}"


def save_prediction_log(
    original_filename: str,
    content_type: str,
    image_uri: str,
    predicted_label: str,
    prob_dog: float,
    confidence: float,
    threshold: float,
    model_uri: str,
    latency_ms: float,
    error_message: Optional[str] = None,
):
    sql = text(
        """
        INSERT INTO prediction_logs (
            original_filename,
            content_type,
            image_uri,
            predicted_label,
            prob_dog,
            confidence,
            threshold,
            model_uri,
            model_name,
            model_version,
            latency_ms,
            error_message
        )
        VALUES (
            :original_filename,
            :content_type,
            :image_uri,
            :predicted_label,
            :prob_dog,
            :confidence,
            :threshold,
            :model_uri,
            :model_name,
            :model_version,
            :latency_ms,
            :error_message
        )
        """
    )

    # Parse thô để dễ xem trong DB
    model_name = None
    model_version = None
    if model_uri.startswith("models:/"):
        parts = model_uri.replace("models:/", "").split("/")
        model_name = parts[0] if len(parts) > 0 else None
        model_version = parts[1] if len(parts) > 1 else None

    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "original_filename": original_filename,
                "content_type": content_type,
                "image_uri": image_uri,
                "predicted_label": predicted_label,
                "prob_dog": prob_dog,
                "confidence": confidence,
                "threshold": threshold,
                "model_uri": model_uri,
                "model_name": model_name,
                "model_version": model_version,
                "latency_ms": latency_ms,
                "error_message": error_message,
            },
        )


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "img_size": IMG_SIZE,
        "threshold": THRESHOLD,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_uri": MODEL_URI,
            "threshold": THRESHOLD,
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ ảnh JPEG hoặc PNG.",
        )

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File quá lớn. Tối đa {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        image_array = preprocess_image(file_bytes)

        prob_dog = float(model.predict(image_array, verbose=0)[0][0])
        predicted_label = "dog" if prob_dog >= THRESHOLD else "cat"
        confidence = max(prob_dog, 1.0 - prob_dog)

        image_uri = upload_image_to_r2(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_image.jpg",
            content_type=file.content_type,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        save_prediction_log(
            original_filename=file.filename or "unknown",
            content_type=file.content_type,
            image_uri=image_uri,
            predicted_label=predicted_label,
            prob_dog=prob_dog,
            confidence=confidence,
            threshold=THRESHOLD,
            model_uri=MODEL_URI,
            latency_ms=latency_ms,
        )

        return {
            "prediction": predicted_label,
            "prob_dog": prob_dog,
            "confidence": confidence,
            "threshold": THRESHOLD,
            "image_uri": image_uri,
            "latency_ms": latency_ms,
            "model_uri": MODEL_URI,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        try:
            save_prediction_log(
                original_filename=file.filename or "unknown",
                content_type=file.content_type or "unknown",
                image_uri="",
                predicted_label="ERROR",
                prob_dog=0.0,
                confidence=0.0,
                threshold=THRESHOLD,
                model_uri=MODEL_URI,
                latency_ms=latency_ms,
                error_message=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/monitoring")
def monitoring():
    sql = text(
        """
        SELECT
            COUNT(*) AS total_requests,
            AVG(confidence) AS avg_confidence,
            AVG(latency_ms) AS avg_latency_ms,
            SUM(CASE WHEN confidence < 0.6 THEN 1 ELSE 0 END) AS low_confidence_count,
            SUM(CASE WHEN predicted_label = 'cat' THEN 1 ELSE 0 END) AS cat_count,
            SUM(CASE WHEN predicted_label = 'dog' THEN 1 ELSE 0 END) AS dog_count,
            SUM(CASE WHEN predicted_label = 'ERROR' THEN 1 ELSE 0 END) AS error_count
        FROM prediction_logs
        """
    )

    with engine.begin() as conn:
        row = conn.execute(sql).mappings().first()

    total = row["total_requests"] or 0
    cat_count = row["cat_count"] or 0
    dog_count = row["dog_count"] or 0

    return {
        "total_requests": total,
        "avg_confidence": float(row["avg_confidence"] or 0),
        "avg_latency_ms": float(row["avg_latency_ms"] or 0),
        "low_confidence_count": int(row["low_confidence_count"] or 0),
        "cat_count": int(cat_count),
        "dog_count": int(dog_count),
        "error_count": int(row["error_count"] or 0),
        "cat_ratio": float(cat_count / total) if total else 0,
        "dog_ratio": float(dog_count / total) if total else 0,
    }


@app.get("/recent-predictions")
def recent_predictions(limit: int = 20):
    limit = min(max(limit, 1), 100)

    sql = text(
        """
        SELECT
            id,
            created_at,
            original_filename,
            image_uri,
            predicted_label,
            prob_dog,
            confidence,
            threshold,
            model_uri,
            latency_ms,
            error_message
        FROM prediction_logs
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()

    return [dict(row) for row in rows]