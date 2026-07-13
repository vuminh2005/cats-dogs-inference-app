import io
import os
import time
import uuid
import threading
import zipfile
from datetime import datetime
from typing import Optional

import boto3
import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from sqlalchemy import create_engine, text
from mlflow.tracking import MlflowClient

# ============================================================================
# CONFIG
# ============================================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_URI = os.getenv("MODEL_URI", "models:/cats_dogs_classifier@champion")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID")
KERAS_MODEL_ARTIFACT_PATH = os.getenv(
    "KERAS_MODEL_ARTIFACT_PATH",
    "model/data/model.keras",
)

# Direct R2 download. Recommended for large model artifacts on Render free tier.
# Example key:
# artifacts/1/0d81fbc6a8c9455188d422298766dea3/artifacts/model/data/model.keras
MODEL_R2_OBJECT_KEY = os.getenv("MODEL_R2_OBJECT_KEY")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/tmp/cats_dogs_model.keras")

DATABASE_URL = os.getenv("DATABASE_URL")

R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_UPLOAD_PREFIX = os.getenv("R2_UPLOAD_PREFIX", "production-uploads")
R2_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "auto")

IMG_SIZE = tuple(int(x.strip()) for x in os.getenv("IMG_SIZE", "299,299").split(","))
NUM_CHANNELS = int(os.getenv("NUM_CHANNELS", "3"))
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
UNCERTAIN_CONFIDENCE_THRESHOLD = float(os.getenv("UNCERTAIN_CONFIDENCE_THRESHOLD", "0"))

MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

# ============================================================================
# APP INIT
# ============================================================================

app = FastAPI(title="Cats vs Dogs Inference API")
templates = Jinja2Templates(directory="app/templates")

model = None
model_load_lock = threading.Lock()
engine = None
s3_client = None
last_model_file_debug = None
model_load_status = {
    "status": "not_started",
    "step": None,
    "started_at": None,
    "finished_at": None,
    "elapsed_seconds": None,
    "error": None,
}


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
def startup_event():
    global engine, s3_client

    if not MLFLOW_TRACKING_URI:
        raise RuntimeError("Missing MLFLOW_TRACKING_URI")
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL")
    if not R2_BUCKET_NAME:
        raise RuntimeError("Missing R2_BUCKET_NAME")
    if not R2_ENDPOINT_URL:
        raise RuntimeError("Missing MLFLOW_S3_ENDPOINT_URL")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Important: do not load TensorFlow/model during startup.
    # Render needs the app to bind the port first.
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

def update_model_load_status(**kwargs):
    """Update simple JSON-serializable model loading status for debugging."""
    global model_load_status
    model_load_status.update(kwargs)


def current_model_load_status() -> dict:
    status = dict(model_load_status)
    if status.get("started_at") and not status.get("finished_at"):
        status["elapsed_seconds"] = round(time.time() - status["started_at"], 2)
    return status


def head_model_object() -> dict:
    if not MODEL_R2_OBJECT_KEY:
        raise RuntimeError("MODEL_R2_OBJECT_KEY is not set")
    if s3_client is None:
        raise RuntimeError("S3 client is not initialized")

    resp = s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=MODEL_R2_OBJECT_KEY)
    return {
        "bucket": R2_BUCKET_NAME,
        "key": MODEL_R2_OBJECT_KEY,
        "content_length_bytes": int(resp.get("ContentLength", 0)),
        "content_length_mb": round(int(resp.get("ContentLength", 0)) / (1024 * 1024), 2),
        "content_type": resp.get("ContentType"),
        "etag": resp.get("ETag"),
        "last_modified": resp.get("LastModified").isoformat() if resp.get("LastModified") else None,
    }

def get_effective_model_uri() -> str:
    if MODEL_R2_OBJECT_KEY:
        return f"s3://{R2_BUCKET_NAME}/{MODEL_R2_OBJECT_KEY}"
    if MODEL_RUN_ID:
        return f"runs:/{MODEL_RUN_ID}/{KERAS_MODEL_ARTIFACT_PATH}"
    return MODEL_URI


def inspect_model_file(path: str) -> dict:
    exists = os.path.exists(path)
    size_bytes = os.path.getsize(path) if exists else 0
    is_zip = zipfile.is_zipfile(path) if exists else False
    first_bytes_hex = None

    if exists:
        with open(path, "rb") as f:
            first_bytes_hex = f.read(16).hex()

    return {
        "path": path,
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "is_zipfile": is_zip,
        "first_16_bytes_hex": first_bytes_hex,
    }


def download_model_from_r2(force: bool = False) -> str:
    if not MODEL_R2_OBJECT_KEY:
        raise RuntimeError("MODEL_R2_OBJECT_KEY is not set")
    if s3_client is None:
        raise RuntimeError("S3 client is not initialized")

    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH) or "/tmp", exist_ok=True)

    # Reuse only if it already looks like a valid .keras zip file.
    # This avoids reusing a partially downloaded/corrupted file in /tmp.
    if not force and os.path.exists(MODEL_LOCAL_PATH):
        info = inspect_model_file(MODEL_LOCAL_PATH)
        if info["size_bytes"] > 1024 * 1024 and info["is_zipfile"]:
            return MODEL_LOCAL_PATH

    update_model_load_status(step="checking_r2_object")
    head_model_object()

    tmp_path = MODEL_LOCAL_PATH + ".download"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    update_model_load_status(step="downloading_model_from_r2")
    s3_client.download_file(
        Bucket=R2_BUCKET_NAME,
        Key=MODEL_R2_OBJECT_KEY,
        Filename=tmp_path,
    )

    os.replace(tmp_path, MODEL_LOCAL_PATH)
    update_model_load_status(step="downloaded_model_from_r2")
    return MODEL_LOCAL_PATH


def download_model_from_mlflow_artifacts() -> str:
    if not MODEL_RUN_ID:
        raise RuntimeError("MODEL_RUN_ID is not set")

    client = MlflowClient()
    return client.download_artifacts(
        run_id=MODEL_RUN_ID,
        path=KERAS_MODEL_ARTIFACT_PATH,
    )


def get_model():
    global model, last_model_file_debug

    if model is not None:
        update_model_load_status(status="loaded", step="already_loaded", error=None)
        return model

    with model_load_lock:
        if model is not None:
            update_model_load_status(status="loaded", step="already_loaded", error=None)
            return model

        started_at = time.time()
        update_model_load_status(
            status="loading",
            step="starting",
            started_at=started_at,
            finished_at=None,
            elapsed_seconds=None,
            error=None,
        )

        try:
            if MODEL_R2_OBJECT_KEY:
                local_model_path = download_model_from_r2()
            elif MODEL_RUN_ID:
                update_model_load_status(step="downloading_model_from_mlflow_artifacts")
                local_model_path = download_model_from_mlflow_artifacts()
            else:
                update_model_load_status(step="loading_with_mlflow_tensorflow")
                import mlflow.tensorflow
                model = mlflow.tensorflow.load_model(MODEL_URI)
                update_model_load_status(
                    status="loaded",
                    step="loaded_with_mlflow_tensorflow",
                    finished_at=time.time(),
                    elapsed_seconds=round(time.time() - started_at, 2),
                    error=None,
                )
                return model

            last_model_file_debug = inspect_model_file(local_model_path)
            if not last_model_file_debug["exists"]:
                raise RuntimeError(f"Downloaded model file does not exist: {last_model_file_debug}")
            if not last_model_file_debug["is_zipfile"]:
                raise RuntimeError(
                    "Downloaded .keras file is not a valid zip file. "
                    f"Debug info: {last_model_file_debug}"
                )

            update_model_load_status(step="importing_tensorflow")
            import tensorflow as tf

            update_model_load_status(step="loading_keras_model")
            model = tf.keras.models.load_model(local_model_path, compile=False)

            update_model_load_status(
                status="loaded",
                step="loaded",
                finished_at=time.time(),
                elapsed_seconds=round(time.time() - started_at, 2),
                error=None,
            )
            return model

        except Exception as exc:
            update_model_load_status(
                status="error",
                step="error",
                finished_at=time.time(),
                elapsed_seconds=round(time.time() - started_at, 2),
                error=str(exc),
            )
            raise


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image).astype("float32")
    array = np.expand_dims(array, axis=0)
    return array


def upload_image_to_r2(file_bytes: bytes, filename: str, content_type: str) -> str:
    if s3_client is None:
        raise RuntimeError("S3 client is not initialized")

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


def parse_model_name_version(model_uri: str):
    model_name = None
    model_version = None

    if model_uri.startswith("models:/"):
        raw = model_uri.replace("models:/", "")
        if "@" in raw:
            model_name, model_version = raw.split("@", 1)
        else:
            parts = raw.split("/")
            model_name = parts[0] if len(parts) > 0 else None
            model_version = parts[1] if len(parts) > 1 else None
    elif MODEL_RUN_ID:
        model_name = "run_artifact"
        model_version = MODEL_RUN_ID
    elif MODEL_R2_OBJECT_KEY:
        model_name = "r2_artifact"
        model_version = MODEL_R2_OBJECT_KEY

    return model_name, model_version


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
    if engine is None:
        raise RuntimeError("Database engine is not initialized")

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

    model_name, model_version = parse_model_name_version(model_uri)

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

@app.head("/")
def head_home():
    return Response(status_code=200)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "effective_model_uri": get_effective_model_uri(),
        "model_run_id": MODEL_RUN_ID,
        "keras_model_artifact_path": KERAS_MODEL_ARTIFACT_PATH,
        "model_r2_object_key": MODEL_R2_OBJECT_KEY,
        "model_local_path": MODEL_LOCAL_PATH,
        "last_model_file_debug": last_model_file_debug,
        "model_load_status": current_model_load_status(),
        "img_size": IMG_SIZE,
        "threshold": THRESHOLD,
        "uncertain_confidence_threshold": UNCERTAIN_CONFIDENCE_THRESHOLD,
    }


@app.get("/debug/load-model")
def debug_load_model():
    try:
        get_model()
        return {
            "status": "loaded",
            "model_uri": MODEL_URI,
            "effective_model_uri": get_effective_model_uri(),
            "model_run_id": MODEL_RUN_ID,
            "keras_model_artifact_path": KERAS_MODEL_ARTIFACT_PATH,
            "model_r2_object_key": MODEL_R2_OBJECT_KEY,
            "last_model_file_debug": last_model_file_debug,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "model_uri": MODEL_URI,
                "effective_model_uri": get_effective_model_uri(),
                "model_run_id": MODEL_RUN_ID,
                "keras_model_artifact_path": KERAS_MODEL_ARTIFACT_PATH,
                "model_r2_object_key": MODEL_R2_OBJECT_KEY,
                "last_model_file_debug": last_model_file_debug,
                "error": str(exc),
            },
        )



@app.get("/debug/model-object")
def debug_model_object():
    try:
        return {
            "status": "ok",
            "object": head_model_object(),
            "effective_model_uri": get_effective_model_uri(),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(exc), "effective_model_uri": get_effective_model_uri()},
        )


@app.get("/debug/download-model-file")
def debug_download_model_file(force: bool = False):
    global last_model_file_debug
    try:
        local_path = download_model_from_r2(force=force)
        last_model_file_debug = inspect_model_file(local_path)
        return {
            "status": "downloaded",
            "effective_model_uri": get_effective_model_uri(),
            "local_path": local_path,
            "last_model_file_debug": last_model_file_debug,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "effective_model_uri": get_effective_model_uri(),
                "last_model_file_debug": last_model_file_debug,
                "error": str(exc),
            },
        )


@app.get("/debug/load-status")
def debug_load_status():
    return {
        "model_loaded": model is not None,
        "model_load_status": current_model_load_status(),
        "last_model_file_debug": last_model_file_debug,
        "effective_model_uri": get_effective_model_uri(),
    }


def _background_model_loader():
    try:
        get_model()
    except Exception:
        # get_model already records the error in model_load_status.
        pass


@app.get("/debug/start-load-model")
def debug_start_load_model():
    if model is not None:
        return {
            "status": "already_loaded",
            "model_load_status": current_model_load_status(),
            "last_model_file_debug": last_model_file_debug,
        }

    current_status = current_model_load_status().get("status")
    if current_status == "loading":
        return {
            "status": "already_loading",
            "model_load_status": current_model_load_status(),
            "last_model_file_debug": last_model_file_debug,
        }

    thread = threading.Thread(target=_background_model_loader, daemon=True)
    thread.start()
    return {
        "status": "started",
        "message": "Model loading started in a background thread. Poll /debug/load-status.",
        "model_load_status": current_model_load_status(),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_uri": get_effective_model_uri(),
            "threshold": THRESHOLD,
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    effective_model_uri = get_effective_model_uri()

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
        loaded_model = get_model()

        prob_dog = float(loaded_model.predict(image_array, verbose=0)[0][0])
        confidence = max(prob_dog, 1.0 - prob_dog)

        if UNCERTAIN_CONFIDENCE_THRESHOLD > 0 and confidence < UNCERTAIN_CONFIDENCE_THRESHOLD:
            predicted_label = "uncertain"
        else:
            predicted_label = "dog" if prob_dog >= THRESHOLD else "cat"

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
            model_uri=effective_model_uri,
            latency_ms=latency_ms,
        )

        return {
            "prediction": predicted_label,
            "prob_dog": prob_dog,
            "confidence": confidence,
            "threshold": THRESHOLD,
            "uncertain_confidence_threshold": UNCERTAIN_CONFIDENCE_THRESHOLD,
            "image_uri": image_uri,
            "latency_ms": latency_ms,
            "model_uri": effective_model_uri,
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
                model_uri=effective_model_uri,
                latency_ms=latency_ms,
                error_message=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/monitoring")
def monitoring():
    if engine is None:
        raise HTTPException(status_code=500, detail="Database engine is not initialized")

    sql = text(
        """
        SELECT
            COUNT(*) AS total_requests,
            AVG(confidence) AS avg_confidence,
            AVG(latency_ms) AS avg_latency_ms,
            SUM(CASE WHEN confidence < 0.6 THEN 1 ELSE 0 END) AS low_confidence_count,
            SUM(CASE WHEN predicted_label = 'cat' THEN 1 ELSE 0 END) AS cat_count,
            SUM(CASE WHEN predicted_label = 'dog' THEN 1 ELSE 0 END) AS dog_count,
            SUM(CASE WHEN predicted_label = 'uncertain' THEN 1 ELSE 0 END) AS uncertain_count,
            SUM(CASE WHEN predicted_label = 'ERROR' THEN 1 ELSE 0 END) AS error_count
        FROM prediction_logs
        """
    )

    with engine.begin() as conn:
        row = conn.execute(sql).mappings().first()

    total = row["total_requests"] or 0
    cat_count = row["cat_count"] or 0
    dog_count = row["dog_count"] or 0
    uncertain_count = row["uncertain_count"] or 0

    return {
        "total_requests": int(total),
        "avg_confidence": float(row["avg_confidence"] or 0),
        "avg_latency_ms": float(row["avg_latency_ms"] or 0),
        "low_confidence_count": int(row["low_confidence_count"] or 0),
        "cat_count": int(cat_count),
        "dog_count": int(dog_count),
        "uncertain_count": int(uncertain_count),
        "error_count": int(row["error_count"] or 0),
        "cat_ratio": float(cat_count / total) if total else 0,
        "dog_ratio": float(dog_count / total) if total else 0,
        "uncertain_ratio": float(uncertain_count / total) if total else 0,
    }


@app.get("/recent-predictions")
def recent_predictions(limit: int = 20):
    if engine is None:
        raise HTTPException(status_code=500, detail="Database engine is not initialized")

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