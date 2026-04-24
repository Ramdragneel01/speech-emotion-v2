
"""FastAPI inference service for speech emotion classification."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.infer import predict_emotion_from_audio
from src.model import EMOTIONS, load_model


@dataclass(frozen=True)
class Settings:
    """Configuration used by the API runtime and request validators."""

    app_name: str
    app_version: str
    model_path: str
    max_upload_bytes: int
    max_requests_per_minute: int
    cors_origins: list[str]
    api_key: str


class InMemoryRateLimiter:
    """Simple per-client sliding-window limiter for abuse prevention."""

    def __init__(self, window_seconds: int = 60) -> None:
        """Initialize state store for request window accounting."""

        self._window_seconds = window_seconds
        self._store: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, limit: int) -> bool:
        """Return True if key has not exceeded the configured request limit."""

        now = perf_counter()
        with self._lock:
            queue = self._store.setdefault(key, deque())
            cutoff = now - self._window_seconds
            while queue and queue[0] < cutoff:
                queue.popleft()
            if len(queue) >= limit:
                return False
            queue.append(now)
        return True

    def clear(self) -> None:
        """Reset limiter state for deterministic tests and controlled resets."""

        with self._lock:
            self._store.clear()


class PredictionResponse(BaseModel):
    """Response payload returned by successful inference requests."""

    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    duration_seconds: float = Field(gt=0.0)
    sample_rate_hz: int = Field(gt=0)
    scores: dict[str, float]
    model_version: str
    request_id: str
    bytes_received: int = Field(gt=0)


class HealthResponse(BaseModel):
    """Health diagnostics used by monitoring and deployment checks."""

    status: str
    timestamp: str
    model_version: str
    max_upload_bytes: int
    max_requests_per_minute: int


def _load_settings() -> Settings:
    """Load runtime settings from environment variables with safe defaults."""

    origin_value = os.getenv("CORS_ORIGINS", "http://127.0.0.1:4174,http://localhost:4174")
    default_model_path = Path(__file__).resolve().parents[1] / "models" / "latest" / "model.json"
    return Settings(
        app_name=os.getenv("APP_NAME", "speech-emotion-v2"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        model_path=os.getenv("MODEL_PATH", str(default_model_path)),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", "5000000")),
        max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "120")),
        cors_origins=[origin.strip() for origin in origin_value.split(",") if origin.strip()],
        api_key=os.getenv("SPEECH_API_KEY", "").strip(),
    )


def _require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce optional API key auth on protected inference endpoints."""

    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


settings = _load_settings()
model = load_model(Path(settings.model_path))
limiter = InMemoryRateLimiter(window_seconds=60)
AuthDep = Annotated[None, Depends(_require_api_key)]

REQUEST_COUNTER = Counter(
    "speech_api_requests_total",
    "Speech API request count",
    labelnames=("method", "endpoint", "status"),
)
REQUEST_LATENCY = Histogram(
    "speech_api_request_duration_seconds",
    "Speech API request duration in seconds",
    labelnames=("method", "endpoint"),
    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
PREDICT_COUNTER = Counter(
    "speech_predictions_total",
    "Predictions by emotion label",
    labelnames=("emotion",),
)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach request ids, standard security headers, and latency metrics."""

    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id

    start = perf_counter()
    response = None
    try:
        response = await call_next(request)
    finally:
        elapsed = perf_counter() - start
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(elapsed)

    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=()"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Normalize errors with request ids for traceable client diagnostics."""

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(exc.detail),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.get("/health")
def health() -> HealthResponse:
    """Return runtime readiness and current inference model metadata."""

    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_version=model.version,
        max_upload_bytes=settings.max_upload_bytes,
        max_requests_per_minute=settings.max_requests_per_minute,
    )


@app.get("/model/info")
def model_info(_: AuthDep = None) -> dict[str, object]:
    """Expose model metadata for debugging and release verification."""

    return {
        "version": model.version,
        "trained_at": model.trained_at,
        "feature_names": model.feature_names,
        "labels": EMOTIONS,
        "artifact_path": str(Path(settings.model_path).resolve()),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    _: AuthDep = None,
    file: UploadFile = File(...),
    expected_sample_rate: int | None = Query(default=None, ge=8000, le=96000),
) -> PredictionResponse:
    """Accept WAV uploads, validate request constraints, and return prediction payload."""

    client_key = request.client.host if request.client else "unknown"
    if not limiter.allow(client_key, settings.max_requests_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav uploads are supported")

    allowed_types = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported content type")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds maximum size")

    try:
        prediction = predict_emotion_from_audio(raw, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if expected_sample_rate and prediction["sample_rate_hz"] != expected_sample_rate:
        raise HTTPException(
            status_code=400,
            detail=f"Expected sample rate {expected_sample_rate}, got {prediction['sample_rate_hz']}",
        )

    PREDICT_COUNTER.labels(emotion=prediction["emotion"]).inc()

    return PredictionResponse(
        emotion=str(prediction["emotion"]),
        confidence=float(prediction["confidence"]),
        duration_seconds=float(prediction["duration_seconds"]),
        sample_rate_hz=int(prediction["sample_rate_hz"]),
        scores={label: float(score) for label, score in prediction["scores"].items()},
        model_version=model.version,
        request_id=str(request.state.request_id),
        bytes_received=len(raw),
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics for API latency and prediction monitoring."""

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
