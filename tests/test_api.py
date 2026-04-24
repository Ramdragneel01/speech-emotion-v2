
"""Integration tests for speech-emotion-v2 FastAPI endpoints."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from api.main import app

client = TestClient(app)


def _wav_payload() -> bytes:
    """Create valid mono WAV bytes for prediction endpoint tests."""

    sample_rate = 16000
    duration_seconds = 1.2
    sample_count = int(sample_rate * duration_seconds)
    timeline = np.linspace(0, duration_seconds, sample_count, endpoint=False)
    audio = 0.25 * np.sin(2 * np.pi * 440 * timeline).astype(np.float32)

    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def test_health_endpoint_returns_runtime_diagnostics():
    """Health endpoint should expose readiness details and limits."""

    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["max_upload_bytes"] > 0
    assert payload["max_requests_per_minute"] > 0


def test_predict_rejects_non_wav_upload():
    """Prediction endpoint should reject unsupported file extensions."""

    response = client.post(
        "/predict",
        files={"file": ("invalid.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only .wav" in response.json()["detail"]


def test_predict_accepts_valid_wav_upload():
    """Prediction endpoint should return stable emotion scoring payload."""

    response = client.post(
        "/predict",
        files={"file": ("sample.wav", _wav_payload(), "audio/wav")},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["emotion"] in payload["scores"]
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["bytes_received"] > 0
    assert payload["sample_rate_hz"] == 16000
