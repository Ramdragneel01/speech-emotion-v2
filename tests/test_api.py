
"""Integration tests for speech-emotion-v2 FastAPI endpoints."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from api import main as main_module
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


def _override_settings(monkeypatch, **changes):
    """Apply temporary runtime setting overrides for security behavior tests."""

    monkeypatch.setattr(main_module, "settings", replace(main_module.settings, **changes))


@pytest.fixture(autouse=True)
def reset_runtime_state(monkeypatch):
    """Reset limiter and auth settings between tests for deterministic assertions."""

    main_module.limiter.clear()
    _override_settings(monkeypatch, api_key="", max_requests_per_minute=120)

    yield

    main_module.limiter.clear()


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


def test_predict_requires_api_key_when_configured(monkeypatch):
    """Prediction endpoint should require API key when runtime key is configured."""

    _override_settings(monkeypatch, api_key="secret-key")

    unauthorized = client.post(
        "/predict",
        files={"file": ("sample.wav", _wav_payload(), "audio/wav")},
    )
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/predict",
        headers={"X-API-Key": "secret-key"},
        files={"file": ("sample.wav", _wav_payload(), "audio/wav")},
    )
    assert authorized.status_code == 200


def test_health_remains_public_when_api_key_enabled(monkeypatch):
    """Health endpoint should stay public for probes even with API key enabled."""

    _override_settings(monkeypatch, api_key="secret-key")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rate_limit_returns_429(monkeypatch):
    """Prediction endpoint should return 429 once per-client quota is exceeded."""

    _override_settings(monkeypatch, max_requests_per_minute=1)

    first = client.post(
        "/predict",
        files={"file": ("sample.wav", _wav_payload(), "audio/wav")},
    )
    assert first.status_code == 200

    second = client.post(
        "/predict",
        files={"file": ("sample.wav", _wav_payload(), "audio/wav")},
    )
    assert second.status_code == 429
