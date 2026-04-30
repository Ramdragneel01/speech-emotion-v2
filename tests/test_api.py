
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


def _assert_error_contract(
    response,
    expected_status: int,
    expected_code: str,
    expected_message: str | None = None,
    expect_details: bool = False,
) -> None:
    """Validate normalized API error contract fields."""

    assert response.status_code == expected_status
    payload = response.json()
    assert isinstance(payload.get("error"), dict)
    assert payload["error"]["code"] == expected_code
    assert isinstance(payload["error"]["message"], str)
    if expected_message is not None:
        assert payload["error"]["message"] == expected_message
    assert isinstance(payload["error"]["request_id"], str)
    assert payload["error"]["request_id"]
    if expect_details:
        assert "details" in payload["error"]


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


def test_probe_alias_endpoints_return_healthy_state():
    """Readiness and probe aliases should be available for deployment checks."""

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    health_alias = client.get("/healthz")
    assert health_alias.status_code == 200
    assert health_alias.json()["status"] == "ok"

    ready_alias = client.get("/readyz")
    assert ready_alias.status_code == 200
    assert ready_alias.json()["status"] == "ready"


def test_predict_rejects_non_wav_upload():
    """Prediction endpoint should reject unsupported file extensions."""

    response = client.post(
        "/predict",
        files={"file": ("invalid.txt", b"hello", "text/plain")},
    )
    _assert_error_contract(
        response,
        expected_status=400,
        expected_code="bad_request",
        expected_message="Only .wav uploads are supported",
    )


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
    _assert_error_contract(
        unauthorized,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

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

    ready_response = client.get("/ready")
    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"


def test_phase1_auth_required_contract(monkeypatch):
    """Protected endpoints should require API key with normalized unauthorized errors."""

    _override_settings(monkeypatch, api_key="phase1-secret")

    unauthorized = client.get("/model/info")
    _assert_error_contract(
        unauthorized,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    invalid_key = client.get("/model/info", headers={"X-API-Key": "wrong"})
    _assert_error_contract(
        invalid_key,
        expected_status=401,
        expected_code="unauthorized",
        expected_message="api_key_invalid",
    )

    authorized = client.get("/model/info", headers={"X-API-Key": "phase1-secret"})
    assert authorized.status_code == 200


def test_phase1_error_contract_response():
    """Missing multipart file should return normalized validation error payload."""

    response = client.post("/predict")
    _assert_error_contract(
        response,
        expected_status=422,
        expected_code="validation_error",
        expected_message="request_validation_failed",
        expect_details=True,
    )


def test_error_responses_include_request_and_security_headers(monkeypatch):
    """Error responses should carry request tracing and baseline security headers."""

    _override_settings(monkeypatch, api_key="header-secret")
    response = client.get("/model/info")

    assert response.status_code == 401
    assert response.headers.get("X-Request-ID")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"


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
    _assert_error_contract(
        second,
        expected_status=429,
        expected_code="rate_limited",
        expected_message="rate_limited",
    )
    assert second.headers.get("Retry-After") == "60"
