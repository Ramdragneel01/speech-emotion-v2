
"""Inference helper functions for speech-emotion-v2."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import soundfile as sf

from .model import PrototypeModel, score_feature_vector

MIN_DURATION_SECONDS = 0.3
MAX_DURATION_SECONDS = 20.0


def read_wav_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int, float]:
    """Decode WAV bytes and return normalized mono waveform, sample rate, and duration."""

    if not raw_bytes:
        raise ValueError("Empty file received")

    try:
        audio, sample_rate = sf.read(BytesIO(raw_bytes), dtype="float32")
    except RuntimeError as exc:
        raise ValueError("Invalid WAV payload") from exc

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio data must be one-dimensional and non-empty")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite values")

    duration_seconds = float(audio.shape[0] / sample_rate)
    if duration_seconds < MIN_DURATION_SECONDS:
        raise ValueError("Audio duration is too short")
    if duration_seconds > MAX_DURATION_SECONDS:
        raise ValueError("Audio duration exceeds maximum supported length")

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak

    return audio.astype(np.float32), int(sample_rate), duration_seconds


def extract_features(audio: np.ndarray) -> np.ndarray:
    """Extract compact audio features used by the lightweight classifier."""

    if audio.ndim != 1:
        raise ValueError("Audio features require a mono waveform")

    energy = float(np.sqrt(np.mean(np.square(audio))))
    zero_crossing = float(np.mean(np.abs(np.diff(np.signbit(audio))).astype(np.float32)))
    peak_abs = float(np.max(np.abs(audio)))
    silence_ratio = float(np.mean(np.abs(audio) < 0.02))

    return np.asarray([energy, zero_crossing, peak_abs, silence_ratio], dtype=np.float32)


def predict_emotion_from_logits(logits: np.ndarray) -> dict[str, object]:
    """Convert model logits into emotion label and confidence score."""

    if logits.ndim != 1:
        raise ValueError("logits must be a 1D array")

    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    index = int(np.argmax(probs))

    return {
        "emotion": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"][index],
        "confidence": float(probs[index]),
        "scores": {
            label: float(prob)
            for label, prob in zip(
                ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
                probs,
                strict=True,
            )
        },
    }


def predict_emotion_from_audio(raw_bytes: bytes, model: PrototypeModel) -> dict[str, object]:
    """Predict emotion probabilities from WAV bytes using trained centroid model."""

    audio, sample_rate, duration_seconds = read_wav_bytes(raw_bytes)
    features = extract_features(audio)
    scores = score_feature_vector(model, features)
    top_emotion = max(scores, key=scores.get)

    return {
        "emotion": top_emotion,
        "confidence": float(scores[top_emotion]),
        "scores": scores,
        "sample_rate_hz": sample_rate,
        "duration_seconds": duration_seconds,
        "feature_vector": [float(value) for value in features],
    }
