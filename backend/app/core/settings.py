"""Runtime configuration for the LFM3 Neuron Explorer backend.

Values come from environment variables so the same code runs in tests, on CPU,
and on the ROCm development machine without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_ID = "LiquidAI/LFM3-1.7B"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw).expanduser().resolve()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_id: str = field(default_factory=lambda: os.environ.get("LFM3_MODEL_ID", DEFAULT_MODEL_ID))
    revision: str = field(default_factory=lambda: os.environ.get("LFM3_REVISION", "main"))
    device: str = field(default_factory=lambda: os.environ.get("LFM3_DEVICE", "auto"))
    host: str = field(default_factory=lambda: os.environ.get("LFM3_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("LFM3_PORT", 8000))
    max_new_tokens: int = field(default_factory=lambda: _env_int("LFM3_MAX_NEW_TOKENS", 256))
    max_steering_strength: float = field(
        default_factory=lambda: float(os.environ.get("LFM3_MAX_STEERING_STRENGTH", "1000") or 1000)
    )
    max_batch_runs: int = field(default_factory=lambda: _env_int("LFM3_MAX_BATCH_RUNS", 2000))
    activation_store_experiments: int = field(
        default_factory=lambda: _env_int("LFM3_ACTIVATION_STORE_EXPERIMENTS", 32)
    )
    activation_store_bytes: int = field(
        default_factory=lambda: _env_int("LFM3_ACTIVATION_STORE_BYTES", 512 * 1024 * 1024)
    )
    allow_cpu_fallback: bool = field(default_factory=lambda: _env_bool("LFM3_ALLOW_CPU_FALLBACK", True))
    data_dir: Path = field(default_factory=lambda: _env_path("LFM3_DATA_DIR", PROJECT_ROOT / "data"))
    persist_activations: bool = field(default_factory=lambda: _env_bool("LFM3_PERSIST_ACTIVATIONS", False))
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"


settings = Settings()
