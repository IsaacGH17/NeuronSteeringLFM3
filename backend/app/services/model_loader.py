"""Model lifecycle: one model instance, loaded on demand and reused.

The real LFM3 checkpoint is never downloaded by the test suite. Tests inject a
small synthetic model through :meth:`ModelManager.inject`, so every other part
of the stack exercises the same code path as production.
"""

from __future__ import annotations

import importlib.metadata
import platform
import threading
from typing import Any

import torch

from backend.app.core.settings import Settings
from backend.app.core.settings import settings as default_settings
from backend.app.schemas.api import (
    DiagnosticsResponse,
    MemoryStats,
    ModelState,
    ModelStatus,
)


class ModelNotLoadedError(RuntimeError):
    """Raised when an operation needs a loaded model and none is available."""


class ModelLoadError(RuntimeError):
    """Raised when the checkpoint or tokenizer cannot be loaded."""


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def resolve_device(requested: str) -> str:
    """Map a user-facing device name to a concrete torch device string."""
    requested = (requested or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested in {"cuda", "gpu", "rocm"}:
        if not cuda_available:
            raise ModelLoadError(
                "A GPU device was requested but torch.cuda.is_available() is False. "
                "Start the backend with LFM3_DEVICE=cpu for CPU inference."
            )
        return "cuda"
    if requested.startswith("cuda:"):
        if not cuda_available:
            raise ModelLoadError(f"Device {requested!r} was requested but no ROCm/CUDA device is visible.")
        return requested
    if requested == "cpu":
        return "cpu"
    raise ModelLoadError(f"Unsupported device {requested!r}; use 'auto', 'cpu', or 'cuda'.")


def _dtype_for(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class ModelManager:
    """Owns the single model/tokenizer pair used for inference."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._state: ModelState = "unloaded"
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._model_id = self.settings.model_id
        self._revision = self.settings.revision
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        self._error: str | None = None
        self._lock = threading.RLock()
        self.inference_lock = threading.Lock()

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def loaded(self) -> bool:
        return self._state == "loaded" and self._model is not None and self._tokenizer is not None

    @property
    def model(self) -> Any:
        if not self.loaded:
            raise ModelNotLoadedError("No model is loaded. POST /api/model/load first.")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if not self.loaded:
            raise ModelNotLoadedError("No model is loaded. POST /api/model/load first.")
        return self._tokenizer

    @property
    def device(self) -> str:
        if self._device is None:
            raise ModelNotLoadedError("No model is loaded. POST /api/model/load first.")
        return self._device

    def status(self) -> ModelStatus:
        return ModelStatus(
            state=self._state,
            model_id=self._model_id,
            revision=self._revision,
            device=self._device,
            dtype=str(self._dtype).replace("torch.", "") if self._dtype is not None else None,
            loaded=self.loaded,
            error=self._error,
        )

    def inject(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str = "synthetic/test-model",
        revision: str = "test",
        device: str = "cpu",
    ) -> None:
        """Install an already-constructed model (tests, offline demos)."""
        with self._lock:
            self._model = model
            self._tokenizer = tokenizer
            self._model_id = model_id
            self._revision = revision
            self._device = device
            dtype = next((p.dtype for p in model.parameters()), None)
            self._dtype = dtype
            self._error = None
            self._state = "loaded"

    def load(
        self,
        *,
        model_id: str | None = None,
        revision: str | None = None,
        device: str | None = None,
    ) -> ModelStatus:
        with self._lock:
            if self.loaded and (model_id is None or model_id == self._model_id):
                return self.status()

            target_model = model_id or self.settings.model_id
            target_revision = revision or self.settings.revision
            requested_device = device or self.settings.device
            self._state = "loading"
            self._error = None
            try:
                target_device = resolve_device(requested_device)
                dtype = _dtype_for(target_device)

                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    target_model,
                    revision=target_revision,
                    trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    target_model,
                    revision=target_revision,
                    trust_remote_code=True,
                    dtype=dtype,
                )
                model.to(target_device)
                model.eval()
            except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the GUI
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                self._model = None
                self._tokenizer = None
                raise ModelLoadError(self._error) from exc

            self._model = model
            self._tokenizer = tokenizer
            self._model_id = target_model
            self._revision = target_revision
            self._device = target_device
            self._dtype = dtype
            self._state = "loaded"
            return self.status()

    def diagnostics(self) -> DiagnosticsResponse:
        cuda_available = torch.cuda.is_available()
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            backend = "rocm"
        elif cuda_available:
            backend = "cuda"
        else:
            backend = "cpu"

        device_name = torch.cuda.get_device_name(0) if cuda_available else None
        memory = None
        if cuda_available:
            try:
                free, total = torch.cuda.mem_get_info(0)
                memory = MemoryStats(
                    device=device_name or "cuda:0",
                    allocated_bytes=torch.cuda.memory_allocated(0),
                    reserved_bytes=torch.cuda.memory_reserved(0),
                    total_bytes=total,
                    free_bytes=free,
                )
            except Exception:  # noqa: BLE001 - memory stats are best-effort
                memory = None

        notes: list[str] = []
        if hip_version:
            notes.append(
                "torch.cuda reports an AMD GPU through ROCm; the 'cuda' API name does not imply NVIDIA hardware."
            )
        if not cuda_available:
            notes.append(
                "No ROCm/CUDA device is visible; inference will run on CPU if loaded with LFM3_DEVICE=cpu."
            )

        return DiagnosticsResponse(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            transformers_version=_package_version("transformers"),
            fastapi_version=_package_version("fastapi"),
            cuda_available=cuda_available,
            hip_version=hip_version,
            backend=backend,
            device_count=torch.cuda.device_count() if cuda_available else 0,
            device_name=device_name,
            inference_device=self._device,
            model_state=self._state,
            memory=memory,
            notes=notes,
        )


manager = ModelManager()
