"""Persistent experiment metadata and optional activation snapshots.

Every run is appended as a single JSONL line so the history survives backend
restarts. Large activation tensors are stored separately as compressed ``.npz``
files only when requested; they are loaded back into the live activation store
lazily and only if they fit the memory budget.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from backend.app.core.settings import PROJECT_ROOT
from backend.app.core.version import APP_VERSION
from backend.app.schemas.api import (
    CaptureInfo,
    EnvironmentInfo,
    ExperimentDetail,
    ExperimentListResponse,
    ExperimentSummary,
    GenerationSettings,
    SteeringSpec,
    TokenInfo,
)
from backend.app.services.activation_store import ActivationStore, StoredExperiment

MAX_LOAD_GIT_SECONDS = 5
_RUN_ID_HEX_LEN = 32

Condition = Literal["baseline", "steered", "control"]
ConditionDetail = Literal["none", "positive", "negative", "zero", "control"]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=MAX_LOAD_GIT_SECONDS,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - commit is best-effort metadata
        pass
    return "unknown"


def build_environment_info() -> EnvironmentInfo:
    """Collect dependency/platform metadata once per store lifetime."""
    return EnvironmentInfo(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=_package_version("transformers"),
        fastapi_version=_package_version("fastapi"),
        app_version=APP_VERSION,
        platform=platform.platform(),
        git_commit=_git_commit(),
    )


def _condition_detail(
    condition: Condition,
    steering: SteeringSpec | None,
) -> ConditionDetail:
    if condition == "control":
        return "control"
    if condition == "baseline" or steering is None:
        return "none"
    if steering.strength == 0:
        return "zero"
    if steering.strength > 0:
        return "positive"
    return "negative"


def _preview(text: str, max_length: int = 200) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _is_valid_run_id(run_id: str) -> bool:
    return len(run_id) == _RUN_ID_HEX_LEN and all(c in "0123456789abcdef" for c in run_id)


@dataclass
class ExperimentRecord:
    """Internal representation of one persisted experiment run."""

    id: str
    created_at: datetime
    model_id: str
    revision: str
    resolved_revision: str
    device: str
    prompt: str
    system: str | None
    output: str
    prompt_tokens: int
    generated_tokens: int
    duration_ms: float
    tokens_per_second: float | None
    settings: GenerationSettings
    steering: SteeringSpec | None
    captured_modules: list[str]
    captures: list[dict[str, Any]]
    warnings: list[str]
    tokens: list[TokenInfo]
    environment: dict[str, Any]
    condition: Condition
    condition_detail: ConditionDetail
    sweep_id: str | None = None
    sweep_index: int | None = None
    repeat_index: int | None = None
    activation_snapshot_path: str | None = None
    activation_snapshot_size: int | None = None


class ExperimentStore:
    """Thread-safe JSONL metadata store with optional activation snapshots."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        persist_activations_default: bool = False,
        activation_store: ActivationStore | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else Path("data")
        self.persist_activations_default = persist_activations_default
        self.activation_store = activation_store or ActivationStore()
        self._lock = threading.RLock()
        self._records: OrderedDict[str, ExperimentRecord] = OrderedDict()
        self._load_warnings: list[str] = []
        self._environment: EnvironmentInfo | None = None

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._activations_dir = self.data_dir / "activations"
        self._metadata_path = self.data_dir / "experiments.jsonl"
        (self.data_dir / ".gitkeep").touch(exist_ok=True)
        self._load_index()

    @property
    def load_warnings(self) -> list[str]:
        return list(self._load_warnings)

    def environment_info(self) -> EnvironmentInfo:
        if self._environment is None:
            self._environment = build_environment_info()
        return self._environment

    def resolve_persist_activations(self, request_value: bool | None) -> bool:
        return request_value if request_value is not None else self.persist_activations_default

    def _snapshot_path(self, experiment_id: str) -> Path:
        if not _is_valid_run_id(experiment_id):
            raise ValueError(f"Invalid experiment id for snapshot: {experiment_id!r}")
        return self._activations_dir / f"{experiment_id}.npz"

    def _load_index(self) -> None:
        if not self._metadata_path.exists():
            return
        with self._metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = self._record_from_dict(json.loads(line))
                except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
                    self._load_warnings.append(f"Skipped corrupt metadata line {line_number}: {exc}")
                    continue
                self._records[record.id] = record

    def _record_to_dict(self, record: ExperimentRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "created_at": record.created_at.isoformat(),
            "model_id": record.model_id,
            "revision": record.revision,
            "resolved_revision": record.resolved_revision,
            "device": record.device,
            "prompt": record.prompt,
            "system": record.system,
            "output": record.output,
            "prompt_tokens": record.prompt_tokens,
            "generated_tokens": record.generated_tokens,
            "duration_ms": record.duration_ms,
            "tokens_per_second": record.tokens_per_second,
            "settings": record.settings.model_dump(mode="json"),
            "steering": record.steering.model_dump(mode="json") if record.steering else None,
            "captured_modules": record.captured_modules,
            "captures": record.captures,
            "warnings": record.warnings,
            "tokens": [token.model_dump(mode="json") for token in record.tokens],
            "environment": record.environment,
            "condition": record.condition,
            "condition_detail": record.condition_detail,
            "sweep_id": record.sweep_id,
            "sweep_index": record.sweep_index,
            "repeat_index": record.repeat_index,
            "activation_snapshot_path": record.activation_snapshot_path,
            "activation_snapshot_size": record.activation_snapshot_size,
        }

    def _record_from_dict(self, data: dict[str, Any]) -> ExperimentRecord:
        steering_data = data.get("steering")
        return ExperimentRecord(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            model_id=data["model_id"],
            revision=data["revision"],
            resolved_revision=data.get("resolved_revision", data["revision"]),
            device=data["device"],
            prompt=data["prompt"],
            system=data.get("system"),
            output=data["output"],
            prompt_tokens=data["prompt_tokens"],
            generated_tokens=data["generated_tokens"],
            duration_ms=data["duration_ms"],
            tokens_per_second=data.get("tokens_per_second"),
            settings=GenerationSettings(**data["settings"]),
            steering=SteeringSpec(**steering_data) if steering_data else None,
            captured_modules=list(data.get("captured_modules", [])),
            captures=list(data.get("captures", [])),
            warnings=list(data.get("warnings", [])),
            tokens=[TokenInfo(**token) for token in data.get("tokens", [])],
            environment=dict(data.get("environment", {})),
            condition=data["condition"],
            condition_detail=data.get("condition_detail", "none"),
            sweep_id=data.get("sweep_id"),
            sweep_index=data.get("sweep_index"),
            repeat_index=data.get("repeat_index"),
            activation_snapshot_path=data.get("activation_snapshot_path"),
            activation_snapshot_size=data.get("activation_snapshot_size"),
        )

    def _append_jsonl(self, record: ExperimentRecord) -> None:
        line = json.dumps(self._record_to_dict(record), ensure_ascii=False)
        with self._metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _add_to_index(self, record: ExperimentRecord) -> None:
        with self._lock:
            self._records[record.id] = record
            self._records.move_to_end(record.id)

    def _save_snapshot(
        self,
        experiment_id: str,
        captures: dict[str, torch.Tensor],
    ) -> tuple[str, int]:
        self._activations_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshot_path(experiment_id)
        arrays = {module_id: tensor.numpy() for module_id, tensor in captures.items()}
        np.savez_compressed(path, **arrays)
        size = path.stat().st_size
        return path.relative_to(self.data_dir).as_posix(), size

    def _resolve_snapshot_path(self, record: ExperimentRecord) -> Path | None:
        if not record.activation_snapshot_path:
            return None
        path = Path(record.activation_snapshot_path)
        candidate = path if path.is_absolute() else self.data_dir / path
        try:
            candidate.resolve().relative_to(self._activations_dir.resolve())
        except ValueError:
            return None
        return candidate

    def record_run(
        self,
        result,  # Avoid circular import: GenerationResult is defined in generation.py.
        *,
        persist_activations: bool | None = None,
        resolved_revision: str | None = None,
        sweep_id: str | None = None,
        sweep_index: int | None = None,
        repeat_index: int | None = None,
    ) -> ExperimentRecord:
        """Persist metadata for a single run and optionally snapshot activations."""
        from backend.app.services.generation import GenerationResult

        if not isinstance(result, GenerationResult):
            raise TypeError("record_run expects a GenerationResult")

        persist = self.resolve_persist_activations(persist_activations)
        response = result.response
        condition_detail = _condition_detail(response.condition, response.steering)
        environment = self.environment_info().model_dump(mode="json")

        snapshot_path: str | None = None
        snapshot_size: int | None = None
        if result.captures and persist:
            snapshot_path, snapshot_size = self._save_snapshot(response.id, result.captures)

        record = ExperimentRecord(
            id=response.id,
            created_at=response.timestamp,
            model_id=response.model_id,
            revision=response.revision,
            resolved_revision=resolved_revision or response.revision,
            device=response.device,
            prompt=response.prompt,
            system=response.system,
            output=response.output,
            prompt_tokens=response.prompt_tokens,
            generated_tokens=response.generated_tokens,
            duration_ms=response.duration_ms,
            tokens_per_second=response.tokens_per_second,
            settings=response.settings,
            steering=response.steering,
            captured_modules=response.captured_modules,
            captures=[
                {
                    "module_id": module_id,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                }
                for module_id, tensor in result.captures.items()
            ],
            warnings=response.warnings,
            tokens=result.tokens,
            environment=environment,
            condition=response.condition,
            condition_detail=condition_detail,
            sweep_id=sweep_id,
            sweep_index=sweep_index,
            repeat_index=repeat_index,
            activation_snapshot_path=snapshot_path,
            activation_snapshot_size=snapshot_size,
        )

        with self._lock:
            self._add_to_index(record)
            self._append_jsonl(record)
            if result.captures:
                self.activation_store.add(
                    StoredExperiment(
                        id=response.id,
                        created_at=response.timestamp,
                        model_id=response.model_id,
                        revision=response.revision,
                        prompt=response.prompt,
                        system=response.system,
                        output=response.output,
                        prompt_token_count=response.prompt_tokens,
                        generated_tokens=response.generated_tokens,
                        tokens=result.tokens,
                        captures=result.captures,
                        warnings=response.warnings,
                    )
                )
        return record

    def get_run(self, experiment_id: str) -> ExperimentRecord:
        with self._lock:
            try:
                return self._records[experiment_id]
            except KeyError as exc:
                raise KeyError(f"Unknown experiment {experiment_id!r}.") from exc

    def _activation_source(
        self, experiment_id: str, record: ExperimentRecord
    ) -> Literal["live", "snapshot"] | None:
        if self.activation_store.is_live(experiment_id):
            return "live"
        snapshot = self._resolve_snapshot_path(record)
        if snapshot is not None and snapshot.exists():
            return "snapshot"
        return None

    def _summary_from_record(self, record: ExperimentRecord) -> ExperimentSummary:
        source = self._activation_source(record.id, record)
        has_activations = bool(record.captured_modules)
        return ExperimentSummary(
            id=record.id,
            timestamp=record.created_at,
            model_id=record.model_id,
            revision=record.revision,
            resolved_revision=record.resolved_revision,
            device=record.device,
            condition=record.condition,
            condition_detail=record.condition_detail,
            sweep_id=record.sweep_id,
            sweep_index=record.sweep_index,
            repeat_index=record.repeat_index,
            module_id=record.steering.module_id if record.steering else None,
            feature_index=record.steering.feature_index if record.steering else None,
            strength=record.steering.strength if record.steering else None,
            prompt_preview=_preview(record.prompt),
            output_preview=_preview(record.output),
            has_activations=has_activations,
            activation_source=source,
            duration_ms=record.duration_ms,
            tokens_per_second=record.tokens_per_second,
        )

    def list_summaries(
        self,
        *,
        condition: Condition | None = None,
        sweep_id: str | None = None,
        module_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ExperimentListResponse:
        with self._lock:
            ordered = list(reversed(self._records.values()))
        filtered: list[ExperimentRecord] = []
        for record in ordered:
            if condition is not None and record.condition != condition:
                continue
            if sweep_id is not None and record.sweep_id != sweep_id:
                continue
            if module_id is not None:
                if record.steering is None or record.steering.module_id != module_id:
                    continue
            filtered.append(record)
        total = len(filtered)
        page = filtered[offset : offset + limit]
        return ExperimentListResponse(
            total=total,
            offset=offset,
            limit=limit,
            items=[self._summary_from_record(record) for record in page],
        )

    def build_detail(self, experiment_id: str) -> ExperimentDetail:
        record = self.get_run(experiment_id)
        summary = self._summary_from_record(record)
        environment = EnvironmentInfo(**record.environment) if record.environment else self.environment_info()
        return ExperimentDetail(
            **summary.model_dump(),
            system=record.system,
            prompt=record.prompt,
            output=record.output,
            prompt_tokens=record.prompt_tokens,
            generated_tokens=record.generated_tokens,
            settings=record.settings,
            steering=record.steering,
            captured_modules=record.captured_modules,
            captures=[CaptureInfo(**capture) for capture in record.captures],
            warnings=record.warnings,
            environment=environment,
            tokens=record.tokens,
        )

    def restore_activations(self, experiment_id: str) -> StoredExperiment:
        """Load a snapshot into the live activation store if it fits the budget."""
        record = self.get_run(experiment_id)
        if not record.activation_snapshot_path:
            raise KeyError(f"Experiment {experiment_id!r} has no activation snapshot.")
        path = self._resolve_snapshot_path(record)
        if path is None or not path.exists():
            raise KeyError(f"Activation snapshot for {experiment_id!r} is missing.")

        with np.load(path, allow_pickle=False) as archive:
            estimated_bytes = 0
            shapes_dtypes: list[tuple[str, tuple[int, ...], np.dtype]] = []
            for module_id in archive.files:
                array = archive[module_id]
                shapes_dtypes.append((module_id, array.shape, array.dtype))
                estimated_bytes += array.size * array.dtype.itemsize
            if estimated_bytes > self.activation_store.max_bytes:
                raise ValueError(
                    f"Snapshot {experiment_id!r} would require ~{estimated_bytes} bytes, "
                    f"exceeding the live store budget of {self.activation_store.max_bytes} bytes."
                )
            captures = {
                module_id: torch.from_numpy(np.array(archive[module_id])) for module_id in archive.files
            }

        experiment = StoredExperiment(
            id=record.id,
            created_at=record.created_at,
            model_id=record.model_id,
            revision=record.revision,
            prompt=record.prompt,
            system=record.system,
            output=record.output,
            prompt_token_count=record.prompt_tokens,
            generated_tokens=record.generated_tokens,
            tokens=record.tokens,
            captures=captures,
            warnings=record.warnings,
        )
        self.activation_store.add(experiment)
        return experiment

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())

    def export_jsonl(self, experiment_ids: list[str]) -> str:
        with self._lock:
            missing = [eid for eid in experiment_ids if eid not in self._records]
            if missing:
                raise KeyError(f"Unknown experiment ids: {', '.join(missing)}")
            lines = [
                json.dumps(self._record_to_dict(self._records[eid]), ensure_ascii=False)
                for eid in experiment_ids
            ]
        return "\n".join(lines) + ("\n" if lines else "")

    def export_csv(self, experiment_ids: list[str]) -> str:
        import csv
        import io

        with self._lock:
            missing = [eid for eid in experiment_ids if eid not in self._records]
            if missing:
                raise KeyError(f"Unknown experiment ids: {', '.join(missing)}")
            records = [self._records[eid] for eid in experiment_ids]

        def _safe_text(value: Any) -> str:
            if value is None:
                return ""
            text = str(value)
            # Guard against CSV injection from spreadsheet formulae.
            stripped = text.lstrip()
            # Prefix spreadsheet formula leaders. A leading "-" is common in
            # ordinary prose and is left untouched; numeric columns are raw.
            if stripped and stripped[0] in {"=", "+", "@"}:
                text = "'" + text
            return text

        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "id",
                "timestamp",
                "model_id",
                "revision",
                "resolved_revision",
                "condition",
                "condition_detail",
                "sweep_id",
                "sweep_index",
                "repeat_index",
                "module_id",
                "feature_index",
                "strength",
                "prompt",
                "output",
                "prompt_tokens",
                "generated_tokens",
                "duration_ms",
                "tokens_per_second",
                "has_activations",
                "activation_source",
                "captured_modules",
                "warnings_count",
            ]
        )

        def _raw(value: Any) -> Any:
            return "" if value is None else value

        for record in records:
            source = self._activation_source(record.id, record)
            writer.writerow(
                [
                    _safe_text(record.id),
                    _safe_text(record.created_at.isoformat()),
                    _safe_text(record.model_id),
                    _safe_text(record.revision),
                    _safe_text(record.resolved_revision),
                    _safe_text(record.condition),
                    _safe_text(record.condition_detail),
                    _safe_text(record.sweep_id),
                    _raw(record.sweep_index),
                    _raw(record.repeat_index),
                    _safe_text(record.steering.module_id if record.steering else None),
                    _raw(record.steering.feature_index if record.steering else None),
                    _raw(record.steering.strength if record.steering else None),
                    _safe_text(record.prompt),
                    _safe_text(record.output),
                    _raw(record.prompt_tokens),
                    _raw(record.generated_tokens),
                    _raw(record.duration_ms),
                    _raw(record.tokens_per_second),
                    bool(record.captured_modules),
                    _safe_text(source),
                    _safe_text(",".join(record.captured_modules)),
                    len(record.warnings),
                ]
            )
        return output.getvalue()


def make_experiment_store(settings=None) -> ExperimentStore:
    """Factory used by the application default wiring."""
    if settings is None:
        from backend.app.core.settings import settings as default_settings

        settings = default_settings
    return ExperimentStore(
        data_dir=settings.data_dir,
        persist_activations_default=settings.persist_activations,
    )
