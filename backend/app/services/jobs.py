"""Background batch runner: modules x strengths x prompts, with progress.

A batch is a matrix of ordinary experiments, executed one at a time on the
single loaded model. Each run is persisted as soon as it finishes (so partial
results survive a failure or a cancellation), and the job exposes progress for
polling. Cancellation is cooperative: the in-flight generation finishes, then
the loop stops before the next run.

Only one batch may run at a time; single experiments still queue behind the
model's inference lock, so the GPU never sees two generations at once.
"""

from __future__ import annotations

import math
import random
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.app.schemas.api import (
    BaselineRequest,
    BaselineResponse,
    BatchCellSummary,
    BatchJobResponse,
    BatchRequest,
    BatchRunRecord,
    SteeringSpec,
    SuiteCheck,
)
from backend.app.services.activation_capture import validate_capture_module_ids
from backend.app.services.experiment_store import ExperimentStore
from backend.app.services.generation import generate_baseline, generate_chat, generate_steered
from backend.app.services.metrics import aggregate_metrics
from backend.app.services.model_loader import ModelManager
from backend.app.services.steering_engine import SteeringSpec as EngineSteeringSpec
from backend.app.services.steering_engine import validate_steering_spec
from backend.app.services.suites import get_suite

BatchKind = str


@dataclass
class _PromptEntry:
    prompt_id: str
    text: str
    checks: list[SuiteCheck]


@dataclass
class _Job:
    id: str
    total: int
    plan: str
    prompts: list[_PromptEntry]
    payload: BatchRequest
    state: str = "queued"
    completed: int = 0
    current: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    runs: list[BatchRunRecord] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)


def _resolve_prompts(payload: BatchRequest) -> list[_PromptEntry]:
    if payload.suite_id:
        suite = get_suite(payload.suite_id)
        return [
            _PromptEntry(prompt_id=prompt.id, text=prompt.text, checks=list(prompt.checks))
            for prompt in suite.prompts
        ]
    return [
        _PromptEntry(prompt_id=f"p{index + 1}", text=text, checks=[])
        for index, text in enumerate(payload.prompts)
    ]


def _plan_label(payload: BatchRequest, prompts: list[_PromptEntry]) -> str:
    plan = payload.plan
    per_prompt = len(plan.strengths) + (1 if plan.include_control else 0)
    total = len(prompts) * (1 if plan.baseline_per_prompt else 0) + (
        len(prompts) * len(plan.module_ids) * per_prompt * plan.repeats
    )
    suite = payload.suite_id or f"{len(prompts)} custom prompts"
    return (
        f"{suite} · {len(plan.module_ids)} module(s) × {len(plan.strengths)} strength(s) "
        f"× {plan.repeats} repeat(s){' + control' if plan.include_control else ''} "
        f"= {total} runs"
    )


class BatchJobRegistry:
    def __init__(self, manager: ModelManager, store: ExperimentStore) -> None:
        self.manager = manager
        self.store = store
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._active: str | None = None

    # ------------------------------------------------------------------ api
    def start(self, payload: BatchRequest) -> BatchJobResponse:
        prompts = _resolve_prompts(payload)
        if not prompts:
            raise ValueError("A batch needs at least one prompt.")
        plan = payload.plan
        per_prompt = len(plan.strengths) + (1 if plan.include_control else 0)
        total = len(prompts) * (1 if plan.baseline_per_prompt else 0) + (
            len(prompts) * len(plan.module_ids) * per_prompt * plan.repeats
        )
        if total > self.manager.settings.max_batch_runs:
            raise ValueError(
                f"This batch would run {total} generations; the limit is "
                f"{self.manager.settings.max_batch_runs}. Reduce prompts, modules, strengths, or repeats."
            )
        if total == 0:
            raise ValueError("This batch would run nothing. Enable a baseline or add strengths.")

        self._validate(payload)

        with self._lock:
            if self._active is not None:
                active = self._jobs.get(self._active)
                if active is not None and active.state in {"queued", "running"}:
                    raise ValueError(
                        f"Batch {self._active} is still {active.state}; wait for it or cancel it first."
                    )
            job = _Job(
                id=uuid.uuid4().hex,
                total=total,
                plan=_plan_label(payload, prompts),
                prompts=prompts,
                payload=payload,
            )
            self._jobs[job.id] = job
            self._active = job.id
            jobs = list(self._jobs.values())

        # Keep memory bounded: drop finished jobs beyond the most recent 20.
        if len(jobs) > 20:
            with self._lock:
                for stale in [
                    item for item in self._jobs.values() if item.state not in {"queued", "running"}
                ][:-20]:
                    self._jobs.pop(stale.id, None)

        thread = threading.Thread(target=self._run, args=(job,), name=f"batch-{job.id[:8]}", daemon=True)
        thread.start()
        return self._snapshot(job, include_runs=False)

    def get(self, job_id: str, *, include_runs: bool = False) -> BatchJobResponse:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown batch job {job_id!r}.")
            return self._snapshot(job, include_runs=include_runs)

    def cancel(self, job_id: str) -> BatchJobResponse:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown batch job {job_id!r}.")
            if job.state in {"queued", "running"}:
                job.cancel.set()
                if job.state == "queued":
                    job.state = "cancelled"
                    job.finished_at = datetime.now(UTC)
                else:
                    job.warnings.append(
                        "Cancellation requested: the current generation finishes, then the batch stops."
                    )
            return self._snapshot(job, include_runs=False)

    # ------------------------------------------------------------- internals
    def _validate(self, payload: BatchRequest) -> None:
        plan = payload.plan
        capture_ids = validate_capture_module_ids(self.manager.model, payload.capture_module_ids)
        components = tuple(
            (component.feature_index, float(component.weight)) for component in plan.components
        )
        if not components:
            raise ValueError("A batch needs at least one steering dimension.")
        for module_id in plan.module_ids:
            for strength in plan.strengths:
                validate_steering_spec(
                    self.manager.model,
                    EngineSteeringSpec(
                        module_id=module_id,
                        feature_index=None,
                        strength=strength,
                        components=components,
                        normalization=plan.normalization,
                    ),
                    strength_limit=self.manager.settings.max_steering_strength,
                )
        del capture_ids

    def _record(self, job: _Job, result, *, sweep_index: int, repeat_index: int) -> None:
        config = getattr(self.manager.model, "config", None)
        resolved = getattr(config, "_commit_hash", None) or result.response.revision
        self.store.record_run(
            result,
            persist_activations=job.payload.persist_activations,
            resolved_revision=resolved,
            sweep_id=job.id,
            sweep_index=sweep_index,
            repeat_index=repeat_index,
        )

    def _run(self, job: _Job) -> None:
        payload = job.payload
        plan = payload.plan
        with self._lock:
            job.state = "running"
            job.started_at = datetime.now(UTC)
        try:
            control_seed = plan.control_seed
            if control_seed is None and payload.settings.seed is not None:
                control_seed = payload.settings.seed
            if control_seed is None:
                control_seed = secrets.randbelow(2**31)
            control_strength = math.copysign(1.0, plan.strengths[-1]) * max(
                abs(strength) for strength in plan.strengths
            )
            target_indices = {component.feature_index for component in plan.components}
            hidden_size = self._hidden_size()

            index = 0
            for prompt in job.prompts:
                if job.cancel.is_set():
                    break
                request = BaselineRequest(
                    prompt=prompt.text,
                    system=payload.system,
                    settings=payload.settings,
                    capture_module_ids=payload.capture_module_ids,
                )
                if plan.baseline_per_prompt:
                    with self._lock:
                        job.current = f"{prompt.prompt_id} · baseline"
                    baseline = generate_baseline(
                        self.manager,
                        request,
                        capture_ids=payload.capture_module_ids,
                        checks=prompt.checks,
                    )
                    self._record(job, baseline, sweep_index=-1, repeat_index=0)
                    self._append(job, index, "baseline", -1, 0, prompt, baseline.response)
                    index += 1
                    self._advance(job)

                for module_id in plan.module_ids:
                    for strength_index, strength in enumerate(plan.strengths):
                        for repeat_index in range(plan.repeats):
                            if job.cancel.is_set():
                                break
                            with self._lock:
                                job.current = f"{prompt.prompt_id} · {module_id} · α {strength}"
                            steering = SteeringSpec(
                                module_id=module_id,
                                strength=strength,
                                components=[
                                    {
                                        "feature_index": component.feature_index,
                                        "weight": component.weight,
                                    }
                                    for component in plan.components
                                ],
                                normalization=plan.normalization,
                            )
                            steered = generate_steered(
                                self.manager,
                                request,
                                steering,
                                capture_ids=payload.capture_module_ids,
                                checks=prompt.checks,
                            )
                            self._record(job, steered, sweep_index=strength_index, repeat_index=repeat_index)
                            self._append(
                                job, index, "steered", strength_index, repeat_index, prompt, steered.response
                            )
                            index += 1
                            self._advance(job)

                    if plan.include_control:
                        candidates = [dim for dim in range(hidden_size) if dim not in target_indices]
                        if not candidates:
                            raise ValueError("A control needs at least one unsteered dimension.")
                        rng = random.Random(control_seed)
                        control_feature = candidates[rng.randrange(len(candidates))]
                        for repeat_index in range(plan.repeats):
                            if job.cancel.is_set():
                                break
                            with self._lock:
                                job.current = (
                                    f"{prompt.prompt_id} · {module_id} · control α {control_strength}"
                                )
                            control_spec = SteeringSpec(
                                module_id=module_id,
                                strength=control_strength,
                                components=[{"feature_index": control_feature, "weight": 1.0}],
                                normalization=plan.normalization,
                            )
                            control = generate_chat(
                                self.manager,
                                request,
                                capture_ids=payload.capture_module_ids,
                                steering=control_spec,
                                condition="control",
                                checks=prompt.checks,
                            )
                            self._record(
                                job, control, sweep_index=len(plan.strengths), repeat_index=repeat_index
                            )
                            self._append(
                                job,
                                index,
                                "control",
                                len(plan.strengths),
                                repeat_index,
                                prompt,
                                control.response,
                            )
                            index += 1
                            self._advance(job)

            with self._lock:
                job.state = "cancelled" if job.cancel.is_set() else "completed"
                job.current = None
                job.finished_at = datetime.now(UTC)
                if job.cancel.is_set():
                    job.warnings.append(
                        f"Cancelled after {job.completed} of {job.total} runs; completed runs are saved."
                    )
        except Exception as exc:  # noqa: BLE001 - surfaced through the job payload
            with self._lock:
                job.state = "failed"
                job.error = str(exc)
                job.current = None
                job.finished_at = datetime.now(UTC)
                job.warnings.append(f"Failed after {job.completed} of {job.total} runs.")
        finally:
            with self._lock:
                if self._active == job.id:
                    self._active = None

    def _hidden_size(self) -> int:
        config = self.manager.model.config
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        return int(text_config.hidden_size)

    def _append(
        self,
        job: _Job,
        index: int,
        kind: str,
        sweep_index: int,
        repeat_index: int,
        prompt: _PromptEntry,
        response: BaselineResponse,
    ) -> None:
        record = BatchRunRecord(
            index=index,
            kind=kind,  # type: ignore[arg-type]
            sweep_index=sweep_index,
            repeat_index=repeat_index,
            prompt_id=prompt.prompt_id,
            prompt=prompt.text,
            run=response,
        )
        with self._lock:
            job.runs.append(record)

    def _advance(self, job: _Job) -> None:
        with self._lock:
            job.completed += 1

    def _snapshot(self, job: _Job, *, include_runs: bool) -> BatchJobResponse:
        cells = self._cells(job.runs)
        return BatchJobResponse(
            job_id=job.id,
            state=job.state,  # type: ignore[arg-type]
            total=job.total,
            completed=job.completed,
            current=job.current,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            plan=job.plan,
            warnings=list(job.warnings),
            runs=list(job.runs) if include_runs else [],
            cells=cells,
        )

    def _cells(self, runs: list[BatchRunRecord]) -> list[BatchCellSummary]:
        groups: dict[tuple[str, str | None, float | None], list[Any]] = {}
        for record in runs:
            key = (
                record.kind,
                record.run.steering.module_id if record.run.steering else None,
                record.run.steering.strength if record.run.steering else None,
            )
            groups.setdefault(key, []).append(record.run)
        cells: list[BatchCellSummary] = []
        for (kind, module_id, strength), entries in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2] or 0.0)
        ):
            aggregated = aggregate_metrics([entry.metrics for entry in entries])
            cells.append(
                BatchCellSummary(
                    kind=kind,  # type: ignore[arg-type]
                    module_id=module_id,
                    strength=strength,
                    runs=len(entries),
                    words_mean=aggregated["words_mean"],
                    distinct_3_mean=aggregated["distinct_3_mean"],
                    refusal_rate=aggregated["refusal_rate"],
                    checks_passed_rate=aggregated["checks_passed_rate"],
                    lexicon_mean=aggregated["lexicon_mean"],
                )
            )
        return cells
