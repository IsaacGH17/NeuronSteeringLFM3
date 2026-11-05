"""Directions from two runs, and the background batch matrix."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _run(client: TestClient, prompt: str, capture: list[str] | None = None) -> dict:
    response = client.post(
        "/api/experiments/baseline",
        json={
            "prompt": prompt,
            "settings": {"max_new_tokens": 2},
            "capture_module_ids": capture or [],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _wait(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/experiments/batch/{job_id}").json()
        if body["state"] in {"completed", "cancelled", "failed"}:
            return body
        time.sleep(0.02)
    raise AssertionError(f"batch {job_id} did not finish: {body}")


def _plan(**overrides) -> dict:
    plan = {
        "module_ids": ["model.layers.0.feed_forward"],
        "components": [{"feature_index": 3, "weight": 1.0}],
        "strengths": [-1.0, 1.0],
        "repeats": 1,
    }
    plan.update(overrides)
    return plan


def test_batch_runs_matrix_and_persists_every_run(client: TestClient):
    start = client.post(
        "/api/experiments/batch",
        json={
            "prompts": ["Say A", "Say B"],
            "settings": {"max_new_tokens": 2},
            "plan": _plan(),
        },
    )
    assert start.status_code == 202, start.text
    job_id = start.json()["job_id"]
    job = _wait(client, job_id)

    assert job["state"] == "completed", job
    assert job["total"] == 6
    assert job["completed"] == 6
    kinds = {(cell["kind"], cell["strength"]) for cell in job["cells"]}
    assert ("baseline", None) in kinds
    assert ("steered", -1.0) in kinds and ("steered", 1.0) in kinds
    baseline_cell = next(cell for cell in job["cells"] if cell["kind"] == "baseline")
    assert baseline_cell["runs"] == 2
    assert "warmth" in baseline_cell["lexicon_mean"]

    history = client.get("/api/experiments", params={"sweep_id": job_id}).json()
    assert history["total"] == 6
    assert all(item["metrics"] is not None for item in history["items"])
    steered_summaries = [item for item in history["items"] if item["steering_summary"]]
    assert steered_summaries
    assert "dim 3" in steered_summaries[0]["steering_summary"]

    detail = client.get(f"/api/experiments/{history['items'][0]['id']}").json()
    assert detail["metrics"] is not None
    assert detail["metrics"]["words"] >= 0
    assert detail["sweep_id"] == job_id


def test_batch_include_runs_returns_records(client: TestClient):
    job_id = client.post(
        "/api/experiments/batch",
        json={"prompts": ["One"], "settings": {"max_new_tokens": 1}, "plan": _plan()},
    ).json()["job_id"]
    _wait(client, job_id)
    body = client.get(f"/api/experiments/batch/{job_id}", params={"include_runs": "true"}).json()
    assert len(body["runs"]) == body["total"] == 3
    assert {record["prompt_id"] for record in body["runs"]} == {"p1"}
    assert {record["kind"] for record in body["runs"]} == {"baseline", "steered"}


def test_batch_control_and_repeats(client: TestClient):
    job_id = client.post(
        "/api/experiments/batch",
        json={
            "prompts": ["Control"],
            "settings": {"max_new_tokens": 1, "seed": 7},
            "plan": _plan(
                strengths=[2.0],
                repeats=2,
                include_control=True,
            ),
        },
    ).json()["job_id"]
    job = _wait(client, job_id)
    assert job["state"] == "completed", job
    assert job["total"] == 1 + 2 + 2  # baseline + 2 repeats + 2 controls
    assert any(cell["kind"] == "control" for cell in job["cells"])


def test_batch_with_a_suite_attaches_checks(client: TestClient):
    job_id = client.post(
        "/api/experiments/batch",
        json={
            "suite_id": "capability-core",
            "settings": {"max_new_tokens": 2},
            "plan": _plan(strengths=[0.5], baseline_per_prompt=False),
        },
    ).json()["job_id"]
    job = _wait(client, job_id)
    assert job["state"] == "completed", job
    assert job["total"] == 10
    cell = job["cells"][0]
    assert cell["checks_passed_rate"] is not None


def test_batch_can_be_cancelled(client: TestClient):
    big_plan = _plan(
        module_ids=["model.layers.0.feed_forward", "model.layers.1.feed_forward"],
        strengths=[-2.0, -1.0, 1.0, 2.0],
    )
    start = client.post(
        "/api/experiments/batch",
        json={
            "prompts": [f"Prompt {index}" for index in range(12)],
            "settings": {"max_new_tokens": 4},
            "plan": big_plan,
        },
    )
    assert start.status_code == 202, start.text
    job_id = start.json()["job_id"]
    cancelled = client.post(f"/api/experiments/batch/{job_id}/cancel")
    assert cancelled.status_code == 200
    job = _wait(client, job_id)
    assert job["state"] == "cancelled", job
    assert job["completed"] < job["total"]
    assert any("Cancelled" in warning or "Cancellation" in warning for warning in job["warnings"])


def test_only_one_batch_runs_at_a_time(client: TestClient):
    big_plan = _plan(
        module_ids=["model.layers.0.feed_forward", "model.layers.1.feed_forward"],
        strengths=[-2.0, -1.0, 1.0, 2.0],
    )
    payload = {
        "prompts": [f"Busy {index}" for index in range(12)],
        "settings": {"max_new_tokens": 4},
        "plan": big_plan,
    }
    first = client.post("/api/experiments/batch", json=payload)
    assert first.status_code == 202
    second = client.post("/api/experiments/batch", json=payload)
    assert second.status_code == 400
    assert "still running" in second.json()["detail"] or "still queued" in second.json()["detail"]
    client.post(f"/api/experiments/batch/{first.json()['job_id']}/cancel")
    _wait(client, first.json()["job_id"])


def test_batch_validation_errors(client: TestClient):
    both_sources = client.post(
        "/api/experiments/batch",
        json={"prompts": ["x"], "suite_id": "trait-neutral", "plan": _plan()},
    )
    assert both_sources.status_code == 422

    unknown_suite = client.post("/api/experiments/batch", json={"suite_id": "nope", "plan": _plan()})
    assert unknown_suite.status_code == 400
    assert "Unknown prompt suite" in unknown_suite.json()["detail"]

    too_big = client.post(
        "/api/experiments/batch",
        json={
            "prompts": [f"Big {index}" for index in range(64)],
            "plan": _plan(
                module_ids=["model.layers.0.feed_forward", "model.layers.1.feed_forward"],
                strengths=[float(value) for value in range(1, 33)],
                repeats=10,
            ),
        },
    )
    assert too_big.status_code == 400
    assert "limit is" in too_big.json()["detail"]


def test_direction_from_two_captured_runs(client: TestClient):
    warm = _run(client, "Describe a rainy morning warmly.", ["model.layers.0.feed_forward"])
    cold = _run(client, "Describe a rainy morning clinically.", ["model.layers.0.feed_forward"])
    response = client.post(
        "/api/experiments/direction",
        json={
            "positive_experiment_id": warm["id"],
            "negative_experiment_id": cold["id"],
            "top_k": 4,
            "position_mode": "mean_prompt",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["module_id"] == "model.layers.0.feed_forward"
    assert len(body["components"]) == 4
    assert len(body["sigma"]) == 4
    assert all(component["weight"] != 0 for component in body["components"])
    assert body["positions_compared"] > 0
    assert body["norm"] > 0


def test_direction_requires_captures(client: TestClient):
    first = _run(client, "No capture")
    second = _run(client, "Also none")
    response = client.post(
        "/api/experiments/direction",
        json={"positive_experiment_id": first["id"], "negative_experiment_id": second["id"]},
    )
    assert response.status_code == 404

    unknown = client.post(
        "/api/experiments/direction",
        json={"positive_experiment_id": "0" * 32, "negative_experiment_id": "1" * 32},
    )
    assert unknown.status_code == 404
