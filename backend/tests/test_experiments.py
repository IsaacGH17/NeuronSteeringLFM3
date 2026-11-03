"""Experiment persistence, export, sweep, and snapshot behavior."""

from __future__ import annotations

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from backend.app.services.experiment_store import ExperimentStore


def _run_baseline(client: TestClient, prompt: str = "Hello", **kwargs):
    payload = {"prompt": prompt, "settings": {"max_new_tokens": 3}}
    payload.update(kwargs)
    response = client.post("/api/experiments/baseline", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _run_steering(client: TestClient, module_id: str, feature_index: int, strength: float, **kwargs):
    payload = {
        "prompt": "Hello",
        "settings": {"max_new_tokens": 3},
        "steering": {"module_id": module_id, "feature_index": feature_index, "strength": strength},
    }
    payload.update(kwargs)
    response = client.post("/api/experiments/steering", json=payload)
    return response


def test_every_run_recorded_including_without_captures(client: TestClient):
    body = _run_baseline(client, prompt="No captures")
    experiment_id = body["id"]

    listing = client.get("/api/experiments").json()
    assert any(item["id"] == experiment_id for item in listing["items"])
    item = next(item for item in listing["items"] if item["id"] == experiment_id)
    assert item["condition"] == "baseline"
    assert item["condition_detail"] == "none"
    assert item["has_activations"] is False
    assert item["activation_source"] is None


def test_persistence_and_restart_visibility(experiment_store: ExperimentStore, client: TestClient):
    body = _run_baseline(client, prompt="Persist me")
    experiment_id = body["id"]

    # Re-open the store from the same directory.
    reloaded = ExperimentStore(data_dir=experiment_store.data_dir)
    record = reloaded.get_run(experiment_id)
    assert record.prompt == "Persist me"
    assert experiment_id in [item.id for item in reloaded.list_summaries().items]


def test_corrupt_tail_tolerance(experiment_store: ExperimentStore, client: TestClient):
    body = _run_baseline(client, prompt="Valid")
    experiment_id = body["id"]

    metadata_path = experiment_store.data_dir / "experiments.jsonl"
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
        handle.write('{"id": "incomplete"}\n')

    reloaded = ExperimentStore(data_dir=experiment_store.data_dir)
    assert experiment_id in reloaded._records
    assert any("Skipped corrupt" in warning for warning in reloaded.load_warnings)


def test_history_filters_and_pagination(client: TestClient):
    _run_baseline(client, prompt="Filter baseline")
    steering = _run_steering(
        client, "model.layers.0.feed_forward", 5, 2.0, capture_module_ids=["model.layers.0.feed_forward"]
    )
    assert steering.status_code == 200
    steered_id = steering.json()["steered"]["id"]

    all_items = client.get("/api/experiments?limit=200").json()
    assert all_items["total"] >= 2

    filtered = client.get("/api/experiments?condition=steered").json()
    assert all(item["condition"] == "steered" for item in filtered["items"])
    assert any(item["id"] == steered_id for item in filtered["items"])

    by_module = client.get("/api/experiments?module_id=model.layers.0.feed_forward").json()
    assert all(item["module_id"] == "model.layers.0.feed_forward" for item in by_module["items"])

    page = client.get("/api/experiments?limit=1&offset=0").json()
    assert len(page["items"]) == 1
    assert page["limit"] == 1
    assert page["offset"] == 0


def test_detail_and_404(client: TestClient):
    body = _run_baseline(client, prompt="Detail test")
    experiment_id = body["id"]

    detail = client.get(f"/api/experiments/{experiment_id}").json()
    assert detail["id"] == experiment_id
    assert detail["prompt"] == "Detail test"
    assert detail["output"] == body["output"]
    assert "environment" in detail
    assert "python_version" in detail["environment"]
    assert detail["resolved_revision"] == body["revision"]

    missing = client.get("/api/experiments/00000000000000000000000000000000")
    assert missing.status_code == 404


def test_detail_records_tensor_shape_and_dtype(client: TestClient):
    body = _run_baseline(client, prompt="Shape metadata", capture_module_ids=["model.layers.0"])
    detail = client.get(f"/api/experiments/{body['id']}").json()
    assert len(detail["captures"]) == 1
    capture = detail["captures"][0]
    assert capture["module_id"] == "model.layers.0"
    assert capture["shape"][1] == 128
    assert capture["dtype"] == "float32"


def test_jsonl_export(client: TestClient):
    body = _run_baseline(client, prompt="Export me")
    experiment_id = body["id"]

    response = client.get(f"/api/experiments/export?format=jsonl&ids={experiment_id}")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert "experiments.jsonl" in response.headers["Content-Disposition"]
    lines = response.text.strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == experiment_id
    assert record["prompt"] == "Export me"


def test_csv_export_and_injection_guard(client: TestClient):
    body = _run_baseline(client, prompt="=SUM(A1:A9)")
    experiment_id = body["id"]

    response = client.get(f"/api/experiments/export?format=csv&ids={experiment_id}")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert "experiments.csv" in response.headers["Content-Disposition"]

    reader = csv.reader(io.StringIO(response.text))
    rows = list(reader)
    assert rows[0][0] == "id"
    data_row = rows[1]
    prompt_cell = data_row[rows[0].index("prompt")]
    assert prompt_cell.startswith("'") or prompt_cell.startswith("\"'")


def test_csv_export_preserves_negative_numbers(client: TestClient):
    steering = _run_steering(client, "model.layers.0.feed_forward", 5, -4.0)
    assert steering.status_code == 200
    steered_id = steering.json()["steered"]["id"]

    sweep = client.post(
        "/api/experiments/sweep",
        json={
            "prompt": "Negative sweep",
            "settings": {"max_new_tokens": 2},
            "module_id": "model.layers.0.feed_forward",
            "feature_index": 5,
            "strengths": [-3.0],
        },
    )
    assert sweep.status_code == 200
    baseline_id = sweep.json()["runs"][0]["run"]["id"]

    response = client.get(f"/api/experiments/export?format=csv&ids={steered_id},{baseline_id}")
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = {row["id"]: row for row in reader}
    assert float(rows[steered_id]["strength"]) == -4.0
    assert int(rows[baseline_id]["sweep_index"]) == -1
    assert rows[steered_id]["condition"] == "steered"
    assert rows[baseline_id]["condition"] == "baseline"


def test_export_all_returns_every_record(client: TestClient):
    first = _run_baseline(client, prompt="Export all one")
    second = _run_baseline(client, prompt="Export all two")
    response = client.get("/api/experiments/export?format=jsonl")
    assert response.status_code == 200
    ids = {json.loads(line)["id"] for line in response.text.strip().splitlines()}
    assert {first["id"], second["id"]}.issubset(ids)


def test_snapshot_path_outside_data_dir_is_rejected(
    client: TestClient, experiment_store: ExperimentStore, tmp_path
):
    body = _run_baseline(client, prompt="Poisoned path", persist_activations=True)
    record = experiment_store.get_run(body["id"])
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"not an archive")
    record.activation_snapshot_path = str(outside)
    with pytest.raises(KeyError, match="missing"):
        experiment_store.restore_activations(body["id"])


def test_export_unknown_ids_returns_400(client: TestClient):
    response = client.get("/api/experiments/export?format=jsonl&ids=abc,def")
    assert response.status_code == 400
    assert "Unknown" in response.json()["detail"]


def test_export_route_not_swallowed_by_detail(client: TestClient):
    _run_baseline(client, prompt="Export all")
    response = client.get("/api/experiments/export?format=jsonl")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/jsonl")
    assert "attachment" in response.headers["Content-Disposition"]


def test_sweep_count_order_and_roles(client: TestClient):
    payload = {
        "prompt": "Sweep test",
        "settings": {"max_new_tokens": 3, "temperature": 0},
        "module_id": "model.layers.0.feed_forward",
        "feature_index": 7,
        "strengths": [2.0, -3.0],
        "repeats": 2,
        "include_control": True,
    }
    response = client.post("/api/experiments/sweep", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    runs = body["runs"]
    assert len(runs) == 7  # baseline + 2 repeats * (2 strengths + 1 control)

    # Ordering check.
    assert runs[0]["kind"] == "baseline"
    assert runs[0]["sweep_index"] == -1
    assert runs[0]["repeat_index"] == 0

    expected = [
        ("strength", 0, 0),
        ("strength", 1, 0),
        ("control", 2, 0),
        ("strength", 0, 1),
        ("strength", 1, 1),
        ("control", 2, 1),
    ]
    for run, (kind, sweep_index, repeat_index) in zip(runs[1:], expected, strict=False):
        assert run["kind"] == kind
        assert run["sweep_index"] == sweep_index
        assert run["repeat_index"] == repeat_index

    # All runs share the same sweep id.
    sweep_id = body["sweep_id"]
    listing = client.get(f"/api/experiments?sweep_id={sweep_id}").json()
    assert listing["total"] == 7


def test_sweep_control_feature_differs_from_target_and_is_seeded(client: TestClient):
    payload = {
        "prompt": "Control seed",
        "settings": {"max_new_tokens": 3, "temperature": 0, "seed": 42},
        "module_id": "model.layers.0.feed_forward",
        "feature_index": 5,
        "strengths": [1.0, 2.0],
        "repeats": 1,
        "include_control": True,
        "control_seed": 99,
    }
    first = client.post("/api/experiments/sweep", json=payload).json()
    second = client.post("/api/experiments/sweep", json=payload).json()

    control_runs = [r for r in first["runs"] if r["kind"] == "control"]
    control_runs_2 = [r for r in second["runs"] if r["kind"] == "control"]
    assert len(control_runs) == 1
    assert control_runs[0]["run"]["steering"]["feature_index"] != 5
    assert (
        control_runs[0]["run"]["steering"]["feature_index"]
        == control_runs_2[0]["run"]["steering"]["feature_index"]
    )


def test_sweep_control_matches_max_magnitude_and_last_sign(client: TestClient):
    payload = {
        "prompt": "Control magnitude",
        "settings": {"max_new_tokens": 3, "temperature": 0},
        "module_id": "model.layers.0.feed_forward",
        "feature_index": 5,
        "strengths": [1.0, -4.0],
        "repeats": 1,
        "include_control": True,
    }
    body = client.post("/api/experiments/sweep", json=payload).json()
    control = next(r for r in body["runs"] if r["kind"] == "control")
    assert control["run"]["steering"]["strength"] == -4.0


def test_sweep_control_actually_steers_the_random_feature(client: TestClient):
    module_id = "model.layers.0.feed_forward"
    payload = {
        "prompt": "Control hook",
        "settings": {"max_new_tokens": 3, "temperature": 0, "seed": 7},
        "module_id": module_id,
        "feature_index": 5,
        "strengths": [4.0],
        "repeats": 1,
        "include_control": True,
        "control_seed": 99,
        "capture_module_ids": [module_id],
    }
    body = client.post("/api/experiments/sweep", json=payload).json()
    baseline = next(run for run in body["runs"] if run["kind"] == "baseline")["run"]
    control = next(run for run in body["runs"] if run["kind"] == "control")["run"]
    control_feature = control["steering"]["feature_index"]
    assert control_feature != 5
    assert control["steering"]["strength"] == 4.0

    def _series(run_id: str) -> list[float]:
        response = client.get(
            f"/api/experiments/{run_id}/activations/{module_id}",
            params={"feature": control_feature, "position": 0},
        )
        assert response.status_code == 200, response.text
        return response.json()["feature_series"]["values"]

    difference = [
        control_value - base_value
        for control_value, base_value in zip(_series(control["id"]), _series(baseline["id"]), strict=True)
    ]
    assert difference
    for value in difference:
        assert value == pytest.approx(4.0, abs=0.02)


def test_sweep_greedy_repeat_warning(client: TestClient):
    payload = {
        "prompt": "Greedy repeats",
        "settings": {"max_new_tokens": 3, "temperature": 0, "seed": 1},
        "module_id": "model.layers.0.feed_forward",
        "feature_index": 5,
        "strengths": [1.0],
        "repeats": 2,
        "include_control": False,
    }
    body = client.post("/api/experiments/sweep", json=payload).json()
    assert any("Greedy" in warning for warning in body["warnings"])
    # Repeated greedy strength runs should be identical.
    strength_runs = [r for r in body["runs"] if r["kind"] == "strength"]
    assert strength_runs[0]["run"]["output"] == strength_runs[1]["run"]["output"]


def test_sweep_validation_errors(client: TestClient):
    # Too many strengths -> 422 from Pydantic.
    payload = {
        "prompt": "Too many",
        "module_id": "model.layers.0.feed_forward",
        "feature_index": 5,
        "strengths": list(range(10)),
    }
    response = client.post("/api/experiments/sweep", json=payload)
    assert response.status_code == 422

    # Strength exceeds safety limit -> 400.
    payload = {
        "prompt": "Too strong",
        "module_id": "model.layers.0.feed_forward",
        "feature_index": 5,
        "strengths": [1000.0],
    }
    response = client.post("/api/experiments/sweep", json=payload)
    assert response.status_code == 400


def test_snapshot_save_restore_and_default_disabled(experiment_store: ExperimentStore, client: TestClient):
    body = _run_baseline(
        client,
        prompt="Snapshot",
        capture_module_ids=["model.layers.0"],
        persist_activations=True,
    )
    experiment_id = body["id"]

    snapshot_path = experiment_store.data_dir / "activations" / f"{experiment_id}.npz"
    assert snapshot_path.exists()

    # Default-disabled run should not create a snapshot.
    no_snapshot = _run_baseline(client, prompt="No snapshot", capture_module_ids=["model.layers.0"])
    assert not (experiment_store.data_dir / "activations" / f"{no_snapshot['id']}.npz").exists()

    # Clear live store and reload from disk.
    new_store = ExperimentStore(data_dir=experiment_store.data_dir)
    assert experiment_id not in new_store.activation_store._entries
    restored = new_store.restore_activations(experiment_id)
    assert "model.layers.0" in restored.captures
    assert experiment_id in new_store.activation_store._entries


def test_activation_route_restores_snapshot(experiment_store: ExperimentStore, client: TestClient):
    body = _run_baseline(
        client,
        prompt="Restore via API",
        capture_module_ids=["model.layers.0"],
        persist_activations=True,
    )
    experiment_id = body["id"]

    new_store = ExperimentStore(data_dir=experiment_store.data_dir)
    new_client = TestClient(client.app)
    # Swap the app state to the new store so the route uses it.
    new_client.app.state.experiment_store = new_store

    activation_list = new_client.get(f"/api/experiments/{experiment_id}/activations")
    assert activation_list.status_code == 200
    assert any(item["module_id"] == "model.layers.0" for item in activation_list.json())

    tensor = new_client.get(f"/api/experiments/{experiment_id}/activations/model.layers.0?position=0")
    assert tensor.status_code == 200
    assert tensor.json()["module_id"] == "model.layers.0"


def test_known_run_with_no_captures_returns_empty_and_tensor_404(client: TestClient):
    body = _run_baseline(client, prompt="No captures")
    experiment_id = body["id"]

    activation_list = client.get(f"/api/experiments/{experiment_id}/activations")
    assert activation_list.status_code == 200
    assert activation_list.json() == []

    tensor = client.get(f"/api/experiments/{experiment_id}/activations/model.layers.0")
    assert tensor.status_code == 404


def test_baseline_persisted_when_steered_fails(
    experiment_store: ExperimentStore, client: TestClient, monkeypatch
):
    import backend.app.api.routes as routes

    original_generate_steered = routes.generate_steered

    def failing_steered(*args, **kwargs):
        raise ValueError("simulated steered failure")

    monkeypatch.setattr(routes, "generate_steered", failing_steered)

    response = _run_steering(client, "model.layers.0.feed_forward", 5, 2.0)
    assert response.status_code == 400
    assert "simulated steered failure" in response.json()["detail"]

    # The baseline run should still be in the store.
    listing = client.get("/api/experiments?condition=baseline").json()
    assert listing["total"] >= 1

    # Restore original to avoid affecting other tests in the same session.
    monkeypatch.setattr(routes, "generate_steered", original_generate_steered)


def test_environment_and_resolved_revision_present(client: TestClient):
    body = _run_baseline(client, prompt="Env check")
    detail = client.get(f"/api/experiments/{body['id']}").json()
    env = detail["environment"]
    assert env["python_version"]
    assert env["torch_version"]
    assert env["transformers_version"]
    assert env["fastapi_version"]
    assert env["app_version"]
    assert env["platform"]
    assert env["git_commit"]
    assert detail["resolved_revision"] == body["revision"]


def test_condition_detail_mapping(client: TestClient):
    _run_baseline(client, prompt="Detail none")
    _run_steering(client, "model.layers.0.feed_forward", 5, 4.0)
    _run_steering(client, "model.layers.0.feed_forward", 5, -4.0)
    _run_steering(client, "model.layers.0.feed_forward", 5, 0.0)
    sweep = client.post(
        "/api/experiments/sweep",
        json={
            "prompt": "Detail control",
            "settings": {"max_new_tokens": 2},
            "module_id": "model.layers.0.feed_forward",
            "feature_index": 5,
            "strengths": [1.0],
            "include_control": True,
            "control_seed": 3,
        },
    )
    assert sweep.status_code == 200

    details = {item["condition_detail"] for item in client.get("/api/experiments?limit=200").json()["items"]}
    assert {"none", "positive", "negative", "zero", "control"}.issubset(details)


def test_sweep_metadata_survives_restart(experiment_store: ExperimentStore, client: TestClient):
    response = client.post(
        "/api/experiments/sweep",
        json={
            "prompt": "Restart sweep",
            "settings": {"max_new_tokens": 2},
            "module_id": "model.layers.0.feed_forward",
            "feature_index": 5,
            "strengths": [1.0, 2.0],
            "include_control": True,
            "control_seed": 11,
        },
    )
    assert response.status_code == 200
    sweep_id = response.json()["sweep_id"]

    reloaded = ExperimentStore(data_dir=experiment_store.data_dir)
    items = reloaded.list_summaries(sweep_id=sweep_id, limit=200).items
    assert len(items) == 4
    assert {item.sweep_index for item in items} == {-1, 0, 1, 2}
    assert all(item.repeat_index == 0 for item in items)
    assert {item.condition for item in items} >= {"baseline", "steered", "control"}


def test_has_activations_remains_true_after_live_eviction(
    client: TestClient, experiment_store: ExperimentStore
):
    body = _run_baseline(client, prompt="Evicted", capture_module_ids=["model.layers.0"])
    experiment_store.activation_store.clear()
    item = next(
        entry
        for entry in client.get("/api/experiments?limit=200").json()["items"]
        if entry["id"] == body["id"]
    )
    assert item["has_activations"] is True
    assert item["activation_source"] is None


def test_snapshot_size_guard(client: TestClient, experiment_store: ExperimentStore):
    body = _run_baseline(
        client,
        prompt="Big snapshot",
        capture_module_ids=["model.layers.0"],
        persist_activations=True,
    )
    experiment_id = body["id"]

    new_store = ExperimentStore(data_dir=experiment_store.data_dir)
    # Artificially lower the budget so the existing snapshot cannot be restored.
    new_store.activation_store.max_bytes = 1
    with pytest.raises(ValueError, match="exceeding the live store budget"):
        new_store.restore_activations(experiment_id)

    client.app.state.experiment_store = new_store
    route_response = client.get(f"/api/experiments/{experiment_id}/activations")
    assert route_response.status_code == 400
    assert "live store budget" in route_response.json()["detail"]
