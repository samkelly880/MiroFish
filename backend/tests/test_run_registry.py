import json

from app.run_registry import RunRegistry


def test_run_registry_create_update_list(tmp_path):
    registry = RunRegistry(root_dir=str(tmp_path / "runs"))
    record = registry.create(requirement="What if X?", platform="parallel", max_rounds=5)
    assert record.run_id.startswith("run_")
    assert (tmp_path / "runs" / record.run_id / "manifest.json").is_file()

    updated = registry.update(
        record.run_id,
        status="completed",
        project_id="proj_1",
        simulation_id="sim_1",
        report_id="report_1",
        artifacts={"report/verdict.json": "/tmp/v.json"},
    )
    assert updated.status == "completed"
    assert updated.project_id == "proj_1"

    loaded = registry.get(record.run_id)
    assert loaded is not None
    assert loaded.simulation_id == "sim_1"

    listed = registry.list(limit=10)
    assert len(listed) == 1
    assert listed[0].run_id == record.run_id

    path = registry.write_artifact(record.run_id, "report/verdict.json", '{"ok":true}')
    assert path.endswith("verdict.json")
    data = json.loads((tmp_path / "runs" / record.run_id / "report" / "verdict.json").read_text())
    assert data == {"ok": True}


def test_run_registry_unknown_update(tmp_path):
    registry = RunRegistry(root_dir=str(tmp_path / "runs"))
    try:
        registry.update("missing", status="failed")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_run_registry_rejects_path_escape(tmp_path):
    registry = RunRegistry(root_dir=str(tmp_path / "runs"))
    record = registry.create(requirement="x")
    try:
        registry.write_artifact(record.run_id, "../outside.txt", "nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert registry.get("../evil") is None
    try:
        registry.artifact_path(record.run_id, "/tmp/abs.txt")
        assert False, "expected ValueError"
    except ValueError:
        pass
