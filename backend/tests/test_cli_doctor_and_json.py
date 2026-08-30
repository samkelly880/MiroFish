import json
import os

from app.cli.main import main
from app.cli.verdict import normalize_verdict
from app.run_registry import RunRegistry


def test_doctor_json_stdout(capsys, monkeypatch):
    from app.config import Config

    monkeypatch.setenv("LLM_PROVIDER", "grok-cli")
    monkeypatch.delenv("ZEP_API_KEY", raising=False)
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)
    code = main(["doctor", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "checks" in payload
    assert payload["ok"] is False  # ZEP missing
    assert code == 1
    names = {c["name"] for c in payload["checks"]}
    assert "grok_cli" in names
    assert "zep" in names


def test_doctor_does_not_require_api_key_for_grok(capsys, monkeypatch):
    from app.config import Config

    monkeypatch.setenv("LLM_PROVIDER", "grok-cli")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("ZEP_API_KEY", "test-zep")
    monkeypatch.setattr(Config, "ZEP_API_KEY", "test-zep")
    monkeypatch.setattr(Config, "LLM_API_KEY", None)
    code = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    api_check = next(c for c in payload["checks"] if c["name"] == "openai_compatible")
    assert api_check["ok"] is True
    assert "LLM_API_KEY unset" in api_check["detail"] or "optional" in api_check["detail"]
    # May still fail if grok binary missing in some CI images
    assert "checks" in payload
    assert isinstance(code, int)


def test_runs_list_json(tmp_path, capsys):
    registry = RunRegistry(root_dir=str(tmp_path))
    registry.create(requirement="r1")
    code = main(["runs", "list", "--output-dir", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["count"] == 1
    assert payload["runs"][0]["status"] == "created"


def test_inspect_unknown_run(capsys, tmp_path):
    code = main(["inspect", "run_missing", "--output-dir", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False


def test_run_json_error_includes_pipeline_ids(capsys, monkeypatch, tmp_path):
    from app.cli import main as cli_main
    from app.cli.pipeline import PipelineError

    def boom(**kwargs):
        del kwargs
        raise PipelineError(
            "report boom",
            run_id="run_abc",
            simulation_id="sim_1",
            report_id="report_9",
        )

    monkeypatch.setattr(cli_main, "run_pipeline", boom)
    code = main(
        [
            "run",
            "--files",
            str(tmp_path / "missing.txt"),
            "--requirement",
            "req",
            "--output-dir",
            str(tmp_path),
            "--json",
        ]
    )
    # --files is validated inside run_pipeline; our mock replaces it, so args reach cmd_run.
    # Create a dummy file so argparse/path isn't the issue — cmd_run catches PipelineError.
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False
    assert payload["error"] == "report boom"
    assert payload["run_id"] == "run_abc"
    assert payload["simulation_id"] == "sim_1"
    assert payload["report_id"] == "report_9"


def test_pipeline_error_attaches_run_id_on_wrap():
    from app.cli.pipeline import PipelineError

    err = PipelineError("x", run_id="run_1", project_id="proj_1")
    assert err.run_id == "run_1"
    assert err.project_id == "proj_1"
    assert err.simulation_id is None
    assert str(err) == "x"


def test_normalize_verdict_insufficient():
    verdict = normalize_verdict(
        {
            "prediction": "",
            "confidence": 9,
            "key_dynamics": ["a"],
            "signals": None,
            "insufficient_data": False,
        }
    )
    assert verdict["insufficient_data"] is True
    assert verdict["confidence"] == 0.0
    assert verdict["signals"] == []
    assert "Insufficient" in verdict["prediction"]
