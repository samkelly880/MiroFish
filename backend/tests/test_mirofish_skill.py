"""Tests for the project MiroFish T3/Grok skill package (no pipeline changes)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GROK_SKILL = REPO_ROOT / ".grok" / "skills" / "mirofish"
CLAUDE_SKILL = REPO_ROOT / ".claude" / "skills" / "mirofish"
EXTRACT = GROK_SKILL / "scripts" / "extract_report.py"
GRAPH_SEARCH = GROK_SKILL / "scripts" / "graph_search.py"


def test_skill_files_exist_in_both_scopes():
    for base in (GROK_SKILL, CLAUDE_SKILL):
        assert (base / "SKILL.md").is_file()
        assert (base / "references" / "artifacts.md").is_file()
        assert (base / "scripts" / "extract_report.py").is_file()
        assert (base / "scripts" / "graph_search.py").is_file()


def test_skill_frontmatter_has_name_and_description():
    text = (GROK_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: mirofish" in text
    assert "description:" in text
    # Dual-write must stay in sync
    assert text == (CLAUDE_SKILL / "SKILL.md").read_text(encoding="utf-8")


def test_skill_documents_modes_and_safety():
    text = (GROK_SKILL / "SKILL.md").read_text(encoding="utf-8")
    for needle in (
        "simulate",
        "social",
        "report",
        "inspect",
        "graph",
        "doctor",
        "not a guarantee",
        "30 minutes",
        "mirofish run",
        "insufficient_data",
        "guaranteed real-world",
    ):
        assert needle in text


def test_extract_report_from_completed_run():
    run_id = "run_f9e5db6e78f1"
    manifest = REPO_ROOT / "backend" / "uploads" / "runs" / run_id / "manifest.json"
    if not manifest.exists():
        pytest.skip(f"benchmark run artifact {run_id} not present")

    proc = subprocess.run(
        [sys.executable, str(EXTRACT), "--run-id", run_id, "--repo-root", str(REPO_ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["run_id"] == run_id
    assert payload["report_id"]
    assert payload["verdict"]["prediction"]
    assert "confidence" in payload["verdict"]
    assert "insufficient_data" in payload["verdict"]
    assert isinstance(payload.get("sections"), list)
    assert len(payload["sections"]) >= 1


def test_extract_report_from_report_id():
    report_id = "report_5564a671c092"
    meta = REPO_ROOT / "backend" / "uploads" / "reports" / report_id / "meta.json"
    if not meta.exists():
        pytest.skip(f"report artifact {report_id} not present")

    proc = subprocess.run(
        [
            sys.executable,
            str(EXTRACT),
            "--report-id",
            report_id,
            "--repo-root",
            str(REPO_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["report_id"] == report_id
    assert payload.get("outline") or payload.get("sections")


def test_extract_report_unknown_run_fails():
    proc = subprocess.run(
        [
            sys.executable,
            str(EXTRACT),
            "--run-id",
            "run_does_not_exist_xyz",
            "--repo-root",
            str(REPO_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_graph_search_script_help():
    proc = subprocess.run(
        [sys.executable, str(GRAPH_SEARCH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--graph-id" in proc.stdout
    assert "--query" in proc.stdout


def test_mirofish_cli_still_available():
    cli = REPO_ROOT / "backend" / ".venv" / "bin" / "mirofish"
    assert cli.is_file()
    proc = subprocess.run(
        [str(cli), "doctor", "--json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT / "backend"),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "ok" in payload
    assert "checks" in payload
