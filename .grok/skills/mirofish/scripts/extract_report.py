#!/usr/bin/env python3
"""Extract structured MiroFish report/verdict data for skill consumers.

Does not run simulations. Reads existing run/report artifacts only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _repo_root() -> Path:
    # scripts/ -> mirofish/ -> skills/ -> .grok/ -> repo
    return Path(__file__).resolve().parents[4]


def _uploads(root: Path) -> Path:
    return root / "backend" / "uploads"


def _safe_id(value: str, *, label: str) -> str:
    if not value or not _SAFE_ID_RE.match(value):
        raise SystemExit(f"Invalid {label}: {value!r}")
    return value


def _safe_child(parent: Path, name: str) -> Path:
    _safe_id(name, label="path id")
    resolved = (parent / name).resolve()
    parent_resolved = parent.resolve()
    if not resolved.is_relative_to(parent_resolved):
        raise SystemExit(f"Path escapes uploads root: {resolved}")
    return resolved


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_ids(
    *,
    root: Path,
    run_id: Optional[str],
    report_id: Optional[str],
) -> Dict[str, Optional[str]]:
    uploads = _uploads(root)
    if run_id:
        run_id = _safe_id(run_id, label="run_id")
        manifest = _load_json(_safe_child(uploads / "runs", run_id) / "manifest.json")
        if not manifest:
            raise SystemExit(f"Unknown run_id: {run_id}")
        return {
            "run_id": run_id,
            "report_id": manifest.get("report_id") or report_id,
            "simulation_id": manifest.get("simulation_id"),
            "graph_id": manifest.get("graph_id"),
            "project_id": manifest.get("project_id"),
            "manifest": manifest,
        }
    if report_id:
        report_id = _safe_id(report_id, label="report_id")
        meta = _load_json(
            _safe_child(uploads / "reports", report_id) / "meta.json"
        )
        if not meta:
            raise SystemExit(f"Unknown report_id: {report_id}")
        return {
            "run_id": None,
            "report_id": report_id,
            "simulation_id": meta.get("simulation_id"),
            "graph_id": meta.get("graph_id"),
            "project_id": None,
            "manifest": None,
            "meta": meta,
        }
    raise SystemExit("Provide --run-id or --report-id")


def extract(root: Path, run_id: Optional[str], report_id: Optional[str]) -> Dict[str, Any]:
    ids = resolve_ids(root=root, run_id=run_id, report_id=report_id)
    uploads = _uploads(root)
    rid = ids["report_id"]
    out: Dict[str, Any] = {
        "ok": True,
        "run_id": ids.get("run_id"),
        "report_id": rid,
        "simulation_id": ids.get("simulation_id"),
        "graph_id": ids.get("graph_id"),
        "project_id": ids.get("project_id"),
    }

    manifest = ids.get("manifest")
    if manifest:
        out["run_status"] = manifest.get("status")
        out["requirement"] = manifest.get("requirement")
        out["platform"] = manifest.get("platform")
        out["max_rounds"] = manifest.get("max_rounds")
        out["error"] = manifest.get("error")
        out["artifacts"] = manifest.get("artifacts")
        verdict = (manifest.get("meta") or {}).get("verdict")
        if verdict:
            out["verdict"] = verdict

    if rid:
        rid = _safe_id(str(rid), label="report_id")
        rdir = _safe_child(uploads / "reports", rid)
        progress = _load_json(rdir / "progress.json")
        outline = _load_json(rdir / "outline.json")
        meta = _load_json(rdir / "meta.json")
        out["report_progress"] = progress
        out["outline"] = outline
        if meta:
            out["report_status"] = meta.get("status")
            out["simulation_requirement"] = meta.get("simulation_requirement")
            out["created_at"] = meta.get("created_at")
            out["completed_at"] = meta.get("completed_at")
            out["report_error"] = meta.get("error")
        # Prefer run verdict; else try run artifact path if present
        if "verdict" not in out and ids.get("run_id"):
            run_dir = _safe_child(uploads / "runs", str(ids["run_id"]))
            verdict = _load_json(run_dir / "report" / "verdict.json")
            if verdict:
                out["verdict"] = verdict
        sections = []
        if outline and isinstance(outline.get("sections"), list):
            for i, sec in enumerate(outline["sections"], 1):
                title = sec.get("title") if isinstance(sec, dict) else str(sec)
                spath = rdir / f"section_{i:02d}.md"
                sections.append(
                    {
                        "index": i,
                        "title": title,
                        "path": str(spath) if spath.exists() else None,
                        "chars": spath.stat().st_size if spath.exists() else 0,
                    }
                )
        out["sections"] = sections
        full = rdir / "full_report.md"
        out["full_report_path"] = str(full) if full.exists() else None
        out["full_report_bytes"] = full.stat().st_size if full.exists() else 0

    return out


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--report-id", default=None)
    parser.add_argument(
        "--repo-root",
        default=None,
        help="MiroFish repo root (default: inferred from script location)",
    )
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    payload = extract(root, args.run_id, args.report_id)
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
