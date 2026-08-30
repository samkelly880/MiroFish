"""
mirofish CLI entrypoint.

Commands:
  doctor              Diagnose environment / provider configuration
  run                 Run a full simulation pipeline
  runs list|status|export
  inspect             Inspect a run (alias of runs status + artifacts)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ..config import Config
from ..run_registry import RunRegistry
from .doctor import doctor_to_dict, run_doctor_checks
from .output import emit, emit_error
from .pipeline import PipelineError, run_pipeline


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON on stdout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mirofish",
        description=(
            "MiroFish CLI — agent-friendly orchestration over the MiroFish engine. "
            "Primary LLM provider: Grok CLI (no xAI API key required)."
        ),
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    sub = parser.add_subparsers(dest="command")

    doctor_p = sub.add_parser("doctor", help="Run environment diagnostics")
    _add_json_flag(doctor_p)

    run_p = sub.add_parser("run", help="Run a full simulation pipeline")
    run_p.add_argument("--files", nargs="+", required=True, help="Source documents")
    run_p.add_argument("--requirement", required=True, help="Simulation requirement")
    run_p.add_argument(
        "--platform",
        choices=["parallel", "twitter", "reddit"],
        default="parallel",
    )
    run_p.add_argument("--max-rounds", type=int, default=None)
    run_p.add_argument(
        "--output-dir",
        default=None,
        help="Run registry root (default: backend/uploads/runs)",
    )
    run_p.add_argument("--skip-report", action="store_true")
    run_p.add_argument("--skip-verdict", action="store_true")
    _add_json_flag(run_p)

    runs_p = sub.add_parser("runs", help="List / inspect / export runs")
    runs_sub = runs_p.add_subparsers(dest="runs_command")

    list_p = runs_sub.add_parser("list", help="List recent runs")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.add_argument("--output-dir", default=None)
    _add_json_flag(list_p)

    status_p = runs_sub.add_parser("status", help="Show one run manifest")
    status_p.add_argument("run_id")
    status_p.add_argument("--output-dir", default=None)
    _add_json_flag(status_p)

    export_p = runs_sub.add_parser("export", help="Export run manifest + artifact paths")
    export_p.add_argument("run_id")
    export_p.add_argument("--output-dir", default=None)
    _add_json_flag(export_p)

    inspect_p = sub.add_parser("inspect", help="Inspect a run (manifest + artifacts)")
    inspect_p.add_argument("run_id")
    inspect_p.add_argument("--output-dir", default=None)
    _add_json_flag(inspect_p)

    return parser


def _registry(output_dir: Optional[str]) -> RunRegistry:
    return RunRegistry(root_dir=output_dir) if output_dir else RunRegistry()


def _run_summary(record) -> Dict[str, Any]:
    return {
        "run_id": record.run_id,
        "status": record.status,
        "created_at": record.created_at,
        "artifact_count": len(record.artifacts or {}),
        "project_id": record.project_id,
        "simulation_id": record.simulation_id,
        "report_id": record.report_id,
        "error": record.error,
    }


def cmd_doctor(as_json: bool) -> int:
    # doctor always runs, even when Config.validate() would fail
    checks = run_doctor_checks()
    payload = doctor_to_dict(checks)
    if as_json:
        emit(payload, as_json=True)
    else:
        for check in payload["checks"]:
            mark = "OK" if check["ok"] else "FAIL"
            line = f"[{mark}] {check['name']}: {check['detail']}"
            if check.get("hint"):
                line += f"\n       hint: {check['hint']}"
            print(line)
        print("overall:", "ok" if payload["ok"] else "failed")
    # Soft env_file warning does not fail doctor; hard failures do
    return 0 if payload["ok"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    def on_progress(stage: str, payload: Dict[str, Any]) -> None:
        if not args.json:
            extra = " ".join(f"{k}={v}" for k, v in payload.items())
            print(f"[mirofish] {stage} {extra}".rstrip(), file=sys.stderr)

    try:
        record = run_pipeline(
            files=args.files,
            requirement=args.requirement,
            platform=args.platform,
            max_rounds=args.max_rounds,
            output_dir=args.output_dir,
            skip_report=args.skip_report,
            skip_verdict=args.skip_verdict,
            progress=on_progress,
        )
    except PipelineError as exc:
        extras = {
            key: value
            for key, value in {
                "run_id": exc.run_id,
                "project_id": exc.project_id,
                "simulation_id": exc.simulation_id,
                "report_id": exc.report_id,
            }.items()
            if value
        }
        return emit_error(str(exc), as_json=args.json, **extras)
    except Exception as exc:  # noqa: BLE001
        return emit_error(str(exc), as_json=args.json)

    payload = {
        "ok": True,
        "run": record.to_dict(),
        "verdict": (record.meta or {}).get("verdict"),
    }
    emit(
        payload,
        as_json=args.json,
        human_text=(
            f"run_id={record.run_id}\n"
            f"status={record.status}\n"
            f"project_id={record.project_id}\n"
            f"simulation_id={record.simulation_id}\n"
            f"report_id={record.report_id}\n"
        ),
    )
    return 0 if record.status == "completed" else 1


def cmd_runs_list(args: argparse.Namespace) -> int:
    registry = _registry(args.output_dir)
    runs = [_run_summary(r) for r in registry.list(limit=args.limit)]
    payload = {"ok": True, "count": len(runs), "runs": runs}
    emit(payload, as_json=args.json)
    return 0


def cmd_runs_status(args: argparse.Namespace) -> int:
    registry = _registry(args.output_dir)
    record = registry.get(args.run_id)
    if record is None:
        return emit_error(f"Unknown run_id: {args.run_id}", as_json=args.json)
    emit({"ok": True, "run": record.to_dict()}, as_json=args.json)
    return 0


def cmd_runs_export(args: argparse.Namespace) -> int:
    registry = _registry(args.output_dir)
    record = registry.get(args.run_id)
    if record is None:
        return emit_error(f"Unknown run_id: {args.run_id}", as_json=args.json)
    payload = {
        "ok": True,
        "run_id": record.run_id,
        "manifest": record.to_dict(),
        "run_dir": registry._run_dir(record.run_id),
        "artifacts": record.artifacts,
    }
    emit(payload, as_json=args.json)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Always allow help/version without validating config
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if argv[0] in {"--version", "-V"}:
        print("mirofish 0.1.0")
        return 0

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "doctor":
        return cmd_doctor(getattr(args, "json", False))

    if args.command == "run":
        return cmd_run(args)

    if args.command == "runs":
        if args.runs_command == "list":
            return cmd_runs_list(args)
        if args.runs_command == "status":
            return cmd_runs_status(args)
        if args.runs_command == "export":
            return cmd_runs_export(args)
        parser.parse_args(["runs", "--help"])
        return 2

    if args.command == "inspect":
        # Alias of runs status with artifact listing
        return cmd_runs_export(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
