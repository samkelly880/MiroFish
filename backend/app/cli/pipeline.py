"""
End-to-end CLI pipeline over existing MiroFish services.

Calls ProjectManager, OntologyGenerator, GraphBuilderService, SimulationManager,
SimulationRunner, and ReportAgent — does not reimplement core logic.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..models.project import ProjectManager, ProjectStatus
from ..models.task import TaskManager, TaskStatus
from ..run_registry import RunRegistry, RunRecord
from ..services.graph_builder import GraphBuilderService
from ..services.ontology_generator import OntologyGenerator
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus
from ..services.simulation_manager import SimulationManager
from ..services.simulation_runner import RunnerStatus, SimulationRunner
from ..services.text_processor import TextProcessor
from ..utils.file_parser import FileParser
from .verdict import generate_verdict, verdict_to_json

ProgressCallback = Callable[[str, Dict[str, Any]], None]


class PipelineError(RuntimeError):
    """Raised when the CLI pipeline cannot continue."""


def _progress(cb: Optional[ProgressCallback], stage: str, **payload: Any) -> None:
    if cb:
        cb(stage, payload)


def _ensure_report_succeeded(report: Any) -> None:
    """Raise if ReportAgent returned a non-COMPLETED status (it catches internally)."""
    if report.status != ReportStatus.COMPLETED:
        raise PipelineError(
            report.error or f"Report generation failed (report_id={report.report_id})"
        )


def _platform_flags(platform: str) -> Dict[str, bool]:
    platform = (platform or "parallel").lower()
    if platform == "twitter":
        return {"enable_twitter": True, "enable_reddit": False}
    if platform == "reddit":
        return {"enable_twitter": False, "enable_reddit": True}
    return {"enable_twitter": True, "enable_reddit": True}


def _wait_for_simulation_terminal(
    *,
    simulation_id: str,
    progress: Optional[ProgressCallback] = None,
    poll_seconds: float = 2.0,
    timeout_seconds: float = 7200.0,
    post_close_grace_seconds: float = 120.0,
) -> None:
    """
    Wait until the runner reaches a terminal status.

    After all enabled platforms finish their rounds, OASIS keeps the process
    alive in IPC wait-for-commands mode (runner_status stays RUNNING). For a
    non-interactive CLI run, send close_env (up to two attempts, spaced ≥30s)
    via the same mechanism as the web /api/simulation/close-env endpoint, then
    continue waiting so the monitor thread can stop the Zep updater and publish
    COMPLETED/STOPPED/FAILED. If both close attempts leave the runner RUNNING,
    fail after a short post-close grace period instead of burning the full
    timeout.
    """
    terminal = {
        RunnerStatus.COMPLETED,
        RunnerStatus.FAILED,
        RunnerStatus.STOPPED,
    }
    close_requested = False
    close_attempts = 0
    last_close_at = 0.0
    last_close_error: Optional[str] = None
    deadline = time.time() + timeout_seconds

    while True:
        now = time.time()
        if now >= deadline:
            run_state = SimulationRunner.get_run_state(simulation_id)
            status_value = run_state.runner_status if run_state else None
            raise PipelineError(
                f"Timed out waiting for simulation {simulation_id} after "
                f"{timeout_seconds:.0f}s (last status={status_value}, "
                f"close_requested={close_requested})"
            )

        run_state = SimulationRunner.get_run_state(simulation_id)
        status_value = run_state.runner_status if run_state else None
        platforms_done = (
            SimulationRunner._check_all_platforms_completed(run_state)
            if run_state
            else False
        )
        _progress(
            progress,
            "simulate_status",
            simulation_id=simulation_id,
            status=str(status_value),
            platforms_done=platforms_done,
            close_requested=close_requested,
        )
        if status_value in terminal:
            if status_value == RunnerStatus.FAILED:
                raise PipelineError(
                    f"Simulation failed: {getattr(run_state, 'error', None) or status_value}"
                )
            return

        if (
            close_attempts >= 2
            and status_value == RunnerStatus.RUNNING
            and platforms_done
            and last_close_at > 0
            and (now - last_close_at) >= post_close_grace_seconds
        ):
            detail = last_close_error or "no exception (close may have been ignored)"
            raise PipelineError(
                f"Simulation {simulation_id} still RUNNING after {close_attempts} "
                f"close_env attempts and {post_close_grace_seconds:.0f}s grace "
                f"(last close error: {detail})"
            )

        can_close = (
            run_state is not None
            and status_value == RunnerStatus.RUNNING
            and platforms_done
            and close_attempts < 2
            and (close_attempts == 0 or (now - last_close_at) >= 30.0)
        )
        if can_close:
            _progress(
                progress,
                "close_env",
                simulation_id=simulation_id,
                reason="all_platforms_completed",
                attempt=close_attempts + 1,
            )
            try:
                result = SimulationRunner.close_simulation_env(simulation_id)
                if isinstance(result, dict) and not result.get("success", True):
                    last_close_error = str(
                        result.get("message") or result.get("error") or result
                    )
                    _progress(
                        progress,
                        "close_env_warning",
                        simulation_id=simulation_id,
                        error=last_close_error,
                        attempt=close_attempts + 1,
                    )
                else:
                    last_close_error = None
            except Exception as exc:  # noqa: BLE001 - keep waiting/monitor owns terminal state
                last_close_error = str(exc)
                _progress(
                    progress,
                    "close_env_warning",
                    simulation_id=simulation_id,
                    error=str(exc),
                    attempt=close_attempts + 1,
                )
            close_requested = True
            close_attempts += 1
            last_close_at = time.time()

        time.sleep(poll_seconds)


def _copy_inputs_into_project(project_id: str, files: List[str]) -> List[str]:
    """Copy local files into the project uploads folder; return extracted texts."""
    files_dir = ProjectManager._get_project_files_dir(project_id)
    os.makedirs(files_dir, exist_ok=True)
    document_texts: List[str] = []
    project = ProjectManager.get_project(project_id)
    assert project is not None

    for path in files:
        ext = os.path.splitext(path)[1].lower() or ".txt"
        saved_name = f"{uuid.uuid4().hex[:8]}{ext}"
        dest = os.path.join(files_dir, saved_name)
        shutil.copy2(path, dest)
        project.files.append(
            {
                "filename": os.path.basename(path),
                "size": os.path.getsize(dest),
            }
        )
        text = FileParser.extract_text(dest)
        text = TextProcessor.preprocess_text(text)
        document_texts.append(text)

    ProjectManager.save_project(project)
    return document_texts


def _wait_for_task(task_manager: TaskManager, task_id: str, timeout: float = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = task_manager.get_task(task_id)
        if task is None:
            raise PipelineError(f"Task disappeared: {task_id}")
        if task.status == TaskStatus.COMPLETED:
            return task.result or {}
        if task.status == TaskStatus.FAILED:
            raise PipelineError(task.error or f"Task failed: {task_id}")
        time.sleep(1.5)
    raise PipelineError(f"Timed out waiting for task {task_id}")


def run_pipeline(
    *,
    files: List[str],
    requirement: str,
    platform: str = "parallel",
    max_rounds: Optional[int] = None,
    output_dir: Optional[str] = None,
    skip_report: bool = False,
    skip_verdict: bool = False,
    progress: Optional[ProgressCallback] = None,
    registry: Optional[RunRegistry] = None,
) -> RunRecord:
    if not files:
        raise PipelineError("At least one --files input is required")
    if not requirement.strip():
        raise PipelineError("--requirement is required")
    for path in files:
        if not os.path.isfile(path):
            raise PipelineError(f"Input file not found: {path}")

    errors = Config.validate()
    if errors:
        raise PipelineError("; ".join(errors))

    registry = registry or RunRegistry(root_dir=output_dir)
    record = registry.create(
        requirement=requirement.strip(),
        platform=platform,
        max_rounds=max_rounds or Config.OASIS_DEFAULT_MAX_ROUNDS,
        meta={"files": [os.path.abspath(p) for p in files]},
    )
    registry.update(record.run_id, status="running")
    registry.write_artifact(record.run_id, "input/requirement.txt", requirement.strip())

    try:
        _progress(progress, "project", run_id=record.run_id)
        project = ProjectManager.create_project(name=f"MiroFish CLI {record.run_id}")
        project.simulation_requirement = requirement.strip()
        ProjectManager.save_project(project)

        document_texts = _copy_inputs_into_project(project.project_id, files)
        if not any(t.strip() for t in document_texts):
            raise PipelineError("No text could be extracted from input files")
        all_text = "\n\n".join(
            f"=== {os.path.basename(path)} ===\n{text}"
            for path, text in zip(files, document_texts)
        )
        project = ProjectManager.get_project(project.project_id)
        assert project is not None
        project.total_text_length = len(all_text)
        ProjectManager.save_extracted_text(project.project_id, all_text)
        ProjectManager.save_project(project)
        registry.update(record.run_id, project_id=project.project_id, status="ontology")

        _progress(progress, "ontology", project_id=project.project_id)
        ontology = OntologyGenerator().generate(
            document_texts=document_texts,
            simulation_requirement=requirement.strip(),
        )
        project.ontology = {
            "entity_types": ontology.get("entity_types", []),
            "edge_types": ontology.get("edge_types", []),
        }
        project.analysis_summary = ontology.get("analysis_summary", "")
        project.status = ProjectStatus.ONTOLOGY_GENERATED
        ProjectManager.save_project(project)
        registry.write_artifact(
            record.run_id,
            "input/ontology.json",
            json.dumps(project.ontology, ensure_ascii=False, indent=2),
        )

        _progress(progress, "graph", project_id=project.project_id)
        registry.update(record.run_id, status="graph")
        builder = GraphBuilderService()
        task_manager = TaskManager()
        task_id = builder.build_graph_async(
            text=all_text,
            ontology=project.ontology,
            graph_name=f"MiroFish {record.run_id}",
        )
        graph_result = _wait_for_task(task_manager, task_id)
        graph_id = graph_result.get("graph_id")
        project = ProjectManager.get_project(project.project_id)
        assert project is not None
        if not graph_id:
            graph_id = project.graph_id
        if not graph_id:
            # Some builders only mutate the project object
            raise PipelineError("Graph build completed without a graph_id")
        if not project.graph_id:
            project.graph_id = graph_id
        project.status = ProjectStatus.GRAPH_COMPLETED
        ProjectManager.save_project(project)
        registry.update(record.run_id, graph_id=graph_id, status="prepare")

        _progress(progress, "prepare", graph_id=graph_id)
        flags = _platform_flags(platform)
        manager = SimulationManager()
        sim_state = manager.create_simulation(
            project_id=project.project_id,
            graph_id=graph_id,
            **flags,
        )
        manager.prepare_simulation(
            simulation_id=sim_state.simulation_id,
            simulation_requirement=requirement.strip(),
            document_text=all_text,
        )
        registry.update(
            record.run_id,
            simulation_id=sim_state.simulation_id,
            status="simulate",
        )

        runner_platform = platform if platform in {"twitter", "reddit", "parallel"} else "parallel"
        _progress(progress, "simulate", simulation_id=sim_state.simulation_id)
        SimulationRunner.start_simulation(
            simulation_id=sim_state.simulation_id,
            platform=runner_platform,
            max_rounds=max_rounds or Config.OASIS_DEFAULT_MAX_ROUNDS,
            enable_graph_memory_update=True,
            graph_id=graph_id,
        )

        # Finite CLI runs: OASIS stays in wait-for-commands mode after rounds
        # finish (same as the web backend). The web UI eventually calls
        # /close-env; the CLI must do the equivalent so the monitor can drain
        # Zep writes and publish COMPLETED before report generation.
        _wait_for_simulation_terminal(
            simulation_id=sim_state.simulation_id,
            progress=progress,
        )

        report_id = None
        report_markdown = ""
        if not skip_report:
            registry.update(record.run_id, status="report")
            _progress(progress, "report", simulation_id=sim_state.simulation_id)
            report_id = f"report_{uuid.uuid4().hex[:12]}"
            agent = ReportAgent(
                graph_id=graph_id,
                simulation_id=sim_state.simulation_id,
                simulation_requirement=requirement.strip(),
            )
            report = agent.generate_report(report_id=report_id)
            ReportManager.save_report(report)
            report_id = report.report_id
            registry.update(record.run_id, report_id=report_id)
            # ReportAgent catches exceptions and returns status=FAILED; mirror the
            # web API and fail the CLI run instead of marking it completed.
            _ensure_report_succeeded(report)
            full_path = os.path.join(
                Config.UPLOAD_FOLDER, "reports", report_id, "full_report.md"
            )
            if os.path.isfile(full_path):
                with open(full_path, "r", encoding="utf-8") as handle:
                    report_markdown = handle.read()
                dest = registry.artifact_path(record.run_id, "report", "report.md")
                shutil.copy2(full_path, dest)
                registry.update(
                    record.run_id,
                    artifacts={"report/report.md": dest},
                )

        if not skip_verdict:
            registry.update(record.run_id, status="verdict")
            _progress(progress, "verdict", report_id=report_id)
            verdict = generate_verdict(report_markdown, requirement.strip())
            verdict_path = registry.write_artifact(
                record.run_id, "report/verdict.json", verdict_to_json(verdict)
            )
            registry.update(
                record.run_id,
                artifacts={"report/verdict.json": verdict_path},
                meta={"verdict": verdict},
            )

        return registry.update(record.run_id, status="completed")
    except Exception as exc:
        registry.update(record.run_id, status="failed", error=str(exc))
        raise
