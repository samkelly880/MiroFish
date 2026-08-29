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
from ..services.report_agent import ReportAgent, ReportManager
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


def _platform_flags(platform: str) -> Dict[str, bool]:
    platform = (platform or "parallel").lower()
    if platform == "twitter":
        return {"enable_twitter": True, "enable_reddit": False}
    if platform == "reddit":
        return {"enable_twitter": False, "enable_reddit": True}
    return {"enable_twitter": True, "enable_reddit": True}


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

        terminal = {
            RunnerStatus.COMPLETED,
            RunnerStatus.FAILED,
            RunnerStatus.STOPPED,
        }
        while True:
            run_state = SimulationRunner.get_run_state(sim_state.simulation_id)
            status_value = run_state.runner_status if run_state else None
            _progress(
                progress,
                "simulate_status",
                simulation_id=sim_state.simulation_id,
                status=str(status_value),
            )
            if status_value in terminal:
                if status_value == RunnerStatus.FAILED:
                    raise PipelineError(
                        f"Simulation failed: {getattr(run_state, 'error', None) or status_value}"
                    )
                break
            time.sleep(2)

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
                    report_id=report_id,
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
