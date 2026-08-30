"""CLI pipeline must fail the run when ReportAgent returns FAILED."""

from types import SimpleNamespace

import pytest

from app.cli.pipeline import PipelineError, _ensure_report_succeeded
from app.services.report_agent import ReportStatus


def test_ensure_report_succeeded_passes_on_completed():
    report = SimpleNamespace(
        report_id="report_ok",
        status=ReportStatus.COMPLETED,
        error=None,
    )
    _ensure_report_succeeded(report)


def test_ensure_report_succeeded_raises_on_failed_with_error():
    report = SimpleNamespace(
        report_id="report_bad",
        status=ReportStatus.FAILED,
        error="section max turns reached",
    )
    with pytest.raises(PipelineError, match="section max turns reached"):
        _ensure_report_succeeded(report)


def test_ensure_report_succeeded_raises_fallback_message():
    report = SimpleNamespace(
        report_id="report_empty_err",
        status=ReportStatus.FAILED,
        error=None,
    )
    with pytest.raises(PipelineError, match="report_empty_err"):
        _ensure_report_succeeded(report)


def test_ensure_report_succeeded_rejects_non_completed_statuses():
    for status in (ReportStatus.PENDING, ReportStatus.PLANNING, ReportStatus.GENERATING):
        report = SimpleNamespace(
            report_id="report_partial",
            status=status,
            error=None,
        )
        with pytest.raises(PipelineError, match="report_partial"):
            _ensure_report_succeeded(report)
