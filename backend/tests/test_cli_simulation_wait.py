"""CLI wait loop must close OASIS wait-mode after platforms finish."""

from types import SimpleNamespace

import pytest

from app.cli.pipeline import PipelineError, _wait_for_simulation_terminal
from app.services.simulation_runner import RunnerStatus


class FakeRunner:
    def __init__(self, states, *, platforms_completed_at=None, close_errors=None):
        self.states = list(states)
        self.platforms_completed_at = platforms_completed_at
        self.close_errors = list(close_errors or [])
        self.close_calls = 0
        self.polls = 0

    def get_run_state(self, simulation_id):
        del simulation_id
        idx = min(self.polls, len(self.states) - 1)
        state = self.states[idx]
        self.polls += 1
        return state

    def _check_all_platforms_completed(self, state):
        del state
        if self.platforms_completed_at is None:
            return False
        # polls already incremented in get_run_state
        return self.polls > self.platforms_completed_at

    def close_simulation_env(self, simulation_id):
        del simulation_id
        self.close_calls += 1
        if self.close_errors:
            err = self.close_errors.pop(0)
            if err is not None:
                raise err
        return {"success": True}


def test_wait_sends_close_env_once_after_platforms_complete(monkeypatch):
    running = SimpleNamespace(runner_status=RunnerStatus.RUNNING, error=None)
    completed = SimpleNamespace(runner_status=RunnerStatus.COMPLETED, error=None)
    fake = FakeRunner(
        [running, running, completed],
        platforms_completed_at=1,  # after first poll, platforms look done
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.get_run_state",
        fake.get_run_state,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner._check_all_platforms_completed",
        fake._check_all_platforms_completed,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.close_simulation_env",
        fake.close_simulation_env,
    )

    _wait_for_simulation_terminal(simulation_id="sim_x", poll_seconds=0)

    assert fake.close_calls == 1
    assert fake.polls >= 3


def test_wait_does_not_close_before_platforms_complete(monkeypatch):
    running = SimpleNamespace(runner_status=RunnerStatus.RUNNING, error=None)
    completed = SimpleNamespace(runner_status=RunnerStatus.COMPLETED, error=None)
    fake = FakeRunner([running, completed], platforms_completed_at=None)
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.get_run_state",
        fake.get_run_state,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner._check_all_platforms_completed",
        fake._check_all_platforms_completed,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.close_simulation_env",
        fake.close_simulation_env,
    )

    _wait_for_simulation_terminal(simulation_id="sim_x", poll_seconds=0)

    assert fake.close_calls == 0


def test_wait_raises_on_failed_status(monkeypatch):
    failed = SimpleNamespace(runner_status=RunnerStatus.FAILED, error="boom")
    fake = FakeRunner([failed], platforms_completed_at=None)
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.get_run_state",
        fake.get_run_state,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner._check_all_platforms_completed",
        fake._check_all_platforms_completed,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.close_simulation_env",
        fake.close_simulation_env,
    )

    try:
        _wait_for_simulation_terminal(simulation_id="sim_x", poll_seconds=0)
        assert False, "expected PipelineError"
    except PipelineError as exc:
        assert "boom" in str(exc)
    assert fake.close_calls == 0


def test_wait_times_out_when_never_terminal(monkeypatch):
    running = SimpleNamespace(runner_status=RunnerStatus.RUNNING, error=None)
    fake = FakeRunner([running], platforms_completed_at=None)
    clock = {"t": 1000.0}

    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.get_run_state",
        fake.get_run_state,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner._check_all_platforms_completed",
        fake._check_all_platforms_completed,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.close_simulation_env",
        fake.close_simulation_env,
    )
    monkeypatch.setattr("app.cli.pipeline.time.time", lambda: clock["t"])

    def fake_sleep(seconds):
        clock["t"] += max(seconds, 0.1)

    monkeypatch.setattr("app.cli.pipeline.time.sleep", fake_sleep)

    with pytest.raises(PipelineError, match="Timed out"):
        _wait_for_simulation_terminal(
            simulation_id="sim_x",
            poll_seconds=1,
            timeout_seconds=2,
        )
    assert fake.close_calls == 0


def test_wait_retries_close_env_after_spacing(monkeypatch):
    running = SimpleNamespace(runner_status=RunnerStatus.RUNNING, error=None)
    completed = SimpleNamespace(runner_status=RunnerStatus.COMPLETED, error=None)
    # Stay running long enough for two closes, then complete
    fake = FakeRunner(
        [running] * 40 + [completed],
        platforms_completed_at=1,
        close_errors=[RuntimeError("ipc flaky"), None],
    )
    clock = {"t": 0.0}

    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.get_run_state",
        fake.get_run_state,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner._check_all_platforms_completed",
        fake._check_all_platforms_completed,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.close_simulation_env",
        fake.close_simulation_env,
    )
    monkeypatch.setattr("app.cli.pipeline.time.time", lambda: clock["t"])

    def fake_sleep(seconds):
        clock["t"] += max(seconds, 1.0)

    monkeypatch.setattr("app.cli.pipeline.time.sleep", fake_sleep)

    _wait_for_simulation_terminal(
        simulation_id="sim_x",
        poll_seconds=1,
        timeout_seconds=600,
    )
    assert fake.close_calls == 2


def test_wait_fails_fast_after_exhausted_close_attempts(monkeypatch):
    running = SimpleNamespace(runner_status=RunnerStatus.RUNNING, error=None)
    fake = FakeRunner(
        [running],
        platforms_completed_at=0,
        close_errors=[RuntimeError("close failed"), RuntimeError("still failed")],
    )
    clock = {"t": 0.0}

    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.get_run_state",
        fake.get_run_state,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner._check_all_platforms_completed",
        fake._check_all_platforms_completed,
    )
    monkeypatch.setattr(
        "app.cli.pipeline.SimulationRunner.close_simulation_env",
        fake.close_simulation_env,
    )
    monkeypatch.setattr("app.cli.pipeline.time.time", lambda: clock["t"])

    def fake_sleep(seconds):
        clock["t"] += max(seconds, 1.0)

    monkeypatch.setattr("app.cli.pipeline.time.sleep", fake_sleep)

    with pytest.raises(PipelineError, match="still RUNNING after 2 close_env"):
        _wait_for_simulation_terminal(
            simulation_id="sim_x",
            poll_seconds=1,
            timeout_seconds=7200,
        )
    assert fake.close_calls == 2
