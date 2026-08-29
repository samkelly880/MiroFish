"""Environment / configuration diagnostics for MiroFish."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..config import Config
from ..providers.factory import resolve_provider_name
from ..providers.grok_cli import grok_binary_available


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    hint: Optional[str] = None


def _check_provider() -> DoctorCheck:
    try:
        provider = resolve_provider_name(Config.LLM_PROVIDER)
    except ValueError as exc:
        return DoctorCheck("llm_provider", False, str(exc), "Set LLM_PROVIDER=grok-cli or openai-compatible")
    return DoctorCheck("llm_provider", True, f"LLM_PROVIDER={provider}")


def _check_grok_cli() -> DoctorCheck:
    provider = resolve_provider_name(Config.LLM_PROVIDER)
    if provider != "grok-cli":
        return DoctorCheck("grok_cli", True, "skipped (provider is not grok-cli)")
    binary = Config.GROK_CLI_BIN or "grok"
    if grok_binary_available(binary):
        path = binary if os.path.isabs(binary) or os.sep in binary else shutil.which(binary)
        return DoctorCheck(
            "grok_cli",
            True,
            f"found {path}",
            "Normal Grok workflow uses `grok login` — no xAI API key required",
        )
    return DoctorCheck(
        "grok_cli",
        False,
        f"binary not found: {binary}",
        "Install from https://x.ai/cli/install.sh then run `grok login`",
    )


def _check_openai_api() -> DoctorCheck:
    provider = resolve_provider_name(Config.LLM_PROVIDER)
    if provider != "openai-compatible":
        has_key = bool(Config.LLM_API_KEY)
        return DoctorCheck(
            "openai_compatible",
            True,
            "optional secondary provider; "
            + ("LLM_API_KEY is set" if has_key else "LLM_API_KEY unset (OK for grok-cli)"),
        )
    if Config.LLM_API_KEY:
        return DoctorCheck(
            "openai_compatible",
            True,
            f"LLM_API_KEY set; base_url={Config.LLM_BASE_URL}; model={Config.LLM_MODEL_NAME}",
        )
    return DoctorCheck(
        "openai_compatible",
        False,
        "LLM_API_KEY missing",
        "Required only when LLM_PROVIDER=openai-compatible",
    )


def _check_zep() -> DoctorCheck:
    if Config.ZEP_API_KEY:
        return DoctorCheck("zep", True, "ZEP_API_KEY is set")
    return DoctorCheck(
        "zep",
        False,
        "ZEP_API_KEY is not configured",
        "Create a key at https://app.getzep.com/",
    )


def _check_zep_url() -> DoctorCheck:
    if os.environ.get("ZEP_API_URL"):
        return DoctorCheck(
            "zep_url",
            False,
            "ZEP_API_URL is set but unsupported",
            "MiroFish only connects to Zep Cloud — unset ZEP_API_URL",
        )
    return DoctorCheck("zep_url", True, "ZEP_API_URL unset (OK)")


def _check_env_file() -> DoctorCheck:
    root_env = os.path.join(os.path.dirname(__file__), "../../../.env")
    root_env = os.path.abspath(root_env)
    if os.path.isfile(root_env):
        return DoctorCheck("env_file", True, f"found {root_env}")
    return DoctorCheck(
        "env_file",
        True,
        ".env not found (soft warning)",
        "Copy .env.example to .env for Zep and optional API settings",
    )


def run_doctor_checks() -> List[DoctorCheck]:
    checks: List[Callable[[], DoctorCheck]] = [
        _check_provider,
        _check_grok_cli,
        _check_openai_api,
        _check_zep,
        _check_zep_url,
        _check_env_file,
    ]
    results: List[DoctorCheck] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001
            results.append(DoctorCheck(check.__name__, False, str(exc)))
    return results


def doctor_to_dict(checks: List[DoctorCheck]) -> dict:
    return {
        "ok": all(c.ok for c in checks if c.name != "env_file"),
        "checks": [
            {
                "name": c.name,
                "ok": c.ok,
                "detail": c.detail,
                **({"hint": c.hint} if c.hint else {}),
            }
            for c in checks
        ],
    }
