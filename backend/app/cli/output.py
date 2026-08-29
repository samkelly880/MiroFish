"""Human and machine-readable CLI output helpers."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional


def emit(payload: Dict[str, Any], *, as_json: bool, human_text: Optional[str] = None) -> None:
    if as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        text = human_text if human_text is not None else json.dumps(payload, ensure_ascii=False, indent=2)
        sys.stdout.write(text.rstrip() + "\n")


def emit_error(message: str, *, as_json: bool, code: int = 1, **extra: Any) -> int:
    payload = {"ok": False, "error": message, **extra}
    if as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stderr.write(f"error: {message}\n")
    return code
