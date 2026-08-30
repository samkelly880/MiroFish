#!/usr/bin/env python3
"""Search a MiroFish Zep graph using existing ZepToolsService.

Read-only. Does not mutate graphs or start simulations.
Requires ZEP_API_KEY (via backend/.env or environment).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_backend_on_path(root: Path) -> None:
    backend = str(root / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    # Load .env if present without printing secrets
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def search(
    *,
    root: Path,
    graph_id: str,
    query: str,
    limit: int = 10,
    mode: str = "quick",
) -> Dict[str, Any]:
    _ensure_backend_on_path(root)
    from app.services.zep_tools import ZepToolsService

    tools = ZepToolsService()
    if mode == "panorama":
        result = tools.panorama_search(graph_id=graph_id, query=query, include_expired=True)
        return {
            "ok": True,
            "mode": mode,
            "graph_id": graph_id,
            "query": query,
            "total_nodes": result.total_nodes,
            "total_edges": result.total_edges,
            "active_count": result.active_count,
            "historical_count": result.historical_count,
            "active_facts": result.active_facts[:limit],
            "historical_facts": result.historical_facts[: max(1, limit // 2)],
            "entities": [
                {
                    "name": n.name,
                    "labels": n.labels,
                    "summary": (n.summary or "")[:300],
                }
                for n in (result.all_nodes or [])[:20]
            ],
        }

    result = tools.quick_search(graph_id=graph_id, query=query, limit=limit)
    return {
        "ok": True,
        "mode": "quick",
        "graph_id": graph_id,
        "query": query,
        "total_count": result.total_count,
        "facts": result.facts,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["quick", "panorama"],
        default="quick",
        help="quick = semantic search; panorama = full graph overview ranked by query",
    )
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    try:
        payload = search(
            root=root,
            graph_id=args.graph_id,
            query=args.query,
            limit=args.limit,
            mode=args.mode,
        )
    except Exception as exc:  # noqa: BLE001 — surface to skill consumer
        json.dump({"ok": False, "error": str(exc)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
