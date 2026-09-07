"""In-memory background jobs for long-running backtests.

The browser polls ``GET /api/backtest/jobs/{id}`` while the worker thread runs
the backtest. Jobs are kept in memory only (last 20) — refresh-safe enough for
a local analysis tool, no broker/queue required.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="backtest")
_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
MAX_JOBS = 20


def _prune() -> None:
    if len(_jobs) <= MAX_JOBS:
        return
    ordered = sorted(_jobs.items(), key=lambda kv: kv[1].get("created_at", 0))
    for key, _ in ordered[: len(_jobs) - MAX_JOBS]:
        del _jobs[key]


def submit(kind: str, label: str, fn: Callable[[], Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job: Dict[str, Any] = {
        "id": job_id, "kind": kind, "label": label,
        "status": "queued", "progress": {"done": 0, "total": 0, "pct": 0},
        "created_at": time.time(), "started_at": None, "finished_at": None,
        "result": None, "error": None,
    }
    with _lock:
        _jobs[job_id] = job
        _prune()

    def _run() -> None:
        with _lock:
            job["status"] = "running"
            job["started_at"] = time.time()

        def _progress(done: int, total: int) -> None:
            with _lock:
                job["progress"] = {"done": done, "total": total,
                                   "pct": round(done / total * 100, 1) if total else 0}

        try:
            # Backtest functions accept an optional progress callback; call with it
            # when supported, otherwise plain.
            try:
                out = fn(_progress)  # type: ignore[call-arg]
            except TypeError:
                out = fn()
            with _lock:
                job["status"] = "done"
                job["result"] = out
                job["progress"] = {"done": 1, "total": 1, "pct": 100}
        except Exception as e:  # noqa: BLE001 — surfaced to the browser, never raised
            with _lock:
                job["status"] = "error"
                job["error"] = str(e)
                job["traceback"] = traceback.format_exc(limit=8)
        finally:
            with _lock:
                job["finished_at"] = time.time()

    _executor.submit(_run)
    return job_id


def get(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _jobs.get(job_id)


def recent() -> List[Dict[str, Any]]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)
        # List view omits the (large) result payload.
        return [{k: v for k, v in j.items() if k != "result"} for j in jobs]
