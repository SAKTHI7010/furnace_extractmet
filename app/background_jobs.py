"""Isolated numerical jobs for the Streamlit operator console.

The physics simulation is CPU-heavy and pure Python/NumPy. Running it in a
thread can still delay Streamlit on small plant PCs because both tasks share the
same Python process. This module launches each heat calculation in a short-lived
child process and exposes a Future-like object for non-blocking polling.
"""
from __future__ import annotations

import atexit
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
_WORKER = ROOT / "app" / "sim_worker.py"
_ACTIVE: set["SimulationJob"] = set()
_LOCK = threading.Lock()


class SimulationJob:
    """Small Future-compatible wrapper around an isolated worker process."""

    def __init__(self, payload: Dict[str, Any]):
        self._dir = Path(tempfile.mkdtemp(prefix="smartmelt_job_"))
        self._input = self._dir / "input.pkl"
        self._output = self._dir / "output.pkl"
        self._error = self._dir / "error.txt"
        self._result_cache = None
        self._cleaned = False
        with self._input.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        env = os.environ.copy()
        extra = os.pathsep.join((str(ROOT), str(ROOT / "app" / "lib")))
        env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # Avoid a small heat spawning a large BLAS thread team on shared PCs.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        kwargs: Dict[str, Any] = {
            "cwd": str(ROOT),
            "env": env,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            [sys.executable, str(_WORKER), str(self._input), str(self._output), str(self._error)],
            **kwargs,
        )
        with _LOCK:
            _ACTIVE.add(self)

    def done(self) -> bool:
        return self._proc.poll() is not None

    def result(self):
        if self._result_cache is not None:
            return self._result_cache
        rc = self._proc.poll()
        if rc is None:
            raise RuntimeError("simulation job is still running")
        try:
            if rc != 0 or not self._output.exists():
                detail = self._error.read_text(encoding="utf-8", errors="replace") if self._error.exists() else ""
                raise RuntimeError(detail.strip() or f"simulation worker exited with code {rc}")
            with self._output.open("rb") as fh:
                self._result_cache = pickle.load(fh)
            return self._result_cache
        finally:
            self._cleanup()

    def cancel(self) -> bool:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=1.5)
            cancelled = True
        else:
            cancelled = False
        self._cleanup()
        return cancelled

    def _cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        with _LOCK:
            _ACTIVE.discard(self)
        shutil.rmtree(self._dir, ignore_errors=True)

    def __hash__(self):
        return id(self)


def start_simulation_job(**payload) -> SimulationJob:
    return SimulationJob(payload)


def _shutdown_jobs():
    with _LOCK:
        jobs = list(_ACTIVE)
    for job in jobs:
        try:
            job.cancel()
        except Exception:
            pass


atexit.register(_shutdown_jobs)
