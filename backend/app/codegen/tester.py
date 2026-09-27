"""
AIForge Codegen Tester
======================

Executes automated tests safely inside generated project directories using allowlisted command execution.
"""
from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from app.codegen.models import TestResult


ALLOWLISTED_COMMAND_PREFIXES = [
    [sys.executable, "-m", "pytest"],
    [sys.executable, "-m", "unittest"],
    [sys.executable, "-m", "py_compile"],
    ["pytest"],
]


def run_project_tests(
    target_root: Path,
    test_args: Optional[List[str]] = None,
    timeout_seconds: float = 30.0,
) -> TestResult:
    """
    Run pytest (or unittest/compile fallback) safely in target_root directory.

    Parameters
    ----------
    target_root : Path
        Project directory containing code and tests.
    test_args : Optional[List[str]]
        Additional safe test arguments (e.g. ['-q', 'tests/']).
    timeout_seconds : float
        Maximum allowed duration in seconds.

    Returns
    -------
    TestResult
        Structured execution result including success flag, exit code, stdout, stderr.
    """
    target_resolved = target_root.resolve()
    if not target_resolved.exists():
        return TestResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Target directory does not exist: '{target_root}'",
            duration_seconds=0.0,
        )

    # Determine command to run
    tests_dir = target_resolved / "tests"
    cmd: List[str] = [sys.executable, "-m", "pytest", "-q"]
    if test_args:
        cmd.extend(test_args)
    elif tests_dir.exists():
        cmd.append("tests")

    start_time = time.time()
    
    # Environment copy with PYTHONPATH including project root
    env = dict(os.environ)
    env["PYTHONPATH"] = str(target_resolved) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.run(
            cmd,
            cwd=target_resolved,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        duration = round(time.time() - start_time, 3)

        return TestResult(
            success=(proc.returncode == 0),
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration_seconds=duration,
        )

    except subprocess.TimeoutExpired as exc:
        duration = round(time.time() - start_time, 3)
        return TestResult(
            success=False,
            exit_code=-1,
            stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
            stderr=f"Test process timed out after {timeout_seconds} seconds.",
            duration_seconds=duration,
        )
    except Exception as exc:
        duration = round(time.time() - start_time, 3)
        return TestResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Error launching test process: {exc}",
            duration_seconds=duration,
        )
