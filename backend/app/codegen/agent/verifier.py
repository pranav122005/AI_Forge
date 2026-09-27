"""
AIForge Agent Project Verification Stage Engine
================================================

Performs rigorous verification on generated/modified projects before packaging:
1. Syntax correctness verification (AST compilation check on Python files).
2. Project entrypoint existence check.
3. Forbidden files enforcement (.env, secrets.json, id_rsa).
4. Hardcoded secret / credential leak detection.
5. Test suite execution check.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from app.codegen.agent.models import VerificationResult
from app.codegen.agent.context import IGNORED_DIRS, IGNORED_FILENAMES
from app.codegen.models import TestResult

SUSPICIOUS_SECRET_PATTERNS = [
    re.compile(r"GEMINI_API_KEY\s*=\s*['\"]AIza[0-9A-Za-z-_]{20,}['\"]"),
    re.compile(r"OPENAI_API_KEY\s*=\s*['\"]sk-[0-9A-Za-z]{20,}['\"]"),
    re.compile(r"AWS_SECRET_ACCESS_KEY\s*=\s*['\"][0-9A-Za-z/+]{30,}['\"]"),
    re.compile(r"-----BEGIN PRIVATE KEY-----"),
]


def verify_project(
    target_root: Path,
    test_result: TestResult | None = None,
    modified_files: List[str] | None = None,
) -> VerificationResult:
    """
    Perform complete multi-check verification on the target project workspace.
    """
    target_resolved = target_root.resolve()
    passed_checks: List[str] = []
    failed_checks: List[str] = []
    details: Dict[str, Any] = {}

    # Check 1: Workspace directory exists
    if target_resolved.exists() and target_resolved.is_dir():
        passed_checks.append("WORKSPACE_EXISTS")
    else:
        failed_checks.append("WORKSPACE_EXISTS")
        details["WORKSPACE_EXISTS"] = f"Directory '{target_root}' not found."
        return VerificationResult(
            success=False,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            details=details,
        )

    # Check 2: Entrypoint check
    entrypoints = ["app/main.py", "main.py", "run.py", "server.py", "Dockerfile"]
    found_entrypoint = any((target_resolved / ep).exists() for ep in entrypoints)
    if found_entrypoint:
        passed_checks.append("ENTRYPOINT_EXISTS")
    else:
        failed_checks.append("ENTRYPOINT_EXISTS")
        details["ENTRYPOINT_EXISTS"] = "No valid entrypoint found (e.g. app/main.py or main.py)."

    # Check 3: Python syntax compilation check (AST parse)
    syntax_errors: List[str] = []
    for path in target_resolved.rglob("*.py"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        rel_path = path.relative_to(target_resolved).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            ast.parse(content, filename=rel_path)
        except SyntaxError as syn_err:
            syntax_errors.append(f"{rel_path}: L{syn_err.lineno} {syn_err.msg}")

    if not syntax_errors:
        passed_checks.append("PYTHON_SYNTAX_VALID")
    else:
        failed_checks.append("PYTHON_SYNTAX_VALID")
        details["PYTHON_SYNTAX_VALID"] = syntax_errors

    # Check 4: Forbidden files check
    forbidden_found: List[str] = []
    for path in target_resolved.rglob("*"):
        if path.is_file() and path.name in IGNORED_FILENAMES:
            rel_path = path.relative_to(target_resolved).as_posix()
            forbidden_found.append(rel_path)

    if not forbidden_found:
        passed_checks.append("NO_FORBIDDEN_FILES")
    else:
        failed_checks.append("NO_FORBIDDEN_FILES")
        details["NO_FORBIDDEN_FILES"] = forbidden_found

    # Check 5: Secret leak detection
    secret_leaks: List[str] = []
    for path in target_resolved.rglob("*.py"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        rel_path = path.relative_to(target_resolved).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SUSPICIOUS_SECRET_PATTERNS:
                if pattern.search(content):
                    secret_leaks.append(f"{rel_path}: Matched suspicious secret pattern")
        except Exception:
            pass

    if not secret_leaks:
        passed_checks.append("NO_SECRET_LEAKS")
    else:
        failed_checks.append("NO_SECRET_LEAKS")
        details["NO_SECRET_LEAKS"] = secret_leaks

    # Check 6: Test suite pass check
    if test_result is not None:
        if test_result.success:
            passed_checks.append("TEST_SUITE_PASSED")
        else:
            failed_checks.append("TEST_SUITE_PASSED")
            details["TEST_SUITE_PASSED"] = test_result.stderr or "Tests failed"

    success = len(failed_checks) == 0

    return VerificationResult(
        success=success,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        details=details,
    )
