"""
AIForge Agent Project Understanding & File Selection Layer
============================================================

Provides safe project inspection, metadata summary generation, and targeted file selection.
Excludes forbidden files, secrets, binaries, caches, and vendor directories.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.codegen.errors import PathValidationError
from app.codegen.validator import validate_file_path
from app.codegen.agent.models import FileSelection
from app.llm.providers.base import BaseLLMProvider, LLMProviderError

IGNORED_DIRS: Set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "build",
    "dist",
    "snapshots",
}

IGNORED_EXTENSIONS: Set[str] = {
    ".pyc",
    ".zip",
    ".tar",
    ".gz",
    ".joblib",
    ".pkl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".csv",
    ".parquet",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
}

IGNORED_FILENAMES: Set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "secrets.json",
    "id_rsa",
    "id_rsa.pub",
    "credentials.json",
}


def is_safe_inspect_file(path: Path, target_root: Path) -> bool:
    """Check if a file path is safe and non-forbidden for inspection."""
    target_resolved = target_root.resolve()
    try:
        resolved = path.resolve()
        resolved.relative_to(target_resolved)
    except Exception:
        return False

    if any(part in IGNORED_DIRS for part in path.parts):
        return False
    if path.name in IGNORED_FILENAMES or path.suffix.lower() in IGNORED_EXTENSIONS:
        return False
    return True


def build_project_summary(target_root: Path) -> Dict[str, Any]:
    """
    Inspect project root and produce a compact, structured project representation.
    """
    target_resolved = target_root.resolve()
    if not target_resolved.exists():
        raise FileNotFoundError(f"Project directory '{target_root}' does not exist.")

    all_files: List[str] = []
    entrypoints: List[str] = []
    tests: List[str] = []
    api_routes: List[str] = []
    dependencies: List[str] = []
    important_files: List[str] = []

    for path in sorted(target_resolved.rglob("*")):
        if not path.is_file():
            continue
        if not is_safe_inspect_file(path, target_resolved):
            continue

        rel_path = path.relative_to(target_resolved).as_posix()
        all_files.append(rel_path)

        # Classify entrypoints
        if rel_path in ("app/main.py", "main.py", "app/api.py", "run.py", "server.py"):
            entrypoints.append(rel_path)

        # Classify test files
        if rel_path.startswith("tests/") or path.name.startswith("test_") or path.name.endswith("_test.py"):
            tests.append(rel_path)

        # Classify requirements / dependencies
        if rel_path in ("requirements.txt", "pyproject.toml", "setup.py", "Pipfile"):
            dependencies.append(rel_path)
            important_files.append(rel_path)

        # Inspect Python files for API route signatures
        if rel_path.endswith(".py") and ("app/" in rel_path or rel_path.startswith("main")):
            important_files.append(rel_path)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                routes = re.findall(r"@(?:app|router)\.(get|post|put|delete|patch)\(\s*['\"]([^'\"]+)['\"]", content)
                for method, route in routes:
                    api_routes.append(f"{method.upper()} {route}")
            except Exception:
                pass

    project_type = "python_fastapi" if any("fastapi" in f or "main.py" in f for f in all_files) else "python_general"

    return {
        "project_type": project_type,
        "total_files": len(all_files),
        "entrypoints": entrypoints,
        "dependencies": dependencies,
        "api_routes": api_routes,
        "tests": tests,
        "important_files": list(set(important_files + entrypoints + dependencies + tests))[:20],
        "file_structure": all_files,
    }


async def select_relevant_files(
    target_root: Path,
    instruction: str,
    summary: Dict[str, Any],
    provider: Optional[BaseLLMProvider] = None,
) -> List[str]:
    """
    Select candidate relevant files based on user instruction and project summary.
    Prevents sending unnecessary files to the LLM context.
    """
    all_files = summary.get("file_structure", [])

    if provider is not None:
        prompt = (
            f"User Instruction:\n{instruction}\n\n"
            f"Project Structure & Files:\n{all_files}\n\n"
            f"Project Summary:\n{summary}\n\n"
            "Select the list of relevant file paths from the project that should be inspected or modified."
        )
        sys_prompt = "You are a software architect selecting relevant project files for modification. Return output adhering strictly to FileSelection schema."
        try:
            selection = await provider.generate_structured(
                prompt=prompt,
                response_model=FileSelection,
                system_prompt=sys_prompt,
            )
            if selection and selection.relevant_files:
                valid_selected = [f for f in selection.relevant_files if f in all_files]
                if valid_selected:
                    return valid_selected
        except Exception:
            pass

    # Rule-based fallback file selection
    instr_lower = instruction.lower()
    selected: Set[str] = set()

    # Always include entrypoints & requirements for context
    for f in summary.get("entrypoints", []):
        selected.add(f)
    for f in summary.get("dependencies", []):
        selected.add(f)

    # Keyword matching
    if "jwt" in instr_lower or "auth" in instr_lower or "login" in instr_lower:
        for f in all_files:
            if "auth" in f.lower() or "user" in f.lower() or "main.py" in f.lower() or "tests/" in f.lower():
                selected.add(f)
    elif "docker" in instr_lower:
        for f in all_files:
            if "docker" in f.lower() or "requirements" in f.lower():
                selected.add(f)
    elif "health" in instr_lower:
        for f in all_files:
            if "main.py" in f.lower() or "tests/" in f.lower():
                selected.add(f)
    elif "db" in instr_lower or "postgres" in instr_lower or "sql" in instr_lower or "database" in instr_lower:
        for f in all_files:
            if "db" in f.lower() or "model" in f.lower() or "schema" in f.lower() or "main.py" in f.lower():
                selected.add(f)
    else:
        # Default: select important files
        for f in summary.get("important_files", []):
            selected.add(f)

    return sorted(list(selected)) if selected else all_files[:10]


def read_selected_file_contents(
    target_root: Path,
    selected_files: List[str],
    max_file_size: int = 40000,
) -> str:
    """
    Safely read contents of selected files using validate_file_path.
    """
    target_resolved = target_root.resolve()
    content_lines: List[str] = ["=== Relevant Selected File Contents ==="]

    for rel_path in selected_files:
        try:
            file_path = validate_file_path(rel_path, target_resolved)
            if not file_path.exists() or not file_path.is_file():
                continue
            if file_path.name in IGNORED_FILENAMES or file_path.suffix.lower() in IGNORED_EXTENSIONS:
                continue

            text = file_path.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_file_size:
                text = text[:max_file_size] + "\n... [content truncated]"
            content_lines.append(f"\n--- File: {rel_path} ---\n{text}\n")
        except Exception:
            pass

    return "\n".join(content_lines)
