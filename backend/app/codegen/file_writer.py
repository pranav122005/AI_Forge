"""
AIForge Codegen File Writer & Explorer
======================================

Handles safe filesystem write operations and file tree browsing for generated projects.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.codegen.errors import PathValidationError
from app.codegen.models import FileSpec, FileTreeNode
from app.codegen.validator import sanitize_relative_path, validate_file_path


def write_file_spec(target_root: Path, file_spec: FileSpec) -> Path:
    """
    Safely write a single FileSpec to target_root.
    """
    target_file = validate_file_path(file_spec.path, target_root)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(file_spec.content, encoding="utf-8")
    return target_file


def write_file_specs(target_root: Path, files: List[FileSpec]) -> List[str]:
    """
    Safely write a list of FileSpecs to target_root.

    Returns
    -------
    List[str]
        List of relative path strings written.
    """
    written_paths: List[str] = []
    for spec in files:
        target_file = write_file_spec(target_root, spec)
        rel_path = target_file.relative_to(target_root.resolve()).as_posix()
        written_paths.append(rel_path)
    return written_paths


def build_file_tree(target_root: Path, current_path: Optional[Path] = None) -> FileTreeNode:
    """
    Build a recursive FileTreeNode hierarchy for target_root.
    """
    root_resolved = target_root.resolve()
    if current_path is None:
        current_path = root_resolved

    relative_posix = (
        current_path.relative_to(root_resolved).as_posix()
        if current_path != root_resolved
        else "."
    )

    if current_path.is_file():
        return FileTreeNode(
            name=current_path.name,
            path=relative_posix,
            is_dir=False,
            size=current_path.stat().st_size,
            children=None,
        )

    children: List[FileTreeNode] = []
    ignored_names = {".git", "__pycache__", ".pytest_cache", "venv", ".venv", "node_modules"}

    for item in sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if item.name in ignored_names or item.name.endswith(".pyc"):
            continue
        children.append(build_file_tree(target_root, item))

    return FileTreeNode(
        name=current_path.name if current_path != root_resolved else target_root.name,
        path=relative_posix,
        is_dir=True,
        size=None,
        children=children,
    )


def read_file_content(target_root: Path, relative_path: str) -> str:
    """
    Safely read file content as UTF-8 string.
    """
    target_file = validate_file_path(relative_path, target_root)
    if not target_file.exists() or not target_file.is_file():
        raise FileNotFoundError(f"File not found: '{relative_path}'")
    return target_file.read_text(encoding="utf-8", errors="replace")
