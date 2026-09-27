"""
AIForge Codegen Path Validator
==============================

Strict file path validation module to prevent directory traversal and insecure file writes.
"""
from __future__ import annotations

import os
from pathlib import Path
from app.codegen.errors import PathValidationError


def sanitize_relative_path(raw_path: str) -> str:
    """
    Clean up raw relative path strings.
    """
    clean = raw_path.strip().replace("\\", "/")
    if len(clean) >= 2 and clean[1] == ":":
        clean = clean[2:]
    clean = clean.lstrip("/")
    return clean


FORBIDDEN_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "secrets.json",
    "id_rsa",
    "id_rsa.pub",
    "credentials.json",
}


def validate_file_path(relative_path: str, target_root: Path) -> Path:
    """
    Validate that relative_path points safely inside target_root directory.
    Rejects path traversal, absolute paths, and forbidden protected files.
    """
    if not relative_path or not relative_path.strip():
        raise PathValidationError("File path cannot be empty.")

    raw = relative_path.strip()

    # Reject absolute paths and drive letters directly
    if raw.startswith("/") or raw.startswith("\\") or (len(raw) >= 2 and raw[1] == ":"):
        raise PathValidationError(f"Absolute paths or drive letters are not allowed: '{relative_path}'")

    clean_rel = raw.replace("\\", "/")
    parts = Path(clean_rel).parts
    if ".." in parts:
        raise PathValidationError(f"Path traversal ('..') detected in path: '{relative_path}'")

    file_name = Path(clean_rel).name.lower()
    if file_name in FORBIDDEN_FILES or ".git" in parts:
        raise PathValidationError(f"Modification or generation of protected file '{relative_path}' is forbidden.")

    root_resolved = target_root.resolve()
    target_path = (root_resolved / clean_rel).resolve()

    try:
        target_path.relative_to(root_resolved)
    except ValueError:
        raise PathValidationError(
            f"Path '{relative_path}' escapes target directory '{target_root}'."
        )

    return target_path
