"""
AIForge Codegen ZIP Packager
============================

Packages completed project codebases into downloadable ZIP archives with security filtering.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional, Set

from app.codegen.errors import PackagingError


DEFAULT_EXCLUDE_FILES: Set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".DS_Store",
    "secrets.json",
    "id_rsa",
    "id_rsa.pub",
}

DEFAULT_EXCLUDE_EXTENSIONS: Set[str] = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".key",
    ".pem",
    ".crt",
    ".tmp",
}

DEFAULT_EXCLUDE_DIRS: Set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".hg",
    ".svn",
    "venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
}


def package_project_zip(
    target_root: Path,
    output_zip_path: Path,
    exclude_dirs: Optional[Set[str]] = None,
    exclude_files: Optional[Set[str]] = None,
) -> Path:
    """
    Package target_root directory into output_zip_path archive.

    Parameters
    ----------
    target_root : Path
        Directory to archive.
    output_zip_path : Path
        Target destination path for .zip file.
    exclude_dirs : Optional[Set[str]]
        Directory names to skip.
    exclude_files : Optional[Set[str]]
        File names to skip.

    Returns
    -------
    Path
        Path to generated ZIP archive.
    """
    root_resolved = target_root.resolve()
    if not root_resolved.exists() or not root_resolved.is_dir():
        raise PackagingError(f"Target directory does not exist or is not a directory: '{target_root}'")

    out_resolved = output_zip_path.resolve()
    out_resolved.parent.mkdir(parents=True, exist_ok=True)

    dirs_to_skip = exclude_dirs if exclude_dirs is not None else DEFAULT_EXCLUDE_DIRS
    files_to_skip = exclude_files if exclude_files is not None else DEFAULT_EXCLUDE_FILES

    try:
        with zipfile.ZipFile(out_resolved, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(root_resolved):
                # Filter out excluded subdirectories in-place
                dirs[:] = [d for d in dirs if d not in dirs_to_skip]

                rel_dir = Path(root).relative_to(root_resolved)

                for file_name in files:
                    if file_name in files_to_skip:
                        continue

                    file_path = Path(root) / file_name
                    if file_path.suffix.lower() in DEFAULT_EXCLUDE_EXTENSIONS:
                        continue

                    # Skip output_zip_path if it resides inside target_root
                    if file_path.resolve() == out_resolved:
                        continue

                    arc_name = (rel_dir / file_name).as_posix()
                    zip_file.write(file_path, arcname=arc_name)

        return out_resolved

    except Exception as exc:
        raise PackagingError(f"Failed to create ZIP package: {exc}") from exc


def cleanup_old_snapshots(snapshots_dir: Path, keep_latest: int = 3) -> int:
    """
    Remove old snapshot directories, retaining only the `keep_latest` newest snapshots.
    Never deletes active/current project files or output ZIP artifacts.
    """
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return 0

    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir() and d.name.startswith("snapshot_")]
    snapshot_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    removed_count = 0
    if len(snapshot_dirs) > keep_latest:
        for old_dir in snapshot_dirs[keep_latest:]:
            try:
                import shutil
                shutil.rmtree(old_dir)
                removed_count += 1
            except Exception:
                pass
    return removed_count
