"""
AIForge Model Registry & Persistence Module
===========================================

Manages saving, loading, and metadata tracking for trained ML models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
GENERATED_PROJECTS_DIR = BASE_DIR / "generated_projects"
PROJECTS_DIR = BASE_DIR / "generated"
MODELS_DIR = BASE_DIR / "models"
GENERATED_PROJECTS_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def get_project_dir(project_id: str) -> Path:
    """Get path to project directory."""
    p1 = GENERATED_PROJECTS_DIR / project_id
    if p1.exists():
        return p1
    p2 = PROJECTS_DIR / project_id
    if p2.exists():
        return p2
    return p1



def get_model_path(project_id: str) -> Path:
    """Get path to model.joblib for project_id."""
    return get_project_dir(project_id) / "model.joblib"


def model_exists(project_id: str) -> bool:
    """Return True if model.joblib exists for project_id."""
    return get_model_path(project_id).exists()


def save_model(
    project_id: str,
    pipeline: Pipeline,
    metrics: dict[str, Any],
    labels: list[str],
) -> tuple[Path, dict[str, Any]]:
    """
    Persist trained model and metadata for project_id.
    """
    project_dir = get_project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    model_path = project_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Also save to global MODELS_DIR registry for backup
    global_model_dir = MODELS_DIR / project_id
    global_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, global_model_dir / "model.joblib")

    metadata = {
        "project_id": project_id,
        "model_type": "tfidf_logistic_regression",
        "labels": labels,
        "feature_type": "tfidf_ngram_1_2",
        "training_timestamp": time.time(),
        "metrics": metrics,
        "dataset_info": {
            "samples": metrics.get("samples", 0),
            "classes": metrics.get("classes", 0),
        },
        "version": "1.0.0",
    }

    # Write metadata.json and training_metrics.json
    (project_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (project_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (global_model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Update project.json status to 'trained' if present
    project_json_path = project_dir / "project.json"
    if project_json_path.exists():
        try:
            p_meta = json.loads(project_json_path.read_text(encoding="utf-8"))
            p_meta["status"] = "trained"
            project_json_path.write_text(json.dumps(p_meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    return model_path, metadata


def load_model(project_id: str) -> Pipeline:
    """
    Load persisted sklearn pipeline for project_id.
    """
    model_path = get_model_path(project_id)
    if not model_path.exists():
        # Fallback check to global MODELS_DIR
        global_path = MODELS_DIR / project_id / "model.joblib"
        if global_path.exists():
            model_path = global_path
        else:
            raise FileNotFoundError(f"No trained model found for project '{project_id}'.")

    return joblib.load(model_path)


def get_model_metadata(project_id: str) -> dict[str, Any]:
    """
    Retrieve metadata for project_id's trained model.
    """
    meta_path = get_project_dir(project_id) / "metadata.json"
    if not meta_path.exists():
        meta_path = MODELS_DIR / project_id / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No model metadata found for project '{project_id}'.")

    return json.loads(meta_path.read_text(encoding="utf-8"))
