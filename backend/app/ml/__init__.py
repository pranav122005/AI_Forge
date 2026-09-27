"""
AIForge ML Package
==================

Provides dataset validation, TF-IDF + Logistic Regression training, evaluation,
model persistence, and metadata registry.
"""
from app.ml.dataset import inspect_dataset
from app.ml.evaluator import evaluate_model
from app.ml.model import build_classifier_pipeline
from app.ml.persistence import (
    get_model_metadata,
    get_model_path,
    get_project_dir,
    load_model,
    model_exists,
    save_model,
)
from app.ml.preprocessing import validate_and_clean_dataframe
from app.ml.trainer import train_text_classifier

__all__ = [
    "inspect_dataset",
    "validate_and_clean_dataframe",
    "build_classifier_pipeline",
    "evaluate_model",
    "train_text_classifier",
    "save_model",
    "load_model",
    "model_exists",
    "get_model_metadata",
    "get_project_dir",
    "get_model_path",
]

