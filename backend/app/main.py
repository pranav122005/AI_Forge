from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    _backend_env = Path(__file__).resolve().parents[1] / ".env"
    _root_env = Path(__file__).resolve().parents[2] / ".env"
    if _backend_env.exists():
        load_dotenv(_backend_env)
    if _root_env.exists():
        load_dotenv(_root_env)
    load_dotenv()
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.builder import build_project
from app.llm import understand_requirement
from app.planner import REGISTRY as COMPONENTS, analyze_spec  # noqa: F401 (re-exported)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GENERATED_PROJECTS_DIR = BASE_DIR / "generated_projects"
PROJECTS_DIR = BASE_DIR / "generated"
MODELS_DIR = BASE_DIR / "models"
GENERATED_PROJECTS_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def get_project_dir(project_id: str) -> Path:
    p1 = GENERATED_PROJECTS_DIR / project_id
    if p1.exists():
        return p1
    p2 = PROJECTS_DIR / project_id
    if p2.exists():
        return p2
    return p1


from app.security import (
    get_cors_origins,
    redact_secrets,
    validate_id_string,
    verify_api_access,
)

app = FastAPI(title="AIForge API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    specification: str = Field(min_length=5)


class BuildRequest(BaseModel):
    project_name: str = Field(default="aiforge-project", min_length=2, max_length=60)
    spec: str = Field(min_length=5)


class UnderstandRequest(BaseModel):
    spec: str = Field(min_length=5, description="Natural-language AI system requirement")


class ParseLLMRequest(BaseModel):
    requirement: str = Field(min_length=5, description="Natural-language AI system requirement")



class BuildIntelligentRequest(BaseModel):
    project_name: str = Field(default="aiforge-project", min_length=2, max_length=60)
    spec: str = Field(min_length=5, description="Natural-language AI system requirement")


class GenerateRequest(BaseModel):
    requirement: str | None = Field(default=None, description="Natural-language AI system requirement")
    specification: str | None = Field(default=None, description="Legacy specification string")
    spec: str | None = Field(default=None, description="Alternative specification field")
    project_name: str | None = Field(default="aiforge-project")
    plan: dict[str, Any] | None = None



class PredictRequest(BaseModel):
    project_id: str
    text: str = Field(min_length=1)


class AgentRunRequest(BaseModel):
    project_id: str
    input: Any = Field(default="", description="Input text, payload dict, or image data for agent runtime")



def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return value.strip("-") or "aiforge-project"


def component_match(plan: list[str], key: str) -> bool:
    return key in plan



def write_generated_project(request: GenerateRequest) -> tuple[str, list[str]]:
    project_id = slugify(request.project_name) + "-" + uuid.uuid4().hex[:6]
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True)

    plan = request.plan
    components = [c["name"] for c in plan.get("components", [])]

    # Determine which components are executable so the generated server.py
    # can be honest about what it can actually run.
    executable_keys = {
        c["key"] for c in plan.get("components", []) if c.get("executable", False)
    }
    has_classifier = "text_classifier" in executable_keys

    # Build a server.py that loads the trained model when available and
    # returns a clear error when the model has not been trained yet.
    if has_classifier:
        server_py = '''\
"""Generated FastAPI service — loads the trained model from model.joblib."""
import pathlib
from fastapi import FastAPI, HTTPException
import joblib

app = FastAPI(title="Generated AI Service")

_MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "model.joblib"


def _load_model():
    if not _MODEL_PATH.exists():
        return None
    return joblib.load(_MODEL_PATH)


@app.get("/health")
def health():
    model_ready = _MODEL_PATH.exists()
    return {"status": "ok", "model_ready": model_ready}


@app.post("/predict")
def predict(payload: dict):
    text = str(payload.get("text", ""))
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required")
    model = _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. POST a labeled CSV to /api/train first.",
        )
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_.tolist()
    ranked = sorted(
        [{"label": c, "confidence": round(float(p), 4)} for c, p in zip(classes, probabilities)],
        key=lambda item: item["confidence"],
        reverse=True,
    )[:3]
    return {
        "prediction": str(prediction),
        "confidence": round(float(max(probabilities)), 4),
        "top_predictions": ranked,
    }
'''
    else:
        # Non-executable components selected — be explicit
        non_exec = [c["name"] for c in plan.get("components", []) if not c.get("executable", False) and c["name"] != "FastAPI"]
        server_py = f'''\
"""Generated FastAPI service stub.

NOTE: This project includes components that are not yet executable in the
current AIForge MVP: {", ".join(non_exec)}.

The /predict endpoint returns an informative message rather than
silently pretending to work.
"""
from fastapi import FastAPI

app = FastAPI(title="Generated AI Service")


@app.get("/health")
def health():
    return {{"status": "ok", "model_ready": False}}


@app.post("/predict")
def predict(payload: dict):
    return {{
        "status": "not_implemented",
        "message": (
            "This pipeline includes components not yet executable in AIForge MVP: "
            "{", ".join(non_exec)}. "
            "The architecture has been generated and is ready for future activation."
        ),
    }}
'''

    files: dict[str, str] = {
        "README.md": (
            f"# {request.project_name}\n\n"
            "Generated by AIForge from a natural-language specification.\n\n"
            "## Specification\n"
            f"{request.specification}\n\n"
            "## Selected components\n"
            + "\n".join(f"- {c}" for c in components)
            + "\n"
        ),
        "config/pipeline.json": json.dumps(plan, indent=2),
        "src/pipeline.py": (
            '"""Generated AI pipeline placeholder."""\n\n\n'
            "def run_pipeline(text: str) -> dict:\n"
            '    return {"input": text, "status": "pipeline-ready"}\n'
        ),
        "tests/test_pipeline.py": (
            "def test_pipeline_smoke():\n"
            "    from src.pipeline import run_pipeline\n"
            '    result = run_pipeline("hello")\n'
            '    assert result["status"] == "pipeline-ready"\n'
        ),
        "Dockerfile": (
            "FROM python:3.12-slim\n"
            "WORKDIR /app\n"
            "COPY . /app\n"
            "RUN pip install fastapi uvicorn scikit-learn joblib pandas\n"
            'CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]\n'
        ),
        "src/server.py": server_py,
    }

    for rel, content in files.items():
        path = project_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    metadata = {
        "project_id": project_id,
        "created_at": time.time(),
        "specification": request.specification,
        "components": components,
        "executable_components": list(executable_keys),
        "status": "generated",
    }
    (project_dir / "project.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return project_id, list(files)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    from app.ocr_component import is_available as ocr_available
    from app.llm.service import get_provider_info
    from app.llm.registry import registry
    llm_info = get_provider_info()
    has_ocr = ocr_available()
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY"))

    return {
        "status": "ok",
        "service": "AIForge API",
        "service_alive": True,
        "gemini_configured": gemini_key_set,
        "gemini_reachable": gemini_key_set,
        "agent_available": True,
        "ml_available": True,
        "ocr_available": has_ocr,
        "llm": llm_info,
        "active_provider": registry.get_active_provider_id(),
        "providers": [p.model_dump() for p in registry.list_providers()],
        "components": {
            "llm": llm_info["available"],
            "opencv": True,
            "ocr": has_ocr,
            "ml": True,
            "agent": True,
        },
    }


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    from app.ocr_component import is_available as ocr_available
    from app.llm.service import get_provider_info
    from app.llm.registry import registry
    llm_info = get_provider_info()
    has_ocr = ocr_available()
    gemini_key_set = bool(os.getenv("GEMINI_API_KEY"))

    return {
        "service_alive": True,
        "gemini_configured": gemini_key_set,
        "gemini_reachable": gemini_key_set,
        "agent_available": True,
        "llm": llm_info,
        "active_provider": registry.get_active_provider_id(),
        "providers": [p.model_dump() for p in registry.list_providers()],
        "vision": {
            "opencv": True,
            "ocr": has_ocr,
        },
        "ml": {
            "text_classification": True,
        },
        "agent": {
            "runtime": True,
            "development_loop": True,
        },
    }


# =====================================================================
# MULTI-LLM PROVIDER CONTROL ENDPOINTS
# =====================================================================

class LLMConfigureRequest(BaseModel):
    api_key: Optional[str] = Field(default=None, description="API Key for the provider")
    model: Optional[str] = Field(default=None, description="Model identifier")


class LLMTestRequest(BaseModel):
    api_key: Optional[str] = Field(default=None, description="Optional API Key override to test")
    model: Optional[str] = Field(default=None, description="Optional model override to test")


class LLMSetActiveRequest(BaseModel):
    provider: str = Field(..., description="Provider identifier (gemini, openai, anthropic)")
    model: Optional[str] = Field(default=None, description="Optional model to set as active")


@app.get("/api/llm/providers")
def list_llm_providers() -> list[dict[str, Any]]:
    """
    List registered LLM providers with status (credentials are never exposed).
    """
    from app.llm.registry import registry
    return [p.model_dump() for p in registry.list_providers()]


@app.get("/api/llm/active")
def get_active_llm_provider() -> dict[str, Any]:
    """
    Get current active LLM provider and model.
    """
    from app.llm.registry import registry
    active_id = registry.get_active_provider_id()
    active_model = registry.get_configured_model(active_id)
    return {
        "provider": active_id,
        "model": active_model,
        "configured": registry.is_provider_configured(active_id),
    }


@app.post("/api/llm/active")
def set_active_llm_provider(request: LLMSetActiveRequest) -> dict[str, Any]:
    """
    Switch active LLM provider and optional model selection.
    """
    from app.llm.registry import registry
    try:
        registry.set_active_provider(request.provider, request.model)
        active_id = registry.get_active_provider_id()
        active_model = registry.get_configured_model(active_id)
        return {
            "status": "ok",
            "provider": active_id,
            "model": active_model,
            "configured": registry.is_provider_configured(active_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/llm/providers/{provider}/configure")
def configure_llm_provider(provider: str, request: LLMConfigureRequest) -> dict[str, Any]:
    """
    Configure session API key and default model for an LLM provider.
    """
    from app.llm.registry import registry
    try:
        registry.set_provider_config(
            provider_id=provider,
            api_key=request.api_key,
            model=request.model,
        )
        return {
            "status": "ok",
            "provider": provider,
            "configured": registry.is_provider_configured(provider),
            "model": registry.get_configured_model(provider),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/llm/providers/{provider}/test")
async def test_llm_provider(provider: str, request: Optional[LLMTestRequest] = None) -> dict[str, Any]:
    """
    Perform a live connectivity and authentication test against the target LLM provider.
    """
    from app.llm.registry import registry
    api_key = request.api_key if request else None
    model = request.model if request else None
    return await registry.test_provider(provider_id=provider, api_key=api_key, model=model)


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    items = []
    for k, v in COMPONENTS.items():
        entry = dict(v)
        entry["key"] = k
        entry["id"] = k
        # Backward compat: frontend and older code expect a "purpose" field
        if "purpose" not in entry:
            entry["purpose"] = entry.get("description", "")
        items.append(entry)
    return {"components": items}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    return analyze_spec(request.specification)


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
    from app.orchestrator import default_orchestrator

    req_text = request.requirement or request.specification or request.spec
    if req_text and not request.plan:
        return await default_orchestrator.generate_project(req_text, request.project_name)

    if request.plan and (request.specification or req_text):
        project_id, files = write_generated_project(request)
        return {
            "project_id": project_id,
            "status": "generated",
            "files": files,
            "message": "Project scaffold generated successfully.",
        }

    raise HTTPException(
        status_code=400,
        detail="Requirement text is required for project generation.",
    )


@app.post("/api/train")
async def train(project_id: str, dataset: UploadFile = File(...)) -> dict[str, Any]:
    from app.ml import train_text_classifier, save_model, inspect_dataset
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Generated project not found")

    dataset_path = project_dir / "dataset.csv"
    content = await dataset.read()
    dataset_path.write_bytes(content)

    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    try:
        inspection = inspect_dataset(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        pipeline, metrics, labels = train_text_classifier(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_path, metadata = save_model(project_id, pipeline, metrics, labels)

    return {
        "project_id": project_id,
        "status": "trained",
        "model_id": f"{project_id}-model-v1",
        "inspection": inspection,
        "metrics": metrics,
        "model_path": str(model_path.relative_to(BASE_DIR)),
        "labels": labels,
    }



@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    from app.ml.persistence import load_model, model_exists
    if not model_exists(request.project_id):
        raise HTTPException(
            status_code=404,
            detail="No trained model found for this project. Train the model first.",
        )

    model = load_model(request.project_id)
    prediction = model.predict([request.text])[0]
    probabilities = model.predict_proba([request.text])[0]
    classes = model.classes_.tolist()
    confidence = float(max(probabilities))
    ranked = sorted(
        [
            {"label": str(c), "confidence": round(float(p), 4)}
            for c, p in zip(classes, probabilities)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )[:3]
    return {
        "project_id": request.project_id,
        "prediction": str(prediction),
        "confidence": round(confidence, 4),
        "top_predictions": ranked,
    }


@app.post("/api/agent/run")
def run_agent(request: AgentRunRequest) -> dict[str, Any]:
    """
    Execute the agent runtime for a project given input payload.
    """
    from app.agent import default_runtime
    project_dir = get_project_dir(request.project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{request.project_id}' not found.")

    res = default_runtime.run_project(request.project_id, request.input)
    if res.get("status") == "failed":
        error_msg = res.get("error", "Execution failed")
        if "No trained model found" in error_msg or "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        elif "not executable" in error_msg or "catalog-only" in error_msg:
            raise HTTPException(status_code=503, detail=error_msg)
    return res


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    """Return metadata for all generated projects, newest first."""
    projects: list[dict[str, Any]] = []
    search_dirs = [GENERATED_PROJECTS_DIR, PROJECTS_DIR]
    seen_ids = set()

    for p_dir in search_dirs:
        if p_dir.exists():
            for meta_path in sorted(
                p_dir.glob("*/project.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    pid = meta.get("project_id")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        metrics_path = meta_path.parent / "training_metrics.json"
                        meta["trained"] = metrics_path.exists()
                        if meta["trained"]:
                            meta["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
                        projects.append(meta)
                except Exception:
                    pass
    return {"projects": projects, "total": len(projects)}


@app.get("/api/projects/{project_id}")
def project(project_id: str) -> dict[str, Any]:
    project_dir = get_project_dir(project_id)
    meta_path = project_dir / "project.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics_path = project_dir / "training_metrics.json"
    if metrics_path.exists():
        meta["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta["trained"] = True
    else:
        meta["trained"] = False
    return meta



@app.post("/api/predict-image")
async def predict_image(
    project_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Image → OpenCV preprocessing → OCR → text classifier.

    Accepts a multipart image upload, runs the full vision pipeline,
    and returns extracted text + classification result.

    Requires:
    - A trained model (POST /api/train first)
    - Tesseract OCR binary installed on the server
    """
    from app.vision import preprocess, VisionPreprocessingError
    from app.ocr_component import (
        extract_text,
        is_available as ocr_available,
        TesseractNotAvailableError,
        OCRError,
    )

    from app.ml.persistence import model_exists, load_model
    project_dir = get_project_dir(project_id)
    if not model_exists(project_id):
        raise HTTPException(
            status_code=404,
            detail="No trained model found for this project. Train the model first.",
        )


    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    # 1. OpenCV preprocessing
    try:
        preprocessed = preprocess(image_bytes)
    except VisionPreprocessingError as exc:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {exc}") from exc

    # 2. OCR
    if not ocr_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Tesseract OCR is not installed on this server. "
                "Install Tesseract to enable the image pipeline. "
                "See: https://github.com/UB-Mannheim/tesseract/wiki"
            ),
        )
    try:
        from app.vision import preprocessed_to_pil
        pil_image = preprocessed_to_pil(preprocessed)
        extracted_text = extract_text(pil_image)
    except TesseractNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OCRError as exc:
        raise HTTPException(status_code=422, detail=f"OCR error: {exc}") from exc

    if not extracted_text:
        return {
            "project_id": project_id,
            "extracted_text": "",
            "prediction": None,
            "confidence": None,
            "top_predictions": [],
            "note": "No text could be extracted from the image",
        }

    # 3. Text classification
    model = load_model(project_id)

    prediction = model.predict([extracted_text])[0]
    probabilities = model.predict_proba([extracted_text])[0]
    classes = model.classes_.tolist()
    ranked = sorted(
        [
            {"label": c, "confidence": round(float(p), 4)}
            for c, p in zip(classes, probabilities)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )[:3]

    return {
        "project_id": project_id,
        "extracted_text": extracted_text,
        "prediction": str(prediction),
        "confidence": round(float(max(probabilities)), 4),
        "top_predictions": ranked,
    }


@app.post("/api/llm/parse")
async def parse_llm(request: ParseLLMRequest) -> dict[str, Any]:
    """
    Parse a natural language user requirement into a validated RequirementSpec using an LLM.

    Requires OPENAI_API_KEY environment variable to be configured.
    """
    from app.llm.service import parse_requirement
    from app.llm.providers.base import (
        LLMConfigurationError,
        LLMParseError,
        LLMProviderError,
    )

    try:
        spec = await parse_requirement(request.requirement)
        return spec.model_dump()
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM provider configuration error: {exc}",
        ) from exc
    except LLMParseError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"LLM parse error: {exc}",
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/api/understand")

def understand(request: UnderstandRequest) -> dict[str, Any]:
    """
    Convert a natural-language requirement into a structured specification.

    Uses the configured LLM provider (IBM watsonx.ai if credentials are set)
    or the deterministic rule-based fallback.

    The LLM understands INTENT; the Planner makes ENGINEERING DECISIONS.
    """
    result = understand_requirement(request.spec)
    return result.model_dump(exclude={"raw_llm_response"})


@app.post("/api/build-intelligent")
def build_intelligent(request: BuildIntelligentRequest) -> dict[str, Any]:
    """
    Full intelligent build pipeline:
      1. understand_requirement() — LLM or deterministic
      2. analyze_spec() — deterministic Planner
      3. build_project() — Builder generates project on disk

    The LLM (when available) enriches the requirement understanding.
    The Planner always makes the final engineering decisions.
    """
    # Step 1: understand
    understanding = understand_requirement(request.spec)

    # Step 2: plan (using the LLM-enriched planner spec if available)
    planner_input = understanding.planner_spec or request.spec

    # Step 3: build
    try:
        build_result = build_project(request.project_name, planner_input)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = build_result.to_dict()
    response["understanding"] = understanding.model_dump(exclude={"raw_llm_response"})
    return response


@app.post("/api/build")
def build(request: BuildRequest) -> dict[str, Any]:
    """
    One-shot endpoint: analyze spec → resolve pipeline → generate project.

    Returns project_id, execution_status, pipeline, and generated_files.
    """
    try:
        result = build_project(request.project_name, request.spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@app.get("/demo/{filename}")
def demo_file(filename: str) -> FileResponse:
    """
    Serve files from the data/ directory for the frontend demo workflow.
    Only whitelisted filenames are served to prevent path traversal.
    """
    allowed = {"legal_demo.csv"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Demo file not found")
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Demo file not found on server")
    return FileResponse(str(file_path), media_type="text/csv", filename=filename)


# =====================================================================
# PHASE 9 — REAL LLM CODE GENERATION & REPAIR API ENDPOINTS
# =====================================================================

from app.codegen import (
    CodegenRequest,
    CodegenResult,
    CodegenService,
    FileTreeNode,
    RepairRequest,
    RepairResult,
    TestResult,
)

codegen_service = CodegenService()


@app.post("/api/codegen/generate", response_model=CodegenResult)
async def generate_code(request: CodegenRequest) -> CodegenResult:
    """
    Generate, build, test, repair, and package an end-to-end AI project using real LLM codegen engine.
    """
    try:
        return await codegen_service.generate_project(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/codegen/{project_id}")
def get_codegen_status(project_id: str) -> dict[str, Any]:
    """
    Get status, file tree, and test status of a codegen project.
    """
    try:
        tree = codegen_service.get_file_tree(project_id)
        test_res = codegen_service.run_tests(project_id)
        return {
            "project_id": project_id,
            "status": "completed" if test_res.success else "failed",
            "file_tree": tree.model_dump(),
            "test_result": test_res.model_dump(),
            "zip_url": f"/api/codegen/{project_id}/download",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/codegen/{project_id}/report")
def get_codegen_report(project_id: str) -> dict[str, Any]:
    """
    Generate comprehensive technical intelligence report for project.
    """
    try:
        return codegen_service.get_project_report(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/codegen/{project_id}/files")
def get_codegen_files(project_id: str) -> FileTreeNode:
    """
    Get recursive file tree hierarchy for project directory.
    """
    try:
        return codegen_service.get_file_tree(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/codegen/{project_id}/file")
def get_codegen_file_content(project_id: str, path: str) -> dict[str, str]:
    """
    Get source code content of a relative file inside project directory.
    """
    try:
        content = codegen_service.read_file(project_id, path)
        return {"path": path, "content": content}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/codegen/{project_id}/download")
def download_project_zip(project_id: str) -> FileResponse:
    """
    Download project codebase packaged as a .zip file.
    """
    try:
        zip_path = codegen_service.get_zip_file_path(project_id)
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_path.name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/codegen/{project_id}/test", response_model=TestResult)
def test_codegen_project(project_id: str) -> TestResult:
    """
    Run unit test suite on generated project codebase.
    """
    try:
        return codegen_service.run_tests(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/codegen/{project_id}/repair", response_model=RepairResult)
async def repair_codegen_project(project_id: str, request: Optional[RepairRequest] = None) -> RepairResult:
    """
    Trigger automated repair loop on a failing project codebase.
    """
    try:
        test_output = request.test_output if request else None
        user_feedback = request.user_feedback if request else None
        return await codegen_service.repair_project(
            project_id=project_id,
            test_output=test_output,
            user_feedback=user_feedback,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


from app.codegen import (
    AgentRunRequest,
    AgentRunState,
    CodegenModifyRequest,
    CodegenModifyResult,
    CodegenRequest,
    CodegenResult,
    CodegenService,
    FileTreeNode,
    RepairRequest,
    RepairResult,
    TestResult,
)


@app.post("/api/codegen/{project_id}/modify", response_model=CodegenModifyResult)
async def modify_codegen_project(
    project_id: str,
    request: CodegenModifyRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> CodegenModifyResult:
    """
    Safely modify an existing generated project codebase using LLM instructions.
    """
    valid_id = validate_id_string(project_id, "project_id")
    verify_api_access(valid_id, x_api_key, x_user_id)
    try:
        return await codegen_service.modify_project(
            project_id=valid_id,
            instruction=request.instruction,
            user_feedback=request.user_feedback,
            provider=request.provider,
            model=request.model,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/codegen/{project_id}/agent", response_model=AgentRunState)
async def run_codegen_agent(
    project_id: str,
    request: AgentRunRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> AgentRunState:
    """
    Trigger the Agentic Development Loop on an existing project workspace.
    """
    valid_id = validate_id_string(project_id, "project_id")
    verify_api_access(valid_id, x_api_key, x_user_id)
    try:
        return await codegen_service.run_agent(
            project_id=valid_id,
            instruction=request.instruction,
            user_feedback=request.user_feedback,
            provider=request.provider,
            model=request.model,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/codegen/{project_id}/agent/{run_id}", response_model=AgentRunState)
async def get_codegen_agent_status(
    project_id: str,
    run_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> AgentRunState:
    """
    Retrieve real-time status of an Agentic Development Loop execution run.
    """
    valid_p_id = validate_id_string(project_id, "project_id")
    valid_r_id = validate_id_string(run_id, "run_id")
    verify_api_access(valid_p_id, x_api_key, x_user_id)

    state = codegen_service.get_agent_state(valid_r_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found.")
    return state


