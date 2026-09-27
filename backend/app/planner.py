"""
AIForge Intelligent Pipeline Planner
=====================================

This module is the AI system architect brain of AIForge. It receives a
natural-language specification and produces a fully structured pipeline plan:

* Detects the **input type** (image, document, text, video, …)
* Identifies every required **AI capability** independently
* Resolves **component dependencies** (e.g. OCR depends on OpenCV for image input)
* Assigns the correct **execution order**
* Annotates every component with its real runtime status
* Determines **execution_ready** — only true when every required runtime component
  is actually executable in the current MVP

Design principles
-----------------
* No LLM dependency: the planner is a deterministic keyword-based classifier.
* Each capability is evaluated independently — no capability suppresses another.
* Dependency injection is explicit: if OCR is selected AND input is image-based,
  OpenCV is automatically required even if not mentioned explicitly.
* Honesty: catalog-only components appear in the architecture with their real
  status; we never claim something is executable when it is not.
* Backward compatibility: analyze_spec() still returns all fields expected by the
  existing API (selected_components, components, reasons, pipeline, …) plus the
  new structured fields.
"""
from __future__ import annotations

import uuid
from typing import Any

# Probe real component availability at import time so the registry
# reflects the actual runtime environment.
def _ocr_is_executable() -> bool:
    """Return True only when Tesseract binary is present and reachable."""
    try:
        from app.ocr_component import is_available
        return is_available()
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Expanded component registry
# ---------------------------------------------------------------------------
# Every entry exposes:
#   id           — machine key
#   name         — human name
#   category     — Vision / Document / Language / Data / Retrieval / Serving
#   description  — what it does
#   icon         — frontend icon name
#   input_type   — what it consumes  (list[str])
#   output_type  — what it produces  (list[str])
#   dependencies — component ids that must precede this one (list[str])
#   status       — "implemented" | "catalog"
#   executable   — can be invoked end-to-end in the current MVP (bool)
#   tested       — covered by a passing automated test (bool)
# ---------------------------------------------------------------------------

REGISTRY: dict[str, dict[str, Any]] = {
    # ---- Vision ----
    "opencv": {
        "name": "OpenCV",
        "category": "Vision",
        "description": "Image preprocessing: grayscale, denoising, binarisation, deskew",
        "icon": "camera",
        "input_type": ["image", "video", "camera"],
        "output_type": ["processed_image"],
        "dependencies": [],
        "status": "implemented",
        "executable": True,
        "tested": True,
        "requires_system_binary": False,
        "python_packages": ["opencv-python-headless>=4.9", "Pillow>=10.0"],
    },
    "image_classification": {
        "name": "Image Classifier",
        "category": "Vision",
        "description": "Classify images into predefined categories using a CNN",
        "icon": "camera",
        "input_type": ["image", "processed_image"],
        "output_type": ["label"],
        "dependencies": ["opencv"],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    "object_detection": {
        "name": "Object Detector",
        "category": "Vision",
        "description": "Detect and localise objects in images or video frames",
        "icon": "camera",
        "input_type": ["image", "video", "processed_image"],
        "output_type": ["bounding_boxes"],
        "dependencies": ["opencv"],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    "video_processing": {
        "name": "Video Processor",
        "category": "Vision",
        "description": "Extract frames, analyse video streams, temporal processing",
        "icon": "camera",
        "input_type": ["video"],
        "output_type": ["frames"],
        "dependencies": ["opencv"],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    # ---- Document ----
    "ocr": {
        "name": "OCR",
        "category": "Document",
        "description": "Extract text from images using Tesseract OCR (requires Tesseract system binary)",
        "icon": "scan",
        "input_type": ["image", "processed_image", "document_image"],
        "output_type": ["text"],
        "dependencies": ["opencv"],
        "status": "implemented",
        "executable": _ocr_is_executable(),  # True only when tesseract binary present
        "requires_system_binary": True,
        "system_binary": "tesseract",
        "install_guide": "https://github.com/UB-Mannheim/tesseract/wiki",
        "tested": True,
    },
    "pdf_parser": {
        "name": "PDF Parser",
        "category": "Document",
        "description": "Parse native PDFs and extract structured text and metadata",
        "icon": "scan",
        "input_type": ["pdf"],
        "output_type": ["text"],
        "dependencies": [],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    "document_classification": {
        "name": "Document Classifier",
        "category": "Document",
        "description": "Classify documents by type (contract, petition, affidavit, …)",
        "icon": "brain",
        "input_type": ["text"],
        "output_type": ["label"],
        "dependencies": [],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    # ---- Language ----
    "text_classifier": {
        "name": "Task Classifier",
        "category": "Language",
        "description": "Train a task-specific text classifier from labeled data (TF-IDF + LogReg)",
        "icon": "brain",
        "input_type": ["text"],
        "output_type": ["label", "confidence"],
        "dependencies": [],
        "status": "implemented",
        "executable": True,
        "tested": True,
    },
    "llm": {
        "name": "LLM",
        "category": "Language",
        "description": "Large language model for generation, reasoning, summarization, QA",
        "icon": "sparkles",
        "input_type": ["text"],
        "output_type": ["text"],
        "dependencies": [],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    "summarization": {
        "name": "Summarization",
        "category": "Language",
        "description": "Condense long documents into concise summaries (LLM-backed)",
        "icon": "sparkles",
        "input_type": ["text"],
        "output_type": ["summary"],
        "dependencies": ["llm"],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    # ---- Data ----
    # csv and preprocessing are used internally by the training pipeline but
    # are not standalone user-facing pipeline components — they are not
    # independently invokable, so executable=False is correct.
    "csv": {
        "name": "CSV Loader",
        "category": "Data",
        "description": "Load and parse CSV datasets for training or batch inference",
        "icon": "database",
        "input_type": ["file"],
        "output_type": ["dataframe"],
        "dependencies": [],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    "preprocessing": {
        "name": "Data Preprocessor",
        "category": "Data",
        "description": "Text cleaning, normalisation, tokenisation",
        "icon": "database",
        "input_type": ["text", "dataframe"],
        "output_type": ["text", "dataframe"],
        "dependencies": [],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    # ---- Retrieval ----
    "embeddings": {
        "name": "Embeddings",
        "category": "Retrieval",
        "description": "Compute semantic vector representations for retrieval and similarity",
        "icon": "network",
        "input_type": ["text"],
        "output_type": ["vector"],
        "dependencies": [],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    "vector_db": {
        "name": "Vector Store",
        "category": "Retrieval",
        "description": "Store and retrieve semantic vectors (Chroma, Pinecone, pgvector)",
        "icon": "database",
        "input_type": ["vector"],
        "output_type": ["vector", "text"],
        "dependencies": ["embeddings"],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    "rag": {
        "name": "RAG Pipeline",
        "category": "Retrieval",
        "description": "Retrieval-augmented generation: retrieve context then generate",
        "icon": "network",
        "input_type": ["text", "vector"],
        "output_type": ["text"],
        "dependencies": ["embeddings", "vector_db", "llm"],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
    # ---- Serving ----
    "fastapi": {
        "name": "FastAPI",
        "category": "Serving",
        "description": "Expose the AI pipeline as a REST API",
        "icon": "server",
        "input_type": ["any"],
        "output_type": ["http_response"],
        "dependencies": [],
        "status": "implemented",
        "executable": True,
        "tested": True,
    },
    "batch_inference": {
        "name": "Batch Inference",
        "category": "Serving",
        "description": "Run predictions on large datasets in batch mode",
        "icon": "server",
        "input_type": ["dataframe"],
        "output_type": ["dataframe"],
        "dependencies": ["text_classifier"],
        "status": "catalog",
        "executable": False,
        "tested": False,
    },
}

# Legacy alias: callers that only know the 7 original components still work
COMPONENTS = REGISTRY


# ---------------------------------------------------------------------------
# Term maps for keyword detection
# ---------------------------------------------------------------------------

# input_type detection
_INPUT_TERMS: list[tuple[str, list[str]]] = [
    ("image", ["image", "photo", "picture", "photograph", "camera", "visual"]),
    ("video", ["video", "stream", "footage", "frame", "webcam"]),
    ("document_image", ["scanned document", "scanned pdf", "document photo",
                        "image of document", "images of document",
                        "photo of document", "scan of"]),
    ("pdf", ["pdf", ".pdf", "native pdf", "digital pdf"]),
    ("text", ["text", "document", "legal document", "article", "paragraph",
              "sentence", "input text", "customer feedback", "review"]),
    ("csv", ["csv", "spreadsheet", "tabular", "dataset"]),
    ("json", ["json", "api response", "structured data"]),
]

# capability → (component_ids, explanation_template)
_CAPABILITY_RULES: list[tuple[list[str], list[str], str]] = [
    # (terms, component_ids, explanation)
    (
        ["classify", "classifies", "classification", "categorize", "categorizes",
         "category", "categories", "document type", "detect type", "label",
         "identify type"],
        ["text_classifier"],
        "text_classifier selected because the specification requires category/label prediction.",
    ),
    (
        ["summarize", "summarizes", "summarization", "summary", "summaries",
         "condense", "abstract", "tldr"],
        ["llm", "summarization"],
        "LLM + Summarization selected because the specification requires condensing text into summaries.",
    ),
    (
        ["generate", "generation", "write", "compose", "draft", "chatbot",
         "conversational", "dialogue"],
        ["llm"],
        "LLM selected because the specification requires text generation or a conversational interface.",
    ),
    (
        ["question answering", "question answer", "qa", "answer questions",
         "reasoning", "reason about"],
        ["llm"],
        "LLM selected because the specification requires question answering or reasoning.",
    ),
    (
        ["search", "semantic search", "similar", "similarity", "nearest",
         "knowledge base", "retrieve", "retrieval", "rag",
         "augmented generation"],
        ["embeddings", "vector_db"],
        "Embeddings + Vector Store selected because the specification requires semantic retrieval.",
    ),
    (
        ["rag", "retrieval augmented", "augmented generation"],
        ["embeddings", "vector_db", "rag"],
        "Full RAG pipeline selected: embeddings, vector store, and retrieval-augmented generation.",
    ),
    (
        ["extract text", "extracts text", "extracting text", "text extraction",
         "ocr", "scan text", "read text from image"],
        ["ocr"],
        "OCR selected because the specification requires text extraction from images or documents.",
    ),
    (
        ["detect object", "object detection", "detect items", "locate object",
         "find object", "bounding box"],
        ["object_detection"],
        "Object Detection selected because the specification requires locating objects in images.",
    ),
    (
        ["image classification", "classify image", "classify photo",
         "identify image", "recognise image", "recognize image"],
        ["image_classification"],
        "Image Classifier selected because the specification requires classifying images.",
    ),
    (
        ["process video", "video analysis", "video processing",
         "analyse video", "analyze video", "video stream"],
        ["video_processing"],
        "Video Processor selected because the specification requires video analysis.",
    ),
    (
        ["parse pdf", "pdf parsing", "extract from pdf", "pdf text",
         "read pdf", "digital pdf"],
        ["pdf_parser"],
        "PDF Parser selected because the specification requires extracting text from native PDF files.",
    ),
    (
        ["batch", "batch inference", "bulk prediction", "bulk classify"],
        ["batch_inference"],
        "Batch Inference selected because the specification requires processing data in bulk.",
    ),
    (
        ["rest api", "api endpoint", "http endpoint", "expose api",
         "web service", "microservice", "serve", "serving", "endpoint"],
        ["fastapi"],
        "FastAPI selected because the specification explicitly requests an API serving layer.",
    ),
]

# vision input terms — used to auto-inject opencv when image/video is detected
_VISION_INPUT_TERMS = ["image", "photo", "picture", "photograph", "camera",
                       "visual", "video", "stream", "footage", "frame",
                       "scanned", "scan", "document image", "images of",
                       "photo of"]


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

def _resolve_dependencies(selected_ids: list[str]) -> list[str]:
    """
    Given a list of selected component IDs, expand it to include all required
    dependencies (transitively), maintaining topological order.

    The returned list preserves the original insertion order and appends
    any newly injected dependencies directly before the component that needs them.
    """
    result: list[str] = []
    seen: set[str] = set()

    def _visit(cid: str) -> None:
        if cid in seen:
            return
        if cid not in REGISTRY:
            return
        # Visit all dependencies first
        for dep in REGISTRY[cid].get("dependencies", []):
            _visit(dep)
        seen.add(cid)
        result.append(cid)

    for cid in selected_ids:
        _visit(cid)

    return result


# ---------------------------------------------------------------------------
# Canonical pipeline ordering
# ---------------------------------------------------------------------------

_PIPELINE_ORDER: list[str] = [
    "opencv",
    "video_processing",
    "image_classification",
    "object_detection",
    "ocr",
    "pdf_parser",
    "document_classification",
    "csv",
    "preprocessing",
    "text_classifier",
    "embeddings",
    "vector_db",
    "rag",
    "llm",
    "summarization",
    "batch_inference",
    "fastapi",
]


def _sort_pipeline(component_ids: list[str]) -> list[str]:
    """Sort component IDs into canonical pipeline execution order."""
    order_index = {k: i for i, k in enumerate(_PIPELINE_ORDER)}
    # Components not in the canonical list go at the end, preserving
    # their relative order.
    max_idx = len(_PIPELINE_ORDER)
    return sorted(
        component_ids,
        key=lambda cid: order_index.get(cid, max_idx),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_spec(spec: str) -> dict[str, Any]:
    """
    Analyse a natural-language AI specification and return a fully structured
    pipeline plan.

    Returns
    -------
    dict with keys:
        id, specification, input, task,
        selected_components (list[str] — backward compat),
        components (list[dict] — full metadata),
        pipeline (list[str] — human-readable step names — backward compat),
        pipeline_steps (list[dict] — structured with order/status/reason),
        reasons (list[dict] — backward compat),
        implemented_components, catalog_components,
        estimated_complexity,
        warnings (list[str]),
        execution_ready (bool),
        mvp_note (str),
    """
    s = spec.lower()

    # ------------------------------------------------------------------ #
    # 1. Detect input type
    # ------------------------------------------------------------------ #
    detected_inputs: list[str] = []
    for input_type, terms in _INPUT_TERMS:
        if any(t in s for t in terms):
            detected_inputs.append(input_type)

    # Normalise: if document_image is detected, image is implied
    if "document_image" in detected_inputs and "image" not in detected_inputs:
        detected_inputs.insert(0, "image")

    primary_input = detected_inputs[0] if detected_inputs else "text"
    is_image_input = any(t in s for t in _VISION_INPUT_TERMS)

    # ------------------------------------------------------------------ #
    # 2. Detect capabilities → select components
    # ------------------------------------------------------------------ #
    selected: list[str] = []
    reasons: list[dict[str, str]] = []
    reason_map: dict[str, str] = {}  # component_id → reason text

    for terms, comp_ids, explanation in _CAPABILITY_RULES:
        if any(t in s for t in terms):
            for cid in comp_ids:
                if cid not in selected:
                    selected.append(cid)
                    if cid not in reason_map:
                        reason_map[cid] = explanation

    # ------------------------------------------------------------------ #
    # 3. Auto-inject vision pipeline when image/video input is detected
    #    and no vision component was explicitly requested
    # ------------------------------------------------------------------ #
    needs_vision = is_image_input and not any(
        cid in selected for cid in ["opencv", "image_classification",
                                    "object_detection", "video_processing"]
    )
    # If OCR is already selected but opencv is not, inject it (OCR depends on opencv)
    needs_opencv_for_ocr = "ocr" in selected and "opencv" not in selected

    if needs_vision or needs_opencv_for_ocr:
        if "opencv" not in selected:
            selected.insert(0, "opencv")
            reason_map["opencv"] = (
                "OpenCV selected because the specification involves image/video input; "
                "it provides the image preprocessing layer required by downstream components."
            )

    # ------------------------------------------------------------------ #
    # 4. Fallback — if still nothing selected, add text_classifier
    # ------------------------------------------------------------------ #
    # Remove fastapi from the "nothing selected" check — it doesn't count
    non_serving = [c for c in selected if c != "fastapi"]
    if not non_serving:
        selected.append("text_classifier")
        reason_map["text_classifier"] = (
            "Task Classifier selected as the minimal executable AI path "
            "for the MVP (no specific capability was detected)."
        )

    # ------------------------------------------------------------------ #
    # 5. Always add fastapi as the serving layer
    # ------------------------------------------------------------------ #
    if "fastapi" not in selected:
        selected.append("fastapi")
        reason_map["fastapi"] = (
            "FastAPI selected because every generated AI system receives "
            "an API serving layer."
        )

    # ------------------------------------------------------------------ #
    # 6. Resolve dependencies (transitive) and sort into pipeline order
    # ------------------------------------------------------------------ #
    selected = list(dict.fromkeys(selected))         # deduplicate, preserve order
    selected = _resolve_dependencies(selected)        # inject missing deps
    selected = _sort_pipeline(selected)               # canonical execution order

    # Build reasons list (backward-compat format)
    for cid in selected:
        if cid in reason_map:
            comp_name = REGISTRY[cid]["name"] if cid in REGISTRY else cid
            reasons.append({"component": comp_name, "reason": reason_map[cid]})

    # ------------------------------------------------------------------ #
    # 7. Annotate pipeline steps
    # ------------------------------------------------------------------ #
    pipeline_steps: list[dict[str, Any]] = []
    for order_idx, cid in enumerate(selected, start=1):
        comp = REGISTRY.get(cid, {})
        pipeline_steps.append({
            "order": order_idx,
            "id": cid,
            "name": comp.get("name", cid),
            "status": comp.get("status", "catalog"),
            "executable": comp.get("executable", False),
            "reason": reason_map.get(cid, ""),
            "dependencies": comp.get("dependencies", []),
        })

    # Human-readable step names (backward-compat "pipeline" field)
    step_names: list[str] = []
    _step_label: dict[str, str] = {
        "opencv": "Vision preprocessing with OpenCV",
        "video_processing": "Video frame extraction and processing",
        "image_classification": "Image classification",
        "object_detection": "Object detection and localisation",
        "ocr": "OCR text extraction",
        "pdf_parser": "PDF text extraction",
        "document_classification": "Document type classification",
        "csv": "Load and parse CSV data",
        "preprocessing": "Text cleaning and normalisation",
        "text_classifier": "Train task-specific text classifier",
        "embeddings": "Generate semantic embeddings",
        "vector_db": "Index and retrieve knowledge",
        "rag": "Retrieval-augmented generation",
        "llm": "LLM generation / reasoning",
        "summarization": "Text summarization",
        "batch_inference": "Batch inference",
        "fastapi": "Expose the pipeline through FastAPI",
    }
    for cid in selected:
        step_names.append(_step_label.get(cid, cid))

    # ------------------------------------------------------------------ #
    # 8. Execution readiness + warnings
    # ------------------------------------------------------------------ #
    # fastapi is always excluded from the runtime-readiness check — its
    # purpose is purely serving and it is always executable.
    runtime_components = [c for c in selected if c != "fastapi"]
    non_executable = [
        cid for cid in runtime_components
        if not REGISTRY.get(cid, {}).get("executable", False)
    ]
    execution_ready = len(non_executable) == 0

    warnings: list[str] = []
    for cid in non_executable:
        name = REGISTRY[cid]["name"] if cid in REGISTRY else cid
        warnings.append(
            f"{name} is not yet executable in the current AIForge MVP. "
            "It will appear in the generated architecture but cannot be invoked at runtime."
        )

    # ------------------------------------------------------------------ #
    # 9. Build output
    # ------------------------------------------------------------------ #
    full_components = [
        REGISTRY[k] | {"key": k, "id": k, "purpose": REGISTRY[k].get("description", "")}
        for k in selected if k in REGISTRY
    ]
    implemented = [k for k in selected if REGISTRY.get(k, {}).get("status") == "implemented"]
    # catalog_only = components that cannot actually be executed right now.
    # This includes both pure-catalog entries AND implemented components that
    # require a system binary (like OCR without Tesseract installed).
    catalog_only = [k for k in selected if not REGISTRY.get(k, {}).get("executable", False)]

    return {
        # Core identity
        "id": str(uuid.uuid4()),
        "specification": spec,

        # Structured input/task
        "input": {
            "detected_types": detected_inputs,
            "primary": primary_input,
        },
        "task": {
            "execution_ready": execution_ready,
            "non_executable_components": non_executable,
            "warnings": warnings,
        },

        # Components (backward-compat: flat list used by existing tests)
        "selected_components": selected,
        "components": full_components,

        # Pipeline (backward-compat string list + new structured list)
        "pipeline": step_names,
        "pipeline_steps": pipeline_steps,

        # Reasons (backward-compat)
        "reasons": reasons,

        # Split (backward-compat)
        "implemented_components": implemented,
        "catalog_components": catalog_only,

        # Top-level convenience copies of key fields
        "warnings": warnings,
        "execution_ready": execution_ready,

        "estimated_complexity": _complexity(selected),
        "mvp_note": (
            "The MVP executes the task-classification + API path. "
            "Catalog components are included in the generated architecture "
            "and can be activated in later phases."
        ),
    }


def _complexity(selected: list[str]) -> str:
    n = len(selected)
    if n <= 2:
        return "Low"
    if n <= 4:
        return "Medium"
    return "High"
