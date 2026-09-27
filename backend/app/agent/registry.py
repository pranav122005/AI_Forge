"""
AIForge Component Execution Interface & Registry
================================================

Defines the standard interface for executable AI components and provides
adapters for OpenCV, OCR, ML text classification, FastAPI serving, and LLM reasoning.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from app.agent.context import AgentContext
from app.agent.errors import ComponentNotExecutableError, ComponentNotFoundError


class BaseComponent(ABC):
    """
    Common execution interface for all AIForge pipeline components.
    """

    name: str
    version: str = "0.1.0"
    executable: bool = True

    def validate_input(self, input_data: Any, context: AgentContext) -> bool:
        """Validate input data before execution. Returns True if valid."""
        return True

    @abstractmethod
    def execute(self, input_data: Any, context: AgentContext) -> Any:
        """Execute component logic using input_data and context."""
        pass

    def validate_output(self, output_data: Any, context: AgentContext) -> bool:
        """Validate component output after execution. Returns True if valid."""
        return output_data is not None


class OpenCVComponent(BaseComponent):
    """
    OpenCV Image Preprocessing Component.
    """

    name = "opencv"
    executable = True

    def validate_input(self, input_data: Any, context: AgentContext) -> bool:
        if input_data is None and "image_bytes" not in context.artifacts:
            return False
        return True

    def execute(self, input_data: Any, context: AgentContext) -> Any:
        from app.vision import preprocess

        raw_bytes = None
        if isinstance(input_data, bytes):
            raw_bytes = input_data
        elif isinstance(input_data, dict) and "image_bytes" in input_data:
            raw_bytes = input_data["image_bytes"]
        elif "image_bytes" in context.artifacts:
            raw_bytes = context.artifacts["image_bytes"]

        if not raw_bytes:
            raise ValueError("OpenCVComponent requires raw image bytes.")

        processed_np = preprocess(raw_bytes)
        context.set_artifact("preprocessed_image", processed_np)
        return processed_np


class OCRComponent(BaseComponent):
    """
    Tesseract OCR Text Extraction Component.
    """

    name = "ocr"

    @property
    def executable(self) -> bool:
        try:
            from app.ocr_component import is_available
            return is_available()
        except Exception:
            return False

    def execute(self, input_data: Any, context: AgentContext) -> str:
        from app.ocr_component import is_available, extract_text, TesseractNotAvailableError

        if not is_available():
            raise ComponentNotExecutableError(
                "Tesseract OCR is not installed or available on this system."
            )

        pil_img = None
        # Check input_data or context artifacts
        if isinstance(input_data, Image.Image):
            pil_img = input_data
        elif "preprocessed_image" in context.artifacts:
            from app.vision import preprocessed_to_pil
            pil_img = preprocessed_to_pil(context.artifacts["preprocessed_image"])
        elif isinstance(input_data, bytes):
            pil_img = Image.open(io.BytesIO(input_data)).convert("RGB")
        elif "image_bytes" in context.artifacts:
            pil_img = Image.open(io.BytesIO(context.artifacts["image_bytes"])).convert("RGB")

        if pil_img is None:
            raise ValueError("OCRComponent requires an image (PIL Image, OpenCV numpy array, or image bytes).")

        extracted_text = extract_text(pil_img)
        context.set_artifact("extracted_text", extracted_text)
        return extracted_text


class TextClassifierComponent(BaseComponent):
    """
    ML Text Classification Component.
    """

    name = "text_classifier"
    executable = True

    def execute(self, input_data: Any, context: AgentContext) -> dict[str, Any]:
        from app.ml.persistence import load_model, model_exists

        project_id = context.project_id
        if not model_exists(project_id):
            raise ComponentNotExecutableError(
                f"No trained model found for project '{project_id}'. Train the model first."
            )

        text_input = ""
        if isinstance(input_data, str):
            text_input = input_data
        elif isinstance(input_data, dict) and "text" in input_data:
            text_input = str(input_data["text"])
        elif "extracted_text" in context.artifacts:
            text_input = str(context.artifacts["extracted_text"])

        if not text_input:
            # Handle empty extracted text gracefully
            result = {
                "prediction": "",
                "confidence": 0.0,
                "top_predictions": [],
                "note": "Empty text input provided to classifier",
            }
            context.set_artifact("prediction", result)
            return result

        pipeline = load_model(project_id)
        prediction = str(pipeline.predict([text_input])[0])
        probabilities = pipeline.predict_proba([text_input])[0]
        classes = pipeline.classes_.tolist()
        confidence = round(float(max(probabilities)), 4)
        ranked = sorted(
            [
                {"label": str(c), "confidence": round(float(p), 4)}
                for c, p in zip(classes, probabilities)
            ],
            key=lambda item: item["confidence"],
            reverse=True,
        )[:3]

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "top_predictions": ranked,
        }
        context.set_artifact("prediction", result)
        return result


class FastAPIComponent(BaseComponent):
    """
    FastAPI Serving Layer Adapter.
    """

    name = "fastapi"
    executable = True

    def execute(self, input_data: Any, context: AgentContext) -> dict[str, Any]:
        # Formats the final output dict for REST endpoints
        if "prediction" in context.artifacts:
            res = dict(context.artifacts["prediction"])
            res["project_id"] = context.project_id
            if "extracted_text" in context.artifacts:
                res["extracted_text"] = context.artifacts["extracted_text"]
            return res
        if isinstance(input_data, dict):
            return input_data
        return {"result": input_data, "project_id": context.project_id}


class LLMComponent(BaseComponent):
    """
    LLM Reasoning Component Adapter.
    """

    name = "llm"
    executable = True

    def execute(self, input_data: Any, context: AgentContext) -> dict[str, Any]:
        from app.llm.provider import understand_requirement

        text_input = str(input_data) if isinstance(input_data, str) else str(context.input_data)
        result = understand_requirement(text_input)
        res_dict = result.model_dump()
        context.set_artifact("llm_understanding", res_dict)
        return res_dict


class CatalogComponent(BaseComponent):
    """
    Adapter for catalog-only (non-executable) components.
    """

    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name
        self.executable = False

    def execute(self, input_data: Any, context: AgentContext) -> Any:
        raise ComponentNotExecutableError(
            f"Component '{self.name}' ({self.key}) is catalog-only and not executable in the current AIForge MVP."
        )


class ComponentRegistry:
    """
    Registry of allowlisted executable and catalog component adapters.
    """

    def __init__(self) -> None:
        self._components: dict[str, BaseComponent] = {
            "opencv": OpenCVComponent(),
            "ocr": OCRComponent(),
            "text_classifier": TextClassifierComponent(),
            "fastapi": FastAPIComponent(),
            "llm": LLMComponent(),
        }

    def register(self, key: str, component: BaseComponent) -> None:
        """Register a component adapter."""
        self._components[key] = component

    def get(self, key: str) -> BaseComponent:
        """Get component by key. Returns CatalogComponent if non-executable."""
        if key in self._components:
            return self._components[key]

        # Check planner registry to see if it's a known catalog component
        from app.planner import REGISTRY as PLANNER_REGISTRY
        if key in PLANNER_REGISTRY:
            info = PLANNER_REGISTRY[key]
            return CatalogComponent(key, info.get("name", key))

        raise ComponentNotFoundError(f"Component '{key}' not found in registry.")

    def is_executable(self, key: str) -> bool:
        """Check if component is executable."""
        try:
            comp = self.get(key)
            return getattr(comp, "executable", False)
        except ComponentNotFoundError:
            return False


# Default global component registry instance
default_registry = ComponentRegistry()
