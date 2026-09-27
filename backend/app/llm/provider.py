"""
AIForge LLM Provider
=====================

Converts free-form natural-language requirements into a structured
RequirementSpec using either IBM watsonx.ai or a deterministic fallback.

ARCHITECTURE
------------
                    User requirement (str)
                            │
                  ┌─────────▼──────────┐
                  │   get_provider()   │
                  └──────┬──────┬──────┘
                         │      │
              credentials│      │no credentials
              available  │      │
                         ▼      ▼
                  ┌─────────┐ ┌─────────────────────┐
                  │Watsonx  │ │DeterministicProvider │
                  │Provider │ │(rule-based fallback) │
                  └────┬────┘ └──────────┬───────────┘
                       │                 │
                       └────────┬────────┘
                                │
                    RequirementSpec (Pydantic)
                                │
                         AIForge Planner

ENVIRONMENT VARIABLES
---------------------
  WATSONX_API_KEY        IBM Cloud IAM API key
  WATSONX_PROJECT_ID     watsonx.ai project ID
  WATSONX_URL            watsonx.ai endpoint URL
                         (default: https://us-south.ml.cloud.ibm.com)
  WATSONX_MODEL_ID       model to use
                         (default: ibm/granite-13b-instruct-v2)

If any required variable is missing or the LLM call fails, the system
automatically falls back to the deterministic provider.

SECURITY
--------
* Credentials are read from environment variables only — never hardcoded.
* The LLM receives only the user specification string; no internal data.
* The LLM output is parsed as JSON and validated by Pydantic before use.
* The LLM CANNOT execute code; it only produces a structured specification.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.llm.models import RequirementSpec, UnderstandingResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an AI requirements analyst for AIForge, an AI system builder platform.

Your task is to analyse a developer's natural-language requirement and return
a structured JSON object that describes the AI system they want to build.

OUTPUT FORMAT
-------------
Return ONLY a JSON object with the following keys (no markdown, no explanation):

{
  "goal":                   "<one-sentence description of what the system does>",
  "input_type":             "<one of: text | image | video | document | pdf | csv | audio | unknown>",
  "input_description":      "<brief description of the input data>",
  "output_type":            "<one of: prediction | summary | search_results | generated_text | structured_data | unknown>",
  "tasks":                  ["<ordered list of AI tasks, e.g. preprocess_image, ocr, classify, summarize>"],
  "requires_vision":        <true|false>,
  "requires_ocr":           <true|false>,
  "requires_classification":<true|false>,
  "requires_llm":           <true|false>,
  "requires_retrieval":     <true|false>,
  "requires_training":      <true|false>,
  "requires_api":           <true|false>,
  "constraints":            ["<list of constraints, e.g. local-only, no-GPU>"],
  "domain":                 "<application domain, e.g. legal, medical, e-commerce, general>",
  "language":               "<primary language code, e.g. en, hi, fr>",
  "notes":                  "<any relevant context that helps AIForge select components>"
}

RULES
-----
1. Return ONLY the JSON object. No text before or after.
2. If the requirement mentions images, photos, scans, or cameras → requires_vision=true.
3. If the requirement asks to extract text from images or documents → requires_ocr=true.
4. If the requirement asks to classify, categorise, or label → requires_classification=true.
5. If the requirement asks to summarise, generate, answer questions, or reason → requires_llm=true.
6. If the requirement asks to search, retrieve, or build a knowledge base → requires_retrieval=true.
7. If the requirement mentions CSV data or tabular training data → requires_training=true.
8. If the requirement mentions an API, endpoint, or REST → requires_api=true.
9. Be conservative: only set a flag to true if the requirement clearly asks for it.
10. tasks must be ordered from input to output.
"""

_USER_PROMPT_TEMPLATE = """\
Analyse this AI system requirement and return a structured JSON specification:

REQUIREMENT:
{requirement}

Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class LLMProvider:
    """
    Abstract base class for LLM providers.

    Subclasses implement generate() to call an LLM API.
    The understand() method is shared and handles JSON parsing + fallback.
    """

    name: str = "base"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Send a prompt to the LLM and return the raw string response.

        Raises
        ------
        NotImplementedError
            In the base class.
        LLMProviderError
            If the LLM call fails for any reason (timeout, auth, rate-limit).
        """
        raise NotImplementedError(f"{self.__class__.__name__}.generate() not implemented")

    def understand(self, requirement: str) -> UnderstandingResult:
        """
        Convert a natural-language requirement to a structured UnderstandingResult.

        Calls generate(), parses the JSON, validates with Pydantic.
        If parsing fails, raises LLMParseError.
        """
        prompt = _USER_PROMPT_TEMPLATE.format(requirement=requirement)
        raw = self.generate(prompt, system_prompt=_SYSTEM_PROMPT)
        spec, warnings = _parse_llm_json(raw, requirement)
        planner_spec = _spec_to_planner_string(spec)
        return UnderstandingResult(
            source="llm",
            requirement=spec,
            planner_spec=planner_spec,
            confidence=0.85,
            warnings=warnings,
            raw_llm_response=raw,
            model_used=getattr(self, "_model_id", None),
        )


class LLMProviderError(RuntimeError):
    """Raised when the LLM API call fails."""


class LLMParseError(ValueError):
    """Raised when the LLM response cannot be parsed as valid JSON."""


# ---------------------------------------------------------------------------
# Deterministic fallback provider
# ---------------------------------------------------------------------------

class DeterministicProvider(LLMProvider):
    """
    Rule-based fallback that derives a RequirementSpec directly from the
    user's specification string without calling any external API.

    This provider is always available and enables AIForge to operate without
    any LLM credentials.  It uses the same keyword logic as the Planner.
    """

    name = "deterministic"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        # The deterministic provider never calls an LLM
        raise NotImplementedError("DeterministicProvider does not call an LLM")

    def understand(self, requirement: str) -> UnderstandingResult:
        spec = _deterministic_parse(requirement)
        planner_spec = _spec_to_planner_string(spec)
        return UnderstandingResult(
            source="deterministic",
            requirement=spec,
            planner_spec=planner_spec,
            confidence=1.0,
            warnings=[
                "LLM provider not configured. Using deterministic rule-based understanding."
            ],
            raw_llm_response=None,
            model_used=None,
        )


# ---------------------------------------------------------------------------
# IBM watsonx.ai provider
# ---------------------------------------------------------------------------

class WatsonxProvider(LLMProvider):
    """
    IBM watsonx.ai provider using the ibm-watsonx-ai Python SDK.

    Credentials are loaded from environment variables — never hardcoded.

    Environment variables
    ---------------------
    WATSONX_API_KEY       IBM Cloud IAM API key (required)
    WATSONX_PROJECT_ID    watsonx.ai project ID (required)
    WATSONX_URL           endpoint URL (default: https://us-south.ml.cloud.ibm.com)
    WATSONX_MODEL_ID      model to use (default: ibm/granite-13b-instruct-v2)
    """

    name = "watsonx"
    _DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
    _DEFAULT_MODEL = "ibm/granite-13b-instruct-v2"

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        url: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("WATSONX_API_KEY", "")
        self._project_id = project_id or os.environ.get("WATSONX_PROJECT_ID", "")
        self._url = url or os.environ.get("WATSONX_URL", self._DEFAULT_URL)
        self._model_id = model_id or os.environ.get("WATSONX_MODEL_ID", self._DEFAULT_MODEL)

        if not self._api_key or not self._project_id:
            raise LLMProviderError(
                "WatsonxProvider requires WATSONX_API_KEY and WATSONX_PROJECT_ID "
                "to be set as environment variables."
            )

    @classmethod
    def is_configured(cls) -> bool:
        """Return True if the required environment variables are present."""
        return bool(
            os.environ.get("WATSONX_API_KEY")
            and os.environ.get("WATSONX_PROJECT_ID")
        )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Call the watsonx.ai text generation API.

        The full prompt is constructed by concatenating the system prompt
        and user prompt with clear delimiters.
        """
        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        except ImportError as exc:
            raise LLMProviderError(
                "ibm-watsonx-ai package is not installed. "
                "pip install ibm-watsonx-ai"
            ) from exc

        try:
            credentials = Credentials(
                url=self._url,
                api_key=self._api_key,
            )
            client = APIClient(credentials)

            full_prompt = (
                f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
                if system_prompt
                else prompt
            )

            model = ModelInference(
                model_id=self._model_id,
                api_client=client,
                project_id=self._project_id,
                params={
                    GenParams.MAX_NEW_TOKENS: 1024,
                    GenParams.MIN_NEW_TOKENS: 10,
                    GenParams.TEMPERATURE: 0.0,   # deterministic output
                    GenParams.TOP_P: 1.0,
                    GenParams.REPETITION_PENALTY: 1.05,
                },
            )

            response = model.generate_text(prompt=full_prompt)
            return response if isinstance(response, str) else str(response)

        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"watsonx.ai API call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_provider() -> LLMProvider:
    """
    Return the best available LLMProvider.

    Priority:
    1. WatsonxProvider — if WATSONX_API_KEY + WATSONX_PROJECT_ID are set
    2. DeterministicProvider — always available as fallback

    Never raises; always returns a usable provider.
    """
    if WatsonxProvider.is_configured():
        try:
            return WatsonxProvider()
        except Exception as exc:
            logger.warning("WatsonxProvider initialisation failed: %s — using fallback", exc)
    return DeterministicProvider()


def understand_requirement(requirement: str) -> UnderstandingResult:
    """
    Top-level function: convert a natural-language requirement to a
    structured UnderstandingResult.

    If the primary provider fails for any reason (network, auth, parse error),
    automatically falls back to DeterministicProvider.

    Never raises — always returns an UnderstandingResult.
    """
    provider = get_provider()

    # Try the configured provider
    if not isinstance(provider, DeterministicProvider):
        try:
            result = provider.understand(requirement)
            logger.info("Requirement understood via %s", provider.name)
            return result
        except Exception as exc:
            logger.warning(
                "LLM provider '%s' failed (%s) — falling back to deterministic",
                provider.name, exc,
            )

    # Fallback
    fallback = DeterministicProvider()
    result = fallback.understand(requirement)
    # If we reached here because the LLM failed, add that warning
    if not isinstance(provider, DeterministicProvider):
        result.warnings.insert(
            0,
            f"LLM provider '{provider.name}' was unavailable. "
            "Using deterministic rule-based understanding.",
        )
    return result


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_llm_json(
    raw: str, original_requirement: str
) -> tuple[RequirementSpec, list[str]]:
    """
    Parse and validate the LLM's raw JSON output.

    Returns
    -------
    (RequirementSpec, warnings)

    Raises
    ------
    LLMParseError
        If the response cannot be parsed or validated after repair attempts.
    """
    warnings: list[str] = []
    cleaned = raw.strip()

    # Strip markdown code fences if present (```json ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Extract the first JSON object if extra text is present
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise LLMParseError(
            f"LLM response does not contain a JSON object. Raw response:\n{raw[:300]}"
        )
    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"LLM JSON is malformed: {exc}\nRaw:\n{json_str[:300]}") from exc

    # Validate with Pydantic — unknown fields are ignored (extra="ignore")
    try:
        spec = RequirementSpec.model_validate(data)
    except Exception as exc:
        warnings.append(f"LLM response had validation issues; using partial data: {exc}")
        # Build a minimal valid spec from whatever we can salvage
        spec = RequirementSpec(
            goal=str(data.get("goal", original_requirement[:100])),
            input_type=str(data.get("input_type", "text")),
            output_type=str(data.get("output_type", "prediction")),
            notes=f"Partial parse from LLM response. Original: {original_requirement[:100]}",
        )

    return spec, warnings


# ---------------------------------------------------------------------------
# Deterministic parse (fallback)
# ---------------------------------------------------------------------------

def _deterministic_parse(requirement: str) -> RequirementSpec:
    """
    Derive a RequirementSpec from a requirement string using keyword matching.
    This mirrors the logic in planner.analyze_spec() but produces a Pydantic model.
    """
    s = requirement.lower()

    # Input type detection
    if any(t in s for t in ["image", "photo", "picture", "camera", "scan", "scanned"]):
        input_type = "image"
    elif any(t in s for t in ["video", "stream", "footage"]):
        input_type = "video"
    elif any(t in s for t in ["pdf", ".pdf"]):
        input_type = "pdf"
    elif any(t in s for t in ["csv", "spreadsheet", "tabular"]):
        input_type = "csv"
    else:
        input_type = "text"

    # Output type detection
    if any(t in s for t in ["summarize", "summary", "summarization"]):
        output_type = "summary"
    elif any(t in s for t in ["search", "retrieve", "find similar"]):
        output_type = "search_results"
    elif any(t in s for t in ["generate", "write", "compose"]):
        output_type = "generated_text"
    else:
        output_type = "prediction"

    # Capability flags
    requires_vision = any(t in s for t in [
        "image", "photo", "picture", "camera", "video", "visual", "scan", "scanned"
    ])
    requires_ocr = any(t in s for t in [
        "ocr", "extract text", "extracts text", "text extraction",
        "scan", "scanned", "document image",
    ])
    requires_classification = any(t in s for t in [
        "classify", "classifies", "classification", "categorize",
        "category", "categories", "label", "detect type",
    ])
    requires_llm = any(t in s for t in [
        "summarize", "summary", "generate", "chatbot",
        "question answering", "qa", "reason",
    ])
    requires_retrieval = any(t in s for t in [
        "search", "knowledge base", "rag", "retrieve", "similar", "semantic",
    ])
    requires_api = any(t in s for t in [
        "api", "rest", "endpoint", "expose", "serve", "service",
    ])

    # Tasks list (ordered)
    tasks: list[str] = []
    if requires_vision:
        tasks.append("preprocess_image")
    if requires_ocr:
        tasks.append("ocr")
    if requires_classification:
        tasks.append("classify")
    if requires_llm:
        tasks.append("summarize" if "summarize" in s or "summary" in s else "generate")
    if requires_retrieval:
        tasks.extend(["embed", "retrieve"])
    if requires_api:
        tasks.append("serve_api")

    if not tasks:
        tasks = ["classify", "serve_api"]

    # Domain detection
    domain_keywords = {
        "legal": ["legal", "law", "court", "petition", "contract", "affidavit"],
        "medical": ["medical", "health", "patient", "clinical", "diagnosis"],
        "finance": ["finance", "financial", "invoice", "payment", "bank"],
        "e-commerce": ["product", "customer", "order", "shop", "retail"],
    }
    domain = "general"
    for d, keywords in domain_keywords.items():
        if any(k in s for k in keywords):
            domain = d
            break

    # Language (simple heuristic)
    language = "en"
    if any(t in s for t in ["hindi", "हिंदी"]):
        language = "hi"
    elif any(t in s for t in ["french", "français"]):
        language = "fr"

    # Compose the goal
    goal = requirement.strip()
    if len(goal) > 150:
        goal = goal[:147] + "..."

    return RequirementSpec(
        goal=goal,
        input_type=input_type,
        input_description=f"Input data for the AI system: {input_type}",
        output_type=output_type,
        tasks=tasks,
        requires_vision=requires_vision,
        requires_ocr=requires_ocr,
        requires_classification=requires_classification,
        requires_llm=requires_llm,
        requires_retrieval=requires_retrieval,
        requires_training=True,
        requires_api=requires_api,
        domain=domain,
        language=language,
        notes="Parsed deterministically from specification keywords.",
    )


# ---------------------------------------------------------------------------
# Spec → planner string conversion
# ---------------------------------------------------------------------------

def _spec_to_planner_string(spec: RequirementSpec) -> str:
    """
    Convert a RequirementSpec into a natural-language string suitable
    for passing to analyze_spec() in the Planner.

    This bridges the LLM understanding layer and the deterministic Planner:
    the Planner receives a clean, unambiguous specification.
    """
    parts: list[str] = [spec.goal]

    if spec.requires_vision:
        parts.append("The system must process image or video input.")
    if spec.requires_ocr:
        parts.append("The system must extract text from images using OCR.")
    if spec.requires_classification:
        parts.append("The system must classify or categorize documents.")
    if spec.requires_llm:
        parts.append("The system must summarize or generate text using an LLM.")
    if spec.requires_retrieval:
        parts.append("The system must support semantic search and retrieval.")
    # Only append REST API sentence if requires_api is True and explicitly requested/present
    api_requested = (
        spec.requires_api
        and (
            "requires_api" in spec.model_fields_set
            or "serve_api" in spec.tasks
            or any(
                kw in spec.goal.lower()
                for kw in ["api", "rest", "endpoint", "expose", "serve", "service"]
            )
        )
    )
    if api_requested:
        parts.append("The system must expose a REST API.")

    return " ".join(parts)
