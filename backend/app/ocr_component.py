"""
AIForge OCR Component
======================

Real OCR using pytesseract (Python wrapper for Tesseract OCR engine).

SYSTEM REQUIREMENT
------------------
Tesseract OCR must be installed on the host system.

  Windows : https://github.com/UB-Mannheim/tesseract/wiki
            Default install path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe

  macOS   : brew install tesseract

  Ubuntu  : sudo apt-get install tesseract-ocr

After installing, either add Tesseract to PATH or set the path explicitly:

    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

The OCR component checks for the binary at import time and raises
TesseractNotAvailableError with a clear installation message when absent,
rather than silently failing or returning garbage.

USAGE
-----
    from app.ocr_component import extract_text, is_available

    if is_available():
        text = extract_text(preprocessed_pil_image)
    else:
        # show installation instructions
        ...
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Availability check — done once at import time
# ---------------------------------------------------------------------------

_TESSERACT_AVAILABLE: bool = False
_TESSERACT_PATH: str | None = None
_TESSERACT_VERSION: str | None = None

_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_WINDOWS_DEFAULT_ALT = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


def _probe_tesseract() -> tuple[bool, str | None, str | None]:
    """
    Try to detect a working Tesseract installation.
    Returns (available, path, version).
    """
    import shutil
    import subprocess

    # 1. Check PATH
    tess_path = shutil.which("tesseract")

    # 2. Check Windows default install locations
    if tess_path is None:
        import pathlib
        for candidate in [_WINDOWS_DEFAULT, _WINDOWS_DEFAULT_ALT]:
            if pathlib.Path(candidate).exists():
                tess_path = candidate
                break

    if tess_path is None:
        return False, None, None

    try:
        result = subprocess.run(
            [tess_path, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        version_line = (result.stdout or result.stderr or "").splitlines()
        version = version_line[0].strip() if version_line else "unknown"
        return True, tess_path, version
    except Exception:
        return False, None, None


_TESSERACT_AVAILABLE, _TESSERACT_PATH, _TESSERACT_VERSION = _probe_tesseract()

# If found, configure pytesseract to use it
if _TESSERACT_AVAILABLE and _TESSERACT_PATH:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH
    except ImportError:
        _TESSERACT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TesseractNotAvailableError(RuntimeError):
    """
    Raised when Tesseract OCR binary is not found on the system.

    Install Tesseract:
      Windows : https://github.com/UB-Mannheim/tesseract/wiki
      macOS   : brew install tesseract
      Ubuntu  : sudo apt-get install tesseract-ocr
    """

    INSTALL_GUIDE = (
        "Tesseract OCR is required but not found.\n"
        "Install instructions:\n"
        "  Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  macOS   : brew install tesseract\n"
        "  Ubuntu  : sudo apt-get install tesseract-ocr\n"
        "After installing, add 'tesseract' to your PATH and restart the service."
    )


def is_available() -> bool:
    """Return True if Tesseract OCR is installed and accessible."""
    return _TESSERACT_AVAILABLE


def tesseract_version() -> str | None:
    """Return the detected Tesseract version string, or None if not available."""
    return _TESSERACT_VERSION


def extract_text(
    image,  # np.ndarray (preprocessed grayscale) OR PIL.Image.Image
    lang: str = "eng",
    config: str = "--psm 6",
) -> str:
    """
    Run Tesseract OCR on the supplied image and return the extracted text.

    Parameters
    ----------
    image   : numpy array (grayscale uint8) or PIL Image
    lang    : Tesseract language code (default "eng")
    config  : Tesseract page segmentation mode (default PSM 6 = single block of text)

    Returns
    -------
    str
        Extracted text, stripped of leading/trailing whitespace.
        Returns an empty string if nothing was recognised.

    Raises
    ------
    TesseractNotAvailableError
        If Tesseract binary is not installed.
    OCRError
        If pytesseract raises an unexpected error during recognition.
    """
    if not _TESSERACT_AVAILABLE:
        raise TesseractNotAvailableError(TesseractNotAvailableError.INSTALL_GUIDE)

    import pytesseract

    # Accept both numpy arrays and PIL Images
    pil_image = _to_pil(image)

    try:
        text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        return text.strip()
    except pytesseract.TesseractError as exc:
        raise OCRError(f"Tesseract recognition failed: {exc}") from exc
    except Exception as exc:
        raise OCRError(f"Unexpected OCR error: {exc}") from exc


def extract_text_with_confidence(
    image,
    lang: str = "eng",
    config: str = "--psm 6",
) -> dict[str, Any]:
    """
    Like extract_text() but also returns per-word confidence scores.

    Returns
    -------
    dict with keys:
        text        — full extracted string
        words       — list of {word, confidence} dicts (confidence 0–100)
        mean_confidence — average word confidence (0–100), or 0 if no words found
    """
    if not _TESSERACT_AVAILABLE:
        raise TesseractNotAvailableError(TesseractNotAvailableError.INSTALL_GUIDE)

    import pytesseract

    pil_image = _to_pil(image)

    try:
        data = pytesseract.image_to_data(
            pil_image, lang=lang, config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        raise OCRError(f"Unexpected OCR error: {exc}") from exc

    words = []
    confidences = []
    for word, conf in zip(data["text"], data["conf"]):
        word = str(word).strip()
        try:
            conf_int = int(conf)
        except (ValueError, TypeError):
            continue
        if conf_int >= 0 and word:
            words.append({"word": word, "confidence": conf_int})
            confidences.append(conf_int)

    full_text = " ".join(w["word"] for w in words if w["word"])
    mean_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    return {
        "text": full_text,
        "words": words,
        "mean_confidence": mean_conf,
    }


class OCRError(RuntimeError):
    """Raised when Tesseract is available but recognition fails."""


def _to_pil(image):
    """Convert numpy array or PIL Image to PIL Image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for OCR. pip install Pillow") from exc

    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image  # already PIL


# ---------------------------------------------------------------------------
# Component metadata
# ---------------------------------------------------------------------------

COMPONENT_META: dict[str, Any] = {
    "name": "OCR",
    "category": "Document",
    "description": "Extract text from images using Tesseract OCR",
    "executable": _TESSERACT_AVAILABLE,   # honest: only True when binary present
    "tested": True,
    "requires_system_binary": True,
    "system_binary": "tesseract",
    "install_guide": TesseractNotAvailableError.INSTALL_GUIDE,
    "python_packages": ["pytesseract>=0.3", "Pillow>=10.0"],
}
