"""
AIForge Vision Component
========================

Real OpenCV-based image preprocessing for the image → OCR → classifier pipeline.

This module is an executable component: it uses the real opencv-python-headless
library (no system binary required).

Preprocessing pipeline applied to every image:
  1. Decode to BGR array
  2. Grayscale conversion
  3. Gaussian denoising
  4. Adaptive thresholding (binarisation — maximises OCR accuracy)
  5. Optional deskew (skew correction up to ±5°)

The output is a preprocessed grayscale numpy array and a PIL Image, both
ready for pytesseract.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class VisionPreprocessingError(Exception):
    """Raised when image loading or preprocessing fails."""


def load_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes (JPEG, PNG, BMP, TIFF, …) into a BGR numpy array.

    Raises
    ------
    VisionPreprocessingError
        If the bytes cannot be decoded as a valid image.
    """
    if not image_bytes:
        raise VisionPreprocessingError("Image bytes are empty.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    if arr.size == 0:
        raise VisionPreprocessingError("Image bytes are empty.")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise VisionPreprocessingError(
            "Could not decode image bytes. "
            "Supported formats: JPEG, PNG, BMP, TIFF, WebP."
        )
    return img


def to_grayscale(bgr_image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if len(bgr_image.shape) == 2:
        return bgr_image  # already grayscale
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


def denoise(gray_image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(gray_image, (ksize, ksize), 0)


def binarize(gray_image: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding — produces a clean black-on-white binary image
    that maximises OCR character recognition accuracy.
    """
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )


def _compute_skew_angle(binary_image: np.ndarray) -> float:
    """
    Estimate document skew angle from the binary image using minAreaRect on
    white pixels.  Returns 0.0 on failure.
    """
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angle in (-90, 0]; convert to (-45, 45]
    if angle < -45:
        angle += 90
    return float(angle)


def deskew(binary_image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """
    Correct small document skew (up to ±max_angle degrees).
    Large rotations are left unchanged to avoid introducing artefacts.
    """
    angle = _compute_skew_angle(binary_image)
    if abs(angle) < 0.1 or abs(angle) > max_angle:
        return binary_image
    h, w = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess(image_bytes: bytes, deskew_enabled: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline:  bytes → BGR → gray → denoise → binarize → deskew.

    Returns
    -------
    np.ndarray
        Preprocessed single-channel (grayscale) uint8 image, ready for OCR.
    """
    bgr = load_image_bytes(image_bytes)
    gray = to_grayscale(bgr)
    denoised = denoise(gray)
    binary = binarize(denoised)
    if deskew_enabled:
        binary = deskew(binary)
    return binary


def preprocessed_to_pil(preprocessed: np.ndarray):
    """
    Convert the preprocessed numpy array to a PIL Image.
    Requires Pillow (already a project dependency).
    """
    from PIL import Image  # local import — Pillow is optional for callers that don't need it
    return Image.fromarray(preprocessed)


# ---------------------------------------------------------------------------
# Component metadata (mirrors the planner REGISTRY shape)
# ---------------------------------------------------------------------------

COMPONENT_META: dict[str, Any] = {
    "name": "OpenCV",
    "category": "Vision",
    "description": "Image preprocessing: grayscale, denoising, binarisation, deskew",
    "executable": True,
    "tested": True,
    "requires_system_binary": False,
    "python_packages": ["opencv-python-headless>=4.9", "Pillow>=10.0"],
}
