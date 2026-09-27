"""
AIForge Vision & OCR Component Tests — Task 5

Tests the vision.py (OpenCV preprocessing) and ocr_component.py modules.
Uses synthetic test images generated in-memory — no binary files committed.

OCR tests use unittest.mock to test the full logic path without requiring
the Tesseract binary to be installed.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from app.main import app
from app.ocr_component import (
    COMPONENT_META as OCR_META,
    OCRError,
    TesseractNotAvailableError,
    _to_pil,
    extract_text,
    extract_text_with_confidence,
    is_available,
    tesseract_version,
)
from app.vision import (
    COMPONENT_META as VISION_META,
    VisionPreprocessingError,
    binarize,
    denoise,
    load_image_bytes,
    preprocess,
    preprocessed_to_pil,
    to_grayscale,
)

client = TestClient(app)
DEMO_CSV = Path(__file__).resolve().parents[2] / "data" / "legal_demo.csv"


# ---------------------------------------------------------------------------
# Helpers: synthetic test images
# ---------------------------------------------------------------------------

def _make_white_png_bytes(width: int = 200, height: int = 100) -> bytes:
    """Create a small white PNG image as bytes."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_gray_png_bytes(width: int = 200, height: int = 100) -> bytes:
    img = np.ones((height, width), dtype=np.uint8) * 128
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_text_image_bytes(text: str = "Hello World", width: int = 300, height: int = 80) -> bytes:
    """Create a simple white image with black text drawn on it."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(
        img, text, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA,
    )
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_pil_image(width: int = 200, height: int = 100) -> PILImage.Image:
    return PILImage.fromarray(np.ones((height, width), dtype=np.uint8) * 200)


# ===========================================================================
# Vision component metadata
# ===========================================================================

class TestVisionMeta:
    def test_vision_is_executable(self):
        assert VISION_META["executable"] is True

    def test_vision_does_not_require_system_binary(self):
        assert VISION_META["requires_system_binary"] is False

    def test_vision_is_tested(self):
        assert VISION_META["tested"] is True

    def test_vision_python_packages_listed(self):
        assert len(VISION_META["python_packages"]) > 0
        assert any("opencv" in p.lower() for p in VISION_META["python_packages"])


# ===========================================================================
# OpenCV: image loading
# ===========================================================================

class TestLoadImageBytes:
    def test_load_valid_png(self):
        img = load_image_bytes(_make_white_png_bytes())
        assert img is not None
        assert img.shape[2] == 3  # BGR channels

    def test_load_valid_jpeg(self):
        png_bytes = _make_white_png_bytes()
        # Convert to JPEG bytes
        img = PILImage.open(io.BytesIO(png_bytes))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()
        result = load_image_bytes(jpeg_bytes)
        assert result is not None

    def test_load_invalid_bytes_raises(self):
        with pytest.raises(VisionPreprocessingError):
            load_image_bytes(b"not an image")

    def test_load_empty_bytes_raises(self):
        with pytest.raises(VisionPreprocessingError):
            load_image_bytes(b"")

    def test_loaded_image_has_correct_shape(self):
        img = load_image_bytes(_make_white_png_bytes(300, 200))
        assert img.shape == (200, 300, 3)


# ===========================================================================
# OpenCV: preprocessing steps
# ===========================================================================

class TestToGrayscale:
    def test_bgr_to_grayscale(self):
        bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        gray = to_grayscale(bgr)
        assert len(gray.shape) == 2  # single channel

    def test_already_grayscale_passthrough(self):
        gray_in = np.ones((100, 100), dtype=np.uint8) * 128
        gray_out = to_grayscale(gray_in)
        assert gray_out.shape == gray_in.shape

    def test_output_dtype_uint8(self):
        bgr = np.ones((50, 50, 3), dtype=np.uint8)
        gray = to_grayscale(bgr)
        assert gray.dtype == np.uint8


class TestDenoise:
    def test_denoised_same_shape(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = denoise(gray)
        assert result.shape == gray.shape

    def test_denoised_dtype_preserved(self):
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        result = denoise(gray)
        assert result.dtype == np.uint8


class TestBinarize:
    def test_output_binary_values(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = binarize(gray)
        unique = set(np.unique(binary).tolist())
        assert unique.issubset({0, 255})

    def test_output_same_shape(self):
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        binary = binarize(gray)
        assert binary.shape == gray.shape

    def test_white_image_is_all_white(self):
        """A uniformly bright image should produce an all-white binary output."""
        gray = np.ones((100, 100), dtype=np.uint8) * 250
        binary = binarize(gray)
        assert np.all(binary == 255)


# ===========================================================================
# OpenCV: full preprocess() pipeline
# ===========================================================================

class TestPreprocess:
    def test_preprocess_returns_ndarray(self):
        result = preprocess(_make_white_png_bytes())
        assert isinstance(result, np.ndarray)

    def test_preprocess_output_is_grayscale(self):
        result = preprocess(_make_white_png_bytes())
        assert len(result.shape) == 2  # grayscale

    def test_preprocess_output_is_binary(self):
        result = preprocess(_make_white_png_bytes())
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})

    def test_preprocess_preserves_dimensions(self):
        result = preprocess(_make_white_png_bytes(300, 200))
        assert result.shape == (200, 300)

    def test_preprocess_invalid_bytes_raises(self):
        with pytest.raises(VisionPreprocessingError):
            preprocess(b"garbage")

    def test_preprocess_without_deskew(self):
        result = preprocess(_make_white_png_bytes(), deskew_enabled=False)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2


# ===========================================================================
# OpenCV: PIL conversion
# ===========================================================================

class TestPreprocessedToPil:
    def test_returns_pil_image(self):
        result = preprocess(_make_white_png_bytes())
        pil = preprocessed_to_pil(result)
        assert isinstance(pil, PILImage.Image)

    def test_pil_image_has_correct_size(self):
        result = preprocess(_make_white_png_bytes(300, 200))
        pil = preprocessed_to_pil(result)
        assert pil.size == (300, 200)


# ===========================================================================
# OCR component metadata
# ===========================================================================

class TestOcrMeta:
    def test_ocr_requires_system_binary(self):
        assert OCR_META["requires_system_binary"] is True

    def test_ocr_system_binary_is_tesseract(self):
        assert OCR_META["system_binary"] == "tesseract"

    def test_ocr_install_guide_present(self):
        assert len(OCR_META["install_guide"]) > 0

    def test_ocr_is_tested(self):
        assert OCR_META["tested"] is True

    def test_ocr_executable_reflects_runtime(self):
        # executable must match is_available() at runtime
        assert OCR_META["executable"] == is_available()

    def test_ocr_python_packages_listed(self):
        assert any("pytesseract" in p.lower() for p in OCR_META["python_packages"])


# ===========================================================================
# OCR: availability check
# ===========================================================================

class TestOcrAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_tesseract_version_is_none_when_unavailable(self):
        if not is_available():
            assert tesseract_version() is None

    def test_tesseract_version_string_when_available(self):
        if is_available():
            assert isinstance(tesseract_version(), str)


# ===========================================================================
# OCR: extract_text with mocked Tesseract
# ===========================================================================

class TestExtractTextMocked:
    """
    Tests the extract_text() logic path using mock.patch so Tesseract
    binary is not required.  We mock pytesseract.image_to_string to
    return controlled values.
    """

    def _get_gray_image(self):
        return np.ones((100, 200), dtype=np.uint8) * 200

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch("pytesseract.image_to_string", return_value="  Hello World  ")
    def test_returns_stripped_text(self, mock_tess):
        result = extract_text(_make_pil_image())
        assert result == "Hello World"

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch("pytesseract.image_to_string", return_value="")
    def test_returns_empty_string_for_blank_image(self, mock_tess):
        result = extract_text(_make_pil_image())
        assert result == ""

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch("pytesseract.image_to_string", return_value="Contract agreement")
    def test_accepts_numpy_array(self, mock_tess):
        gray = np.ones((100, 200), dtype=np.uint8) * 200
        result = extract_text(gray)
        assert result == "Contract agreement"

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch("pytesseract.image_to_string", return_value="Test text")
    def test_accepts_pil_image(self, mock_tess):
        result = extract_text(_make_pil_image())
        assert result == "Test text"

    def test_raises_when_tesseract_not_available(self):
        with mock.patch("app.ocr_component._TESSERACT_AVAILABLE", False):
            with pytest.raises(TesseractNotAvailableError):
                extract_text(_make_pil_image())

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    def test_ocr_error_propagated(self):
        import pytesseract
        with mock.patch("pytesseract.image_to_string", side_effect=pytesseract.TesseractError(1, "fail")):
            with pytest.raises(OCRError):
                extract_text(_make_pil_image())

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch("pytesseract.image_to_string", return_value="legal petition text")
    def test_custom_lang_forwarded(self, mock_tess):
        extract_text(_make_pil_image(), lang="eng")
        _, kwargs = mock_tess.call_args
        assert kwargs.get("lang") == "eng" or mock_tess.call_args[0][1] == "eng"


# ===========================================================================
# OCR: extract_text_with_confidence with mocked Tesseract
# ===========================================================================

class TestExtractTextWithConfidenceMocked:
    def _mock_data(self):
        return {
            "text": ["Hello", "World", "", "  "],
            "conf": [95, 88, -1, 45],
        }

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    def test_returns_dict_with_required_keys(self):
        with mock.patch("pytesseract.image_to_data", return_value=self._mock_data()):
            result = extract_text_with_confidence(_make_pil_image())
        assert "text" in result
        assert "words" in result
        assert "mean_confidence" in result

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    def test_mean_confidence_is_float(self):
        with mock.patch("pytesseract.image_to_data", return_value=self._mock_data()):
            result = extract_text_with_confidence(_make_pil_image())
        assert isinstance(result["mean_confidence"], float)

    def test_raises_when_unavailable(self):
        with mock.patch("app.ocr_component._TESSERACT_AVAILABLE", False):
            with pytest.raises(TesseractNotAvailableError):
                extract_text_with_confidence(_make_pil_image())


# ===========================================================================
# _to_pil helper
# ===========================================================================

class TestToPil:
    def test_numpy_to_pil(self):
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        pil = _to_pil(arr)
        assert isinstance(pil, PILImage.Image)

    def test_pil_passthrough(self):
        pil_in = _make_pil_image()
        pil_out = _to_pil(pil_in)
        assert pil_out is pil_in


# ===========================================================================
# TesseractNotAvailableError
# ===========================================================================

class TestTesseractNotAvailableError:
    def test_has_install_guide(self):
        assert "Install" in TesseractNotAvailableError.INSTALL_GUIDE
        assert "tesseract" in TesseractNotAvailableError.INSTALL_GUIDE.lower()

    def test_is_runtime_error(self):
        err = TesseractNotAvailableError("test")
        assert isinstance(err, RuntimeError)

    def test_message_mentions_windows(self):
        assert "Windows" in TesseractNotAvailableError.INSTALL_GUIDE

    def test_message_mentions_linux(self):
        assert "Ubuntu" in TesseractNotAvailableError.INSTALL_GUIDE


# ===========================================================================
# /api/predict-image endpoint
# ===========================================================================

class TestPredictImageEndpoint:
    @pytest.fixture(scope="class")
    def trained_project_id(self):
        """Set up a trained project to use for image prediction tests."""
        from app.planner import analyze_spec
        from app.main import app as fastapi_app, PROJECTS_DIR
        test_client = TestClient(fastapi_app)

        plan = test_client.post(
            "/api/analyze",
            json={"specification": "Classify legal documents and expose a REST API."},
        ).json()
        generated = test_client.post(
            "/api/generate",
            json={
                "specification": plan["specification"],
                "project_name": "image-test",
                "plan": plan,
            },
        ).json()
        project_id = generated["project_id"]

        with DEMO_CSV.open("rb") as handle:
            test_client.post(
                f"/api/train?project_id={project_id}",
                files={"dataset": ("legal_demo.csv", handle, "text/csv")},
            )
        return project_id

    def test_predict_image_without_model_returns_404(self):
        img_bytes = _make_white_png_bytes()
        r = client.post(
            "/api/predict-image?project_id=nonexistent-id",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 404

    def test_predict_image_invalid_image_returns_400(self, trained_project_id):
        r = client.post(
            f"/api/predict-image?project_id={trained_project_id}",
            files={"file": ("bad.png", b"not an image", "image/png")},
        )
        assert r.status_code == 400

    def test_predict_image_without_tesseract_returns_503(self, trained_project_id):
        """When Tesseract is not installed, endpoint must return 503."""
        img_bytes = _make_white_png_bytes()
        with mock.patch("app.ocr_component._TESSERACT_AVAILABLE", False):
            r = client.post(
                f"/api/predict-image?project_id={trained_project_id}",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
        assert r.status_code == 503
        assert "Tesseract" in r.json()["detail"] or "tesseract" in r.json()["detail"].lower()

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch("pytesseract.image_to_string", return_value="The petitioner respectfully submits this petition.")
    def test_predict_image_with_mocked_ocr_returns_result(self, mock_tess, trained_project_id):
        """Full pipeline with mocked OCR — verifies the complete flow works."""
        img_bytes = _make_white_png_bytes()
        r = client.post(
            f"/api/predict-image?project_id={trained_project_id}",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "extracted_text" in body
        assert "prediction" in body
        assert "confidence" in body
        assert "top_predictions" in body

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch("pytesseract.image_to_string", return_value="")
    def test_predict_image_with_no_text_extracted(self, mock_tess, trained_project_id):
        """When OCR returns empty text, endpoint should return graceful response."""
        img_bytes = _make_white_png_bytes()
        r = client.post(
            f"/api/predict-image?project_id={trained_project_id}",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["extracted_text"] == ""
        assert body["prediction"] is None

    @mock.patch("app.ocr_component._TESSERACT_AVAILABLE", True)
    @mock.patch(
        "pytesseract.image_to_string",
        return_value="The petitioner submits this petition before the court.",
    )
    def test_predict_image_full_pipeline_prediction_is_petition(self, mock_tess, trained_project_id):
        """End-to-end image pipeline predicts 'Petition' from petition-like OCR text."""
        img_bytes = _make_text_image_bytes("Petition")
        r = client.post(
            f"/api/predict-image?project_id={trained_project_id}",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] == "Petition"
        assert body["confidence"] > 0.1
        assert len(body["top_predictions"]) == 3

    def test_predict_image_empty_file_returns_400(self, trained_project_id):
        r = client.post(
            f"/api/predict-image?project_id={trained_project_id}",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code == 400


# ===========================================================================
# Builder: generated project with opencv+ocr pipeline
# ===========================================================================

class TestBuilderWithVisionPipeline:
    def test_opencv_component_in_generated_pipeline(self):
        from app.builder import build_project
        result = build_project(
            "vision-pipeline-test",
            "Extract text from images and classify documents and expose an API.",
        )
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "OpencvComponent" in pipeline_src

    def test_ocr_component_in_generated_pipeline(self):
        from app.builder import build_project
        result = build_project(
            "ocr-pipeline-test",
            "Extract text from images and classify documents and expose an API.",
        )
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "OcrComponent" in pipeline_src

    def test_opencv_generates_real_code(self):
        from app.builder import build_project
        result = build_project(
            "opencv-code-test",
            "Extract text from images and classify documents.",
        )
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "cv2" in pipeline_src
        assert "adaptiveThreshold" in pipeline_src

    def test_ocr_component_mentions_tesseract(self):
        from app.builder import build_project
        result = build_project(
            "tesseract-mention-test",
            "Extract text from images and classify documents.",
        )
        pipeline_src = (result.project_dir / "agent" / "pipeline.py").read_text(encoding="utf-8")
        assert "pytesseract" in pipeline_src

    def test_requirements_includes_opencv_when_in_pipeline(self):
        from app.builder import build_project
        result = build_project(
            "req-opencv-test",
            "Extract text from images and classify documents.",
        )
        reqs = (result.project_dir / "requirements.txt").read_text(encoding="utf-8")
        assert "opencv-python-headless" in reqs

    def test_dockerfile_includes_sklearn_and_opencv(self):
        from app.builder import build_project
        result = build_project(
            "docker-vision-test",
            "Extract text from images, classify documents, expose API.",
        )
        docker = (result.project_dir / "Dockerfile").read_text(encoding="utf-8")
        assert "opencv-python-headless" in docker
        assert "scikit-learn" in docker
