from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.gui_element import GUIElement
from src.perception.image_preprocess import ImageProcessor
from src.perception.paddle_ocr import PaddleOCREngine
from src.perception.perception_pipeline import (
    PerceptionPipeline,
    PerceptionResult,
)
from src.perception.screen_capture import ScreenCapture
from src.perception.ui_detector import UIDetector


# ---------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path('D:/GUIAgent_project/screenshots')

FULL_SCREEN_SOURCE_PATH = (
    OUTPUT_DIR
    / "pp_full_screen_source.png"
)

FULL_SCREEN_ORIGINAL_PATH = (
    OUTPUT_DIR
    / "pp_full_screen_original.png"
)

FULL_SCREEN_PROCESSED_PATH = (
    OUTPUT_DIR
    / "pp_full_screen_processed.png"
)

FULL_SCREEN_VISUALIZATION_PATH = (
    OUTPUT_DIR
    / "pp_full_screen_visualization.png"
)

REGION_ORIGINAL_PATH = (
    OUTPUT_DIR
    / "pp_region_original.png"
)

REGION_PROCESSED_PATH = (
    OUTPUT_DIR
    / "pp_region_processed.png"
)

REGION_VISUALIZATION_PATH = (
    OUTPUT_DIR
    / "pp_region_visualization.png"
)

PREPROCESS_VISUALIZATION_PATH = (
    OUTPUT_DIR
    / "pp_preprocess_visualization.png"
)


SEARCH_QUERY = "Agent"

OCR_LANGUAGE = "ch"
OCR_CONFIDENCE_THRESHOLD = 0.50

# 指定区域格式：
# (left, top, width, height)
TEST_REGION = (60, 10, 890, 1000)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def output_dir() -> Path:
    """
    Create the test output directory.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return OUTPUT_DIR


@pytest.fixture(scope="session")
def screen_capture() -> ScreenCapture:
    """
    Create a shared ScreenCapture object.
    """

    return ScreenCapture()


@pytest.fixture(scope="session")
def image_processor() -> ImageProcessor:
    """
    Create a shared ImageProcessor object.
    """

    return ImageProcessor()


@pytest.fixture(scope="session")
def ocr_engine() -> PaddleOCREngine:
    """
    Create a shared CPU PaddleOCR engine.
    """

    return PaddleOCREngine(
        lang=OCR_LANGUAGE,
        confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_gpu=False,
        sort_results=True,
    )


@pytest.fixture(scope="session")
def ui_detector() -> UIDetector:
    """
    Create a shared OpenCV UI detector.
    """

    return UIDetector(
        min_width=12,
        min_height=12,
        min_area=120,
        rectangularity_threshold=0.45,
        duplicate_iou_threshold=0.75,
        use_adaptive_threshold=True,
    )


@pytest.fixture(scope="session")
def pipeline(
    screen_capture: ScreenCapture,
    image_processor: ImageProcessor,
    ocr_engine: PaddleOCREngine,
    ui_detector: UIDetector,
) -> PerceptionPipeline:
    """
    Create the complete perception pipeline.
    """

    return PerceptionPipeline(
        screen_capture=screen_capture,
        image_processor=image_processor,
        ocr_engine=ocr_engine,
        ui_detector=ui_detector,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
        include_unmatched_ocr=True,
    )


@pytest.fixture(scope="session")
def full_screen_image(
    screen_capture: ScreenCapture,
    output_dir: Path,
) -> np.ndarray:
    """
    Capture the desktop once and reuse it in tests.
    """

    image = screen_capture.capture_screen()

    assert isinstance(image, np.ndarray)
    assert image.size > 0

    success = cv2.imwrite(
        str(FULL_SCREEN_SOURCE_PATH),
        image,
    )

    assert success is True
    assert FULL_SCREEN_SOURCE_PATH.exists()

    logger.info(
        "Source screenshot saved to %s",
        FULL_SCREEN_SOURCE_PATH,
    )

    return image


@pytest.fixture(scope="session")
def full_result(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> PerceptionResult:
    """
    Run the complete pipeline once on the full screenshot.
    """

    result = pipeline.process_image(
        image=full_screen_image,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    logger.info(
        "Full pipeline result: %s",
        result.summary(),
    )

    return result


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _print_separator(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def _print_elements(
    title: str,
    elements: Sequence[GUIElement],
) -> None:
    """
    Print GUI elements in a readable format.
    """

    _print_separator(title)

    if not elements:
        print("No elements detected.")
        return

    for index, element in enumerate(
        elements,
        start=1,
    ):
        print(
            f"[{index:03d}] "
            f"type={element.element_type!r}, "
            f"text={element.text!r}, "
            f"confidence={element.confidence:.4f}, "
            f"bbox={element.bbox}, "
            f"center={element.center}"
        )


def _validate_gui_element(
    element: GUIElement,
) -> None:
    """
    Validate one GUIElement.
    """

    assert isinstance(element, GUIElement)

    assert isinstance(element.text, str)

    assert isinstance(element.bbox, tuple)
    assert len(element.bbox) == 4

    x1, y1, x2, y2 = element.bbox

    assert all(
        isinstance(value, int)
        for value in (
            x1,
            y1,
            x2,
            y2,
        )
    )

    assert x1 >= 0
    assert y1 >= 0
    assert x2 > x1
    assert y2 > y1

    assert isinstance(
        element.confidence,
        float,
    )

    assert 0.0 <= element.confidence <= 1.0

    assert isinstance(
        element.element_type,
        str,
    )

    assert element.element_type.strip() != ""

    if element.center is not None:
        assert isinstance(element.center, tuple)
        assert len(element.center) == 2

        center_x, center_y = element.center

        assert isinstance(center_x, int)
        assert isinstance(center_y, int)

        assert x1 <= center_x <= x2
        assert y1 <= center_y <= y2


def _validate_result(
    result: PerceptionResult,
) -> None:
    """
    Validate a complete PerceptionResult.
    """

    assert isinstance(result, PerceptionResult)

    assert isinstance(
        result.original_image,
        np.ndarray,
    )

    assert isinstance(
        result.processed_image,
        np.ndarray,
    )

    assert result.original_image.size > 0
    assert result.processed_image.size > 0

    assert isinstance(result.ocr_elements, list)
    assert isinstance(result.ui_elements, list)
    assert isinstance(result.merged_elements, list)

    assert isinstance(result.elapsed_time, float)
    assert result.elapsed_time >= 0.0

    assert isinstance(result.metadata, dict)

    for element in result.ocr_elements:
        _validate_gui_element(element)

    for element in result.ui_elements:
        _validate_gui_element(element)

    for element in result.merged_elements:
        _validate_gui_element(element)


def _get_valid_region(
    image: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Adjust TEST_REGION when the screen is smaller.
    """

    image_height, image_width = image.shape[:2]

    left, top, width, height = TEST_REGION

    if left >= image_width:
        raise ValueError(
            f"Region left coordinate {left} exceeds "
            f"image width {image_width}."
        )

    if top >= image_height:
        raise ValueError(
            f"Region top coordinate {top} exceeds "
            f"image height {image_height}."
        )

    valid_width = min(
        width,
        image_width - left,
    )

    valid_height = min(
        height,
        image_height - top,
    )

    if valid_width <= 0 or valid_height <= 0:
        raise ValueError(
            "Calculated test region is invalid."
        )

    return (
        left,
        top,
        valid_width,
        valid_height,
    )



# ---------------------------------------------------------------------
# Basic component tests
# ---------------------------------------------------------------------

def test_pipeline_initialization(
    pipeline: PerceptionPipeline,
) -> None:
    """
    Test pipeline dependency initialization.
    """

    _print_separator(
        "TEST: PIPELINE INITIALIZATION"
    )

    assert isinstance(
        pipeline.screen_capture,
        ScreenCapture,
    )

    assert isinstance(
        pipeline.image_processor,
        ImageProcessor,
    )

    assert isinstance(
        pipeline.ocr_engine,
        PaddleOCREngine,
    )

    assert isinstance(
        pipeline.ui_detector,
        UIDetector,
    )

    print(repr(pipeline))

    logger.info(
        "Pipeline initialization test passed."
    )


def test_full_screen_image(
    full_screen_image: np.ndarray,
) -> None:
    """
    Test source screenshot validity.
    """

    _print_separator(
        "TEST: FULL SCREEN IMAGE"
    )

    assert isinstance(
        full_screen_image,
        np.ndarray,
    )

    assert full_screen_image.size > 0
    assert full_screen_image.ndim in (2, 3)

    height, width = full_screen_image.shape[:2]

    print(
        f"Image shape: "
        f"{full_screen_image.shape}"
    )

    print(f"Width: {width}")
    print(f"Height: {height}")
    print(
        f"Saved to: "
        f"{FULL_SCREEN_SOURCE_PATH}"
    )

    logger.info(
        "Full-screen image test passed."
    )



# ---------------------------------------------------------------------
# Complete pipeline tests
# ---------------------------------------------------------------------

def test_complete_perception_pipeline(
    full_result: PerceptionResult,
) -> None:
    """
    Test the full pipeline:
    screenshot/image -> OCR -> UI detection -> merge.
    """

    _print_separator(
        "TEST: COMPLETE PERCEPTION PIPELINE"
    )

    _validate_result(full_result)

    assert (
        full_result.metadata[
            "ocr_enabled"
        ]
        is True
    )

    assert (
        full_result.metadata[
            "ui_detection_enabled"
        ]
        is True
    )

    assert (
        full_result.metadata[
            "merge_enabled"
        ]
        is True
    )

    assert full_result.element_count == len(
        full_result.merged_elements
    )

    assert full_result.text_count == len(
        full_result.ocr_elements
    )

    assert full_result.ui_count == len(
        full_result.ui_elements
    )

    _print_elements(
        title="OCR ELEMENTS",
        elements=full_result.ocr_elements,
    )

    _print_elements(
        title="UI ELEMENTS",
        elements=full_result.ui_elements,
    )

    _print_elements(
        title="MERGED ELEMENTS",
        elements=full_result.merged_elements,
    )

    print("Pipeline summary:")

    for key, value in (
        full_result
        .summary()
        .items()
    ):
        print(f"  {key}: {value}")

    logger.info(
        "Complete perception pipeline test passed."
    )


def test_pipeline_result_summary(
    full_result: PerceptionResult,
) -> None:
    """
    Test PerceptionResult.summary().
    """

    _print_separator(
        "TEST: RESULT SUMMARY"
    )

    summary = full_result.summary()

    assert isinstance(summary, dict)

    required_keys = {
        "ocr_element_count",
        "ui_element_count",
        "merged_element_count",
        "element_type_counts",
        "capture_region",
        "elapsed_time_seconds",
    }

    assert required_keys.issubset(
        summary.keys()
    )

    assert (
        summary["ocr_element_count"]
        == len(full_result.ocr_elements)
    )

    assert (
        summary["ui_element_count"]
        == len(full_result.ui_elements)
    )

    assert (
        summary["merged_element_count"]
        == len(full_result.merged_elements)
    )

    print(summary)

    logger.info(
        "Result-summary test passed."
    )


def test_get_result_texts(
    full_result: PerceptionResult,
) -> None:
    """
    Test text extraction from merged results.
    """

    _print_separator(
        "TEST: GET RESULT TEXTS"
    )

    texts = full_result.get_texts()

    assert isinstance(texts, list)

    assert all(
        isinstance(text, str)
        for text in texts
    )

    assert all(
        text.strip()
        for text in texts
    )

    print(
        f"Text count: {len(texts)}"
    )

    for index, text in enumerate(
        texts,
        start=1,
    ):
        print(f"[{index:03d}] {text}")

    logger.info(
        "Result text-extraction test passed."
    )


# ---------------------------------------------------------------------
# OCR-only and UI-only tests
# ---------------------------------------------------------------------

def test_ocr_only_pipeline(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test OCR without UI detection.
    """

    _print_separator(
        "TEST: OCR-ONLY PIPELINE"
    )

    result = pipeline.process_image(
        image=full_screen_image,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=False,
        merge_results=True,
    )

    _validate_result(result)

    assert len(result.ui_elements) == 0

    assert (
        result.merged_elements
        == result.ocr_elements
    )

    _print_elements(
        title="OCR-ONLY RESULTS",
        elements=result.merged_elements,
    )

    logger.info(
        "OCR-only pipeline test passed."
    )


def test_ui_only_pipeline(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test UI detection without OCR.
    """

    _print_separator(
        "TEST: UI-ONLY PIPELINE"
    )

    result = pipeline.process_image(
        image=full_screen_image,
        enable_preprocessing=False,
        enable_ocr=False,
        enable_ui_detection=True,
        merge_results=True,
    )

    _validate_result(result)

    assert len(result.ocr_elements) == 0

    assert (
        result.merged_elements
        == result.ui_elements
    )

    _print_elements(
        title="UI-ONLY RESULTS",
        elements=result.merged_elements,
    )

    logger.info(
        "UI-only pipeline test passed."
    )


def test_all_perception_disabled(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test pipeline when OCR and UI detection are disabled.
    """

    _print_separator(
        "TEST: ALL DETECTION DISABLED"
    )

    result = pipeline.process_image(
        image=full_screen_image,
        enable_preprocessing=False,
        enable_ocr=False,
        enable_ui_detection=False,
        merge_results=True,
    )

    _validate_result(result)

    assert result.ocr_elements == []
    assert result.ui_elements == []
    assert result.merged_elements == []

    print(
        "Pipeline correctly returned "
        "an empty element list."
    )

    logger.info(
        "All-detection-disabled test passed."
    )


# ---------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------

def test_pipeline_preprocessing(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test preprocessing inside the perception pipeline.

    OCR and UI detection are disabled here to isolate preprocessing.
    """

    _print_separator(
        "TEST: PIPELINE PREPROCESSING"
    )

    result = pipeline.process_image(
        image=full_screen_image,
        enable_preprocessing=True,
        enable_ocr=False,
        enable_ui_detection=False,
        preprocess_options={
            "resize_width": None,
            "resize_height": None,
            "use_gray": True,
            "use_gaussian": True,
            "use_median": False,
            "use_binary": False,
            "use_adaptive": False,
            "use_clahe": True,
            "use_sharpen": True,
        },
    )

    _validate_result(result)

    assert (
        result.metadata[
            "preprocessing_enabled"
        ]
        is True
    )

    assert result.processed_image.ndim == 2

    success = cv2.imwrite(
        str(PREPROCESS_VISUALIZATION_PATH),
        result.processed_image,
    )

    assert success is True

    assert (
        PREPROCESS_VISUALIZATION_PATH
        .exists()
    )

    print(
        "Original shape: "
        f"{result.original_image.shape}"
    )

    print(
        "Processed shape: "
        f"{result.processed_image.shape}"
    )

    print(
        "Processed image saved to: "
        f"{PREPROCESS_VISUALIZATION_PATH}"
    )

    logger.info(
        "Pipeline preprocessing test passed."
    )


# ---------------------------------------------------------------------
# Region tests
# ---------------------------------------------------------------------

def test_region_perception(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test perception on the selected region.
    """

    _print_separator(
        "TEST: REGION PERCEPTION"
    )

    region = _get_valid_region(
        full_screen_image
    )

    left, top, width, height = region

    result = pipeline.process_image(
        image=full_screen_image,
        region=region,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    _validate_result(result)

    assert result.capture_region == region

    for element in result.merged_elements:
        x1, y1, x2, y2 = element.bbox

        assert x1 >= left
        assert y1 >= top
        assert x2 <= left + width
        assert y2 <= top + height

    _print_elements(
        title="REGION MERGED ELEMENTS",
        elements=result.merged_elements,
    )

    print(
        "Region: "
        f"left={left}, top={top}, "
        f"width={width}, height={height}"
    )

    logger.info(
        "Region perception test passed."
    )


def test_capture_and_run_region(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test direct region capture from ScreenCapture.
    """

    _print_separator(
        "TEST: CAPTURE AND RUN REGION"
    )

    region = _get_valid_region(
        full_screen_image
    )

    result = pipeline.capture_and_run(
        region=region,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    _validate_result(result)

    assert result.capture_region == region

    print(result.summary())

    logger.info(
        "Capture-and-run region test passed."
    )


# ---------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------

def test_find_text(
    pipeline: PerceptionPipeline,
    full_result: PerceptionResult,
) -> None:
    """
    Test text lookup in merged perception results.
    """

    _print_separator(
        "TEST: FIND TEXT"
    )

    matches = pipeline.find_text(
        result=full_result,
        query=SEARCH_QUERY,
        exact_match=False,
        case_sensitive=False,
    )

    assert isinstance(matches, list)

    print(
        f"Search query: "
        f"{SEARCH_QUERY!r}"
    )

    print(
        f"Match count: "
        f"{len(matches)}"
    )

    _print_elements(
        title="TEXT SEARCH RESULTS",
        elements=matches,
    )

    if not matches:
        pytest.skip(
            f"Text {SEARCH_QUERY!r} was not "
            "detected on the current screen. "
            "Change SEARCH_QUERY to visible text."
        )

    expected = SEARCH_QUERY.lower()

    for element in matches:
        assert expected in element.text.lower()

    logger.info(
        "Text-search test passed."
    )


def test_find_best_text_match(
    pipeline: PerceptionPipeline,
    full_result: PerceptionResult,
) -> None:
    """
    Test best-confidence text lookup.
    """

    _print_separator(
        "TEST: FIND BEST TEXT MATCH"
    )

    text_elements = [
        element
        for element
        in full_result.merged_elements
        if element.text.strip()
    ]

    if not text_elements:
        pytest.skip(
            "No detected text is available."
        )

    query = text_elements[0].text

    best_match = pipeline.find_best_text_match(
        result=full_result,
        query=query,
        exact_match=True,
        case_sensitive=True,
    )

    assert best_match is not None
    assert best_match.text == query

    matches = pipeline.find_text(
        result=full_result,
        query=query,
        exact_match=True,
        case_sensitive=True,
    )

    expected_confidence = max(
        element.confidence
        for element in matches
    )

    assert (
        best_match.confidence
        == expected_confidence
    )

    print(
        f"Query: {query!r}"
    )

    print(
        "Best match: "
        f"{best_match}"
    )

    logger.info(
        "Best-text-match test passed."
    )


def test_find_by_type(
    pipeline: PerceptionPipeline,
    full_result: PerceptionResult,
) -> None:
    """
    Test type-based element filtering.
    """

    _print_separator(
        "TEST: FIND BY TYPE"
    )

    available_types = sorted(
        {
            element.element_type
            for element
            in full_result.merged_elements
        }
    )

    if not available_types:
        pytest.skip(
            "No element types are available."
        )

    selected_type = available_types[0]

    matches = pipeline.find_by_type(
        result=full_result,
        element_type=selected_type,
    )

    assert isinstance(matches, list)
    assert len(matches) > 0

    assert all(
        element.element_type
        == selected_type
        for element in matches
    )

    print(
        f"Available types: "
        f"{available_types}"
    )

    print(
        f"Selected type: "
        f"{selected_type}"
    )

    print(
        f"Match count: "
        f"{len(matches)}"
    )

    logger.info(
        "Type-search test passed."
    )


def test_perception_result_get_elements_by_type(
    full_result: PerceptionResult,
) -> None:
    """
    Test PerceptionResult.get_elements_by_type().
    """

    _print_separator(
        "TEST: RESULT GET ELEMENTS BY TYPE"
    )

    if not full_result.merged_elements:
        pytest.skip(
            "No merged elements available."
        )

    selected_type = (
        full_result
        .merged_elements[0]
        .element_type
    )

    matches = full_result.get_elements_by_type(
        selected_type
    )

    assert len(matches) > 0

    assert all(
        element.element_type.lower()
        == selected_type.lower()
        for element in matches
    )

    print(
        f"Selected type: {selected_type}"
    )

    print(
        f"Match count: {len(matches)}"
    )

    logger.info(
        "PerceptionResult type-filter test passed."
    )



# ---------------------------------------------------------------------
# Visualization and file saving
# ---------------------------------------------------------------------

def test_pipeline_visualization(
    pipeline: PerceptionPipeline,
    full_result: PerceptionResult,
) -> None:
    """
    Test merged-result visualization.
    """

    _print_separator(
        "TEST: PIPELINE VISUALIZATION"
    )

    visualization = pipeline.visualize(
        result=full_result,
        use_processed_image=False,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    assert isinstance(
        visualization,
        np.ndarray,
    )

    assert visualization.size > 0

    assert (
        visualization.shape[:2]
        == full_result.original_image.shape[:2]
    )

    success = cv2.imwrite(
        str(
            FULL_SCREEN_VISUALIZATION_PATH
        ),
        visualization,
    )

    assert success is True

    assert (
        FULL_SCREEN_VISUALIZATION_PATH
        .exists()
    )

    print(
        "Visualization saved to: "
        f"{FULL_SCREEN_VISUALIZATION_PATH}"
    )

    logger.info(
        "Pipeline visualization test passed."
    )


def test_save_full_pipeline_result(
    pipeline: PerceptionPipeline,
    full_result: PerceptionResult,
) -> None:
    """
    Test saving original, processed and visualization images.
    """

    _print_separator(
        "TEST: SAVE FULL RESULT"
    )

    paths = pipeline.save_result(
        result=full_result,
        output_dir=OUTPUT_DIR,
        prefix="full_screen",
        save_original=True,
        save_processed=True,
        save_visualization=True,
    )

    assert isinstance(paths, dict)

    assert {
        "original",
        "processed",
        "visualization",
    }.issubset(paths.keys())

    for name, path in paths.items():
        assert isinstance(path, Path)
        assert path.exists()
        assert path.stat().st_size > 0

        print(
            f"{name}: "
            f"{path}"
        )

    assert (
        paths["original"]
        == FULL_SCREEN_ORIGINAL_PATH
    )

    assert (
        paths["processed"]
        == FULL_SCREEN_PROCESSED_PATH
    )

    assert (
        paths["visualization"]
        == FULL_SCREEN_VISUALIZATION_PATH
    )

    logger.info(
        "Full-result saving test passed."
    )


def test_save_region_pipeline_result(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test saving region perception outputs.
    """

    _print_separator(
        "TEST: SAVE REGION RESULT"
    )

    region = _get_valid_region(
        full_screen_image
    )

    result = pipeline.process_image(
        image=full_screen_image,
        region=region,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    paths = pipeline.save_result(
        result=result,
        output_dir=OUTPUT_DIR,
        prefix="region",
        save_original=True,
        save_processed=True,
        save_visualization=True,
    )

    assert (
        paths["original"]
        == REGION_ORIGINAL_PATH
    )

    assert (
        paths["processed"]
        == REGION_PROCESSED_PATH
    )

    assert (
        paths["visualization"]
        == REGION_VISUALIZATION_PATH
    )

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    print(paths)

    logger.info(
        "Region-result saving test passed."
    )



# ---------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------

def test_invalid_region_tuple(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test malformed region tuples.
    """

    _print_separator(
        "TEST: INVALID REGION TUPLE"
    )

    with pytest.raises(TypeError):
        pipeline.process_image(
            image=full_screen_image,
            region=[0, 0, 100, 100],
        )

    with pytest.raises(ValueError):
        pipeline.process_image(
            image=full_screen_image,
            region=(0, 0, 100),
        )

    with pytest.raises(ValueError):
        pipeline.process_image(
            image=full_screen_image,
            region=(-1, 0, 100, 100),
        )

    with pytest.raises(ValueError):
        pipeline.process_image(
            image=full_screen_image,
            region=(0, 0, -100, 100),
        )

    print(
        "Invalid region tuples correctly "
        "raised exceptions."
    )

    logger.info(
        "Invalid region tuple test passed."
    )


def test_region_outside_image(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test region outside screenshot dimensions.
    """

    _print_separator(
        "TEST: REGION OUTSIDE IMAGE"
    )

    image_height, image_width = (
        full_screen_image.shape[:2]
    )

    with pytest.raises(ValueError):
        pipeline.process_image(
            image=full_screen_image,
            region=(
                image_width + 1,
                0,
                100,
                100,
            ),
        )

    with pytest.raises(ValueError):
        pipeline.process_image(
            image=full_screen_image,
            region=(
                0,
                image_height + 1,
                100,
                100,
            ),
        )

    with pytest.raises(ValueError):
        pipeline.process_image(
            image=full_screen_image,
            region=(
                0,
                0,
                image_width + 100,
                100,
            ),
        )

    print(
        "Regions outside the image correctly "
        "raised ValueError."
    )

    logger.info(
        "Outside-image region test passed."
    )


def test_invalid_boolean_override(
    pipeline: PerceptionPipeline,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test invalid Boolean pipeline options.
    """

    _print_separator(
        "TEST: INVALID BOOLEAN OVERRIDE"
    )

    with pytest.raises(TypeError):
        pipeline.process_image(
            image=full_screen_image,
            enable_ocr="yes",
        )

    with pytest.raises(TypeError):
        pipeline.process_image(
            image=full_screen_image,
            enable_ui_detection=1,
        )

    print(
        "Invalid Boolean overrides correctly "
        "raised TypeError."
    )

    logger.info(
        "Invalid Boolean override test passed."
    )


def test_invalid_image_input(
    pipeline: PerceptionPipeline,
) -> None:
    """
    Test invalid image inputs.
    """

    _print_separator(
        "TEST: INVALID IMAGE INPUT"
    )

    with pytest.raises(TypeError):
        pipeline.process_image(
            image=12345,
        )

    with pytest.raises(ValueError):
        pipeline.process_image(
            image=np.array([]),
        )

    with pytest.raises(FileNotFoundError):
        pipeline.process_image(
            image="not_existing_image.png",
        )

    print(
        "Invalid images correctly raised "
        "exceptions."
    )

    logger.info(
        "Invalid image-input test passed."
    )


def test_invalid_text_query(
    pipeline: PerceptionPipeline,
    full_result: PerceptionResult,
) -> None:
    """
    Test empty text query.
    """

    _print_separator(
        "TEST: INVALID TEXT QUERY"
    )

    with pytest.raises(ValueError):
        pipeline.find_text(
            result=full_result,
            query="",
        )

    with pytest.raises(ValueError):
        pipeline.find_text(
            result=full_result,
            query="   ",
        )

    print(
        "Empty text queries correctly "
        "raised ValueError."
    )

    logger.info(
        "Invalid text-query test passed."
    )


def test_invalid_element_type(
    pipeline: PerceptionPipeline,
    full_result: PerceptionResult,
) -> None:
    """
    Test empty element-type query.
    """

    _print_separator(
        "TEST: INVALID ELEMENT TYPE"
    )

    with pytest.raises(ValueError):
        pipeline.find_by_type(
            result=full_result,
            element_type="",
        )

    with pytest.raises(ValueError):
        full_result.get_elements_by_type(
            ""
        )

    print(
        "Empty element types correctly "
        "raised ValueError."
    )

    logger.info(
        "Invalid element-type test passed."
    )



# ---------------------------------------------------------------------
# Manual integration test
# ---------------------------------------------------------------------

def run_manual_test() -> None:
    """
    Run the complete perception workflow without pytest.

    Usage
    -----
    python tests/test_perception_pipeline.py
    """

    _print_separator(
        "MANUAL COMPLETE PERCEPTION TEST"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "1. Initializing perception components..."
    )

    capture = ScreenCapture()
    processor = ImageProcessor()

    ocr = PaddleOCREngine(
        lang=OCR_LANGUAGE,
        confidence_threshold=(
            OCR_CONFIDENCE_THRESHOLD
        ),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_gpu=False,
        sort_results=True,
    )

    detector = UIDetector(
        min_width=12,
        min_height=12,
        min_area=120,
        rectangularity_threshold=0.45,
        duplicate_iou_threshold=0.75,
        use_adaptive_threshold=True,
    )

    pipeline = PerceptionPipeline(
        screen_capture=capture,
        image_processor=processor,
        ocr_engine=ocr,
        ui_detector=detector,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
        include_unmatched_ocr=True,
    )

    print(
        "2. Capturing and processing full screen..."
    )

    full_result = pipeline.capture_and_run()

    _print_elements(
        title="FULL-SCREEN OCR RESULTS",
        elements=full_result.ocr_elements,
    )

    _print_elements(
        title="FULL-SCREEN UI RESULTS",
        elements=full_result.ui_elements,
    )

    _print_elements(
        title="FULL-SCREEN MERGED RESULTS",
        elements=full_result.merged_elements,
    )

    print(
        "3. Saving full-screen results..."
    )

    full_paths = pipeline.save_result(
        result=full_result,
        output_dir=OUTPUT_DIR,
        prefix="full_screen",
    )

    for name, path in full_paths.items():
        print(f"   {name}: {path}")

    print(
        "4. Running selected-region perception..."
    )

    region = _get_valid_region(
        full_result.original_image
    )

    region_result = pipeline.capture_and_run(
        region=region,
    )

    _print_elements(
        title="REGION MERGED RESULTS",
        elements=region_result.merged_elements,
    )

    region_paths = pipeline.save_result(
        result=region_result,
        output_dir=OUTPUT_DIR,
        prefix="region",
    )

    for name, path in region_paths.items():
        print(f"   {name}: {path}")

    print(
        f"5. Searching visible text: "
        f"{SEARCH_QUERY!r}"
    )

    matches = pipeline.find_text(
        result=full_result,
        query=SEARCH_QUERY,
        exact_match=False,
        case_sensitive=False,
    )

    _print_elements(
        title=(
            f"SEARCH RESULTS: "
            f"{SEARCH_QUERY!r}"
        ),
        elements=matches,
    )

    best_match = pipeline.find_best_text_match(
        result=full_result,
        query=SEARCH_QUERY,
        exact_match=False,
        case_sensitive=False,
    )

    if best_match is None:
        print(
            "No best text match found. "
            "Change SEARCH_QUERY to visible text."
        )
    else:
        print(
            "Best text match:"
        )

        print(
            f"  text={best_match.text!r}"
        )

        print(
            f"  type={best_match.element_type}"
        )

        print(
            f"  confidence="
            f"{best_match.confidence:.4f}"
        )

        print(
            f"  bbox={best_match.bbox}"
        )

        print(
            f"  center={best_match.center}"
        )

    print()
    print("=" * 100)
    print("MANUAL TEST COMPLETED")
    print("=" * 100)

    print("Full-screen summary:")

    for key, value in (
        full_result
        .summary()
        .items()
    ):
        print(f"  {key}: {value}")

    print("Region summary:")

    for key, value in (
        region_result
        .summary()
        .items()
    ):
        print(f"  {key}: {value}")

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    run_manual_test()