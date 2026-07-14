from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.perception.gui_element import GUIElement
from src.perception.paddle_ocr import PaddleOCREngine
from src.perception.screen_capture import ScreenCapture
from src.perception.ui_detector import UIDetector


# ---------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("D:/GUIAgent_project/screenshots")

SCREENSHOT_PATH = OUTPUT_DIR / "ui_full_screen.png"
OCR_VISUALIZATION_PATH = OUTPUT_DIR / "ui_visualization.png"
UI_VISUALIZATION_PATH = OUTPUT_DIR / "ui_visualization.png"
MERGED_VISUALIZATION_PATH = OUTPUT_DIR / "ui_merged_visualization.png"
REGION_VISUALIZATION_PATH = OUTPUT_DIR / "ui_region_visualization.png"
REGION_IMAGE_PATH = OUTPUT_DIR / "ui_region_image.png"

# Change this value to text currently visible on your screen.
SEARCH_QUERY = "BBC"

OCR_LANGUAGE = "ch"
OCR_CONFIDENCE_THRESHOLD = 0.50

# Test region
REGION_LEFT = 10
REGION_TOP = 10
REGION_WIDTH = 890
REGION_HEIGHT = 1000


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def output_dir() -> Path:
    """
    Create and return the UI-detector output directory.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="session")
def screen_capture() -> ScreenCapture:
    """
    Create one ScreenCapture instance for the test session.
    """

    return ScreenCapture()


@pytest.fixture(scope="session")
def ocr_engine() -> PaddleOCREngine:
    """
    Create one PaddleOCREngine instance for the test session.
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
    Create one UIDetector instance for the test session.
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
def full_screen_image(
    screen_capture: ScreenCapture,
    output_dir: Path,
) -> np.ndarray:
    """
    Capture and save the current desktop screenshot.
    """

    image = screen_capture.capture_screen()

    assert isinstance(image, np.ndarray)
    assert image.size > 0
    assert image.ndim in (2, 3)

    success = cv2.imwrite(str(SCREENSHOT_PATH), image)

    assert success is True
    assert SCREENSHOT_PATH.exists()
    assert SCREENSHOT_PATH.stat().st_size > 0

    logger.info(
        "Full-screen screenshot saved to: %s",
        SCREENSHOT_PATH,
    )

    return image


@pytest.fixture(scope="session")
def ocr_elements(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> list[GUIElement]:
    """
    Run OCR once and reuse the results.
    """

    elements = ocr_engine.detect(full_screen_image)

    assert isinstance(elements, list)

    logger.info(
        "PaddleOCR detected %d text elements.",
        len(elements),
    )

    return elements


@pytest.fixture(scope="session")
def ui_elements(
    ui_detector: UIDetector,
    full_screen_image: np.ndarray,
    ocr_elements: list[GUIElement],
) -> list[GUIElement]:
    """
    Detect graphical UI components.
    """

    elements = ui_detector.detect(
        image=full_screen_image,
        ocr_elements=ocr_elements,
    )

    assert isinstance(elements, list)

    logger.info(
        "UIDetector detected %d graphical elements.",
        len(elements),
    )

    return elements


@pytest.fixture(scope="session")
def merged_elements(
    ui_detector: UIDetector,
    full_screen_image: np.ndarray,
    ocr_elements: list[GUIElement],
) -> list[GUIElement]:
    """
    Detect and merge graphical UI elements with OCR text elements.
    """

    elements = ui_detector.detect_and_merge(
        image=full_screen_image,
        ocr_elements=ocr_elements,
        include_unmatched_ocr=True,
    )

    assert isinstance(elements, list)

    logger.info(
        "Merged perception output contains %d elements.",
        len(elements),
    )

    return elements


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _print_separator(title: str) -> None:
    """
    Print a readable test section separator.
    """

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def _print_elements(
    title: str,
    elements: Sequence[GUIElement],
) -> None:
    """
    Print GUIElement objects in a readable form.
    """

    _print_separator(title)

    if not elements:
        print("No GUI elements detected.")
        return

    for index, element in enumerate(elements, start=1):
        print(
            f"[{index:03d}] "
            f"type={element.element_type!r}, "
            f"text={element.text!r}, "
            f"confidence={element.confidence:.4f}, "
            f"bbox={element.bbox}, "
            f"center={element.center}"
        )


def _validate_gui_element(element: GUIElement) -> None:
    """
    Validate one GUIElement instance.
    """

    assert isinstance(element, GUIElement)

    assert isinstance(element.text, str)

    assert isinstance(element.bbox, tuple)
    assert len(element.bbox) == 4

    x1, y1, x2, y2 = element.bbox

    assert all(
        isinstance(value, int)
        for value in (x1, y1, x2, y2)
    )

    assert x1 >= 0
    assert y1 >= 0
    assert x2 > x1
    assert y2 > y1

    assert isinstance(element.confidence, float)
    assert 0.0 <= element.confidence <= 1.0

    assert isinstance(element.element_type, str)
    assert element.element_type.strip() != ""

    if element.center is not None:
        assert isinstance(element.center, tuple)
        assert len(element.center) == 2

        center_x, center_y = element.center

        assert isinstance(center_x, int)
        assert isinstance(center_y, int)

        assert x1 <= center_x <= x2
        assert y1 <= center_y <= y2


def _validate_region_against_image(
    image: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Validate the configured region against the current screenshot.

    If the configured region is larger than the screen, it is reduced to
    remain inside the screen.
    """

    image_height, image_width = image.shape[:2]

    left = REGION_LEFT
    top = REGION_TOP

    if left >= image_width or top >= image_height:
        raise ValueError(
            "Configured test region starts outside the screenshot."
        )

    width = min(
        REGION_WIDTH,
        image_width - left,
    )

    height = min(
        REGION_HEIGHT,
        image_height - top,
    )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Configured test region is invalid."
        )

    return left, top, width, height


def _crop_region(
    image: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Crop a region from an image.
    """

    right = left + width
    bottom = top + height

    region = image[top:bottom, left:right]

    if region.size == 0:
        raise ValueError("Cropped test region is empty.")

    return region


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_screen_capture(
    full_screen_image: np.ndarray,
) -> None:
    """
    Test that the desktop screenshot is valid.
    """

    _print_separator("TEST: SCREEN CAPTURE")

    assert isinstance(full_screen_image, np.ndarray)
    assert full_screen_image.size > 0
    assert SCREENSHOT_PATH.exists()
    assert SCREENSHOT_PATH.stat().st_size > 0

    height, width = full_screen_image.shape[:2]

    print(f"Screenshot shape: {full_screen_image.shape}")
    print(f"Screenshot width: {width}")
    print(f"Screenshot height: {height}")
    print(f"Saved path: {SCREENSHOT_PATH}")

    logger.info("Screen-capture test passed.")


def test_ocr_elements_structure(
    ocr_elements: list[GUIElement],
) -> None:
    """
    Validate PaddleOCR output before UI detection.
    """

    _print_elements(
        title="OCR ELEMENTS",
        elements=ocr_elements,
    )

    assert isinstance(ocr_elements, list)

    for element in ocr_elements:
        _validate_gui_element(element)
        assert element.element_type == "text"

    print(f"OCR element count: {len(ocr_elements)}")

    logger.info("OCR structure test passed.")


def test_ui_detection(
    ui_elements: list[GUIElement],
) -> None:
    """
    Test graphical UI-element detection.
    """

    _print_elements(
        title="UI DETECTOR RESULTS",
        elements=ui_elements,
    )

    assert isinstance(ui_elements, list)

    for element in ui_elements:
        _validate_gui_element(element)

    detected_types = sorted(
        {
            element.element_type
            for element in ui_elements
        }
    )

    print(f"UI element count: {len(ui_elements)}")
    print(f"Detected element types: {detected_types}")

    logger.info("UI-detection test passed.")


def test_merged_detection(
    merged_elements: list[GUIElement],
    ocr_elements: list[GUIElement],
) -> None:
    """
    Test merging OCR and graphical UI elements.
    """

    _print_elements(
        title="MERGED OCR AND UI RESULTS",
        elements=merged_elements,
    )

    assert isinstance(merged_elements, list)

    for element in merged_elements:
        _validate_gui_element(element)

    assert len(merged_elements) >= 0

    if ocr_elements:
        merged_texts = {
            element.text
            for element in merged_elements
            if element.text
        }

        original_texts = {
            element.text
            for element in ocr_elements
            if element.text
        }

        assert merged_texts.intersection(original_texts), (
            "No OCR text survived the merge process."
        )

    type_counts: dict[str, int] = {}

    for element in merged_elements:
        type_counts[element.element_type] = (
            type_counts.get(element.element_type, 0) + 1
        )

    print("Element type counts:")

    for element_type, count in sorted(type_counts.items()):
        print(f"  {element_type}: {count}")

    logger.info("Merged detection test passed.")


def test_find_by_text(
    ui_detector: UIDetector,
    merged_elements: list[GUIElement],
) -> None:
    """
    Test finding UI elements by visible text.
    """

    _print_separator("TEST: FIND BY TEXT")

    matches = ui_detector.find_by_text(
        elements=merged_elements,
        query=SEARCH_QUERY,
        exact_match=False,
        case_sensitive=False,
    )

    assert isinstance(matches, list)

    print(f"Search query: {SEARCH_QUERY!r}")
    print(f"Match count: {len(matches)}")

    _print_elements(
        title=f"TEXT MATCH RESULTS: {SEARCH_QUERY!r}",
        elements=matches,
    )

    if not matches:
        pytest.skip(
            f"The configured query {SEARCH_QUERY!r} was not detected. "
            "Change SEARCH_QUERY to text currently visible on the screen."
        )

    expected = SEARCH_QUERY.lower()

    for element in matches:
        assert expected in element.text.lower()

    logger.info("find_by_text test passed.")


def test_exact_text_search(
    ui_detector: UIDetector,
    merged_elements: list[GUIElement],
) -> None:
    """
    Test exact text matching using a real detected element.
    """

    _print_separator("TEST: EXACT TEXT SEARCH")

    text_elements = [
        element
        for element in merged_elements
        if element.text.strip()
    ]

    if not text_elements:
        pytest.skip(
            "No text-bearing elements available for exact matching."
        )

    query = text_elements[0].text

    matches = ui_detector.find_by_text(
        elements=merged_elements,
        query=query,
        exact_match=True,
        case_sensitive=True,
    )

    assert len(matches) > 0

    for element in matches:
        assert element.text == query

    print(f"Exact query: {query!r}")
    print(f"Exact match count: {len(matches)}")

    logger.info("Exact text-search test passed.")


def test_find_by_type(
    ui_detector: UIDetector,
    merged_elements: list[GUIElement],
) -> None:
    """
    Test filtering GUI elements by type.
    """

    _print_separator("TEST: FIND BY TYPE")

    available_types = sorted(
        {
            element.element_type
            for element in merged_elements
        }
    )

    if not available_types:
        pytest.skip("No GUI element types available.")

    print(f"Available types: {available_types}")

    selected_type = available_types[0]

    matches = ui_detector.find_by_type(
        elements=merged_elements,
        element_type=selected_type,
    )

    assert isinstance(matches, list)
    assert len(matches) > 0

    for element in matches:
        assert element.element_type == selected_type

    print(f"Selected type: {selected_type!r}")
    print(f"Match count: {len(matches)}")

    logger.info("find_by_type test passed.")


def test_region_detection(
    ui_detector: UIDetector,
    full_screen_image: np.ndarray,
    ocr_elements: list[GUIElement],
) -> None:
    """
    Test UI detection inside the selected region.
    """

    _print_separator("TEST: REGION UI DETECTION")

    left, top, width, height = _validate_region_against_image(
        full_screen_image
    )

    print(
        "Region configuration: "
        f"left={left}, "
        f"top={top}, "
        f"width={width}, "
        f"height={height}"
    )

    region_elements = ui_detector.detect_region(
        image=full_screen_image,
        left=left,
        top=top,
        width=width,
        height=height,
        ocr_elements=ocr_elements,
    )

    assert isinstance(region_elements, list)

    for element in region_elements:
        _validate_gui_element(element)

        x1, y1, x2, y2 = element.bbox

        assert x1 >= left
        assert y1 >= top
        assert x2 <= left + width
        assert y2 <= top + height

    _print_elements(
        title="REGION UI DETECTION RESULTS",
        elements=region_elements,
    )

    cropped_region = _crop_region(
        image=full_screen_image,
        left=left,
        top=top,
        width=width,
        height=height,
    )

    success = cv2.imwrite(
        str(REGION_IMAGE_PATH),
        cropped_region,
    )

    assert success is True
    assert REGION_IMAGE_PATH.exists()

    print(f"Region element count: {len(region_elements)}")
    print(f"Region image saved to: {REGION_IMAGE_PATH}")

    logger.info("Region detection test passed.")


def test_ocr_visualization_save(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
    ocr_elements: list[GUIElement],
) -> None:
    """
    Test saving OCR bounding-box visualization.
    """

    _print_separator("TEST: OCR VISUALIZATION")

    visualization = ocr_engine.visualize(
        image=full_screen_image,
        elements=ocr_elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    saved_path = ocr_engine.save_visualization(
        image=visualization,
        save_path=OCR_VISUALIZATION_PATH,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    print(f"OCR visualization saved to: {saved_path}")

    logger.info("OCR visualization test passed.")


def test_ui_visualization_save(
    ui_detector: UIDetector,
    full_screen_image: np.ndarray,
    ui_elements: list[GUIElement],
) -> None:
    """
    Test saving graphical UI-detection visualization.
    """

    _print_separator("TEST: UI VISUALIZATION")

    visualization = ui_detector.visualize(
        image=full_screen_image,
        elements=ui_elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    assert isinstance(visualization, np.ndarray)
    assert visualization.shape[:2] == full_screen_image.shape[:2]

    saved_path = ui_detector.save_visualization(
        image=visualization,
        save_path=UI_VISUALIZATION_PATH,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    print(f"UI visualization saved to: {saved_path}")

    logger.info("UI visualization test passed.")


def test_merged_visualization_save(
    ui_detector: UIDetector,
    full_screen_image: np.ndarray,
    merged_elements: list[GUIElement],
) -> None:
    """
    Test saving the merged OCR and UI visualization.
    """

    _print_separator("TEST: MERGED VISUALIZATION")

    visualization = ui_detector.visualize(
        image=full_screen_image,
        elements=merged_elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    saved_path = ui_detector.save_visualization(
        image=visualization,
        save_path=MERGED_VISUALIZATION_PATH,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    print(f"Merged visualization saved to: {saved_path}")

    logger.info("Merged visualization test passed.")


def test_region_visualization_save(
    ui_detector: UIDetector,
    full_screen_image: np.ndarray,
    ocr_elements: list[GUIElement],
) -> None:
    """
    Test saving region UI-detection visualization.
    """

    _print_separator("TEST: REGION VISUALIZATION")

    left, top, width, height = _validate_region_against_image(
        full_screen_image
    )

    region_elements = ui_detector.detect_region(
        image=full_screen_image,
        left=left,
        top=top,
        width=width,
        height=height,
        ocr_elements=ocr_elements,
    )

    visualization = ui_detector.visualize(
        image=full_screen_image,
        elements=region_elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    cv2.rectangle(
        visualization,
        (left, top),
        (left + width, top + height),
        (255, 255, 0),
        thickness=3,
    )

    saved_path = ui_detector.save_visualization(
        image=visualization,
        save_path=REGION_VISUALIZATION_PATH,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    print(f"Region visualization saved to: {saved_path}")

    logger.info("Region visualization test passed.")


def test_invalid_text_query_raises_error(
    ui_detector: UIDetector,
    merged_elements: list[GUIElement],
) -> None:
    """
    Test empty text-query validation.
    """

    _print_separator("TEST: INVALID TEXT QUERY")

    with pytest.raises(ValueError):
        ui_detector.find_by_text(
            elements=merged_elements,
            query="",
        )

    with pytest.raises(ValueError):
        ui_detector.find_by_text(
            elements=merged_elements,
            query="   ",
        )

    print("Invalid text queries correctly raised ValueError.")

    logger.info("Invalid text-query test passed.")


def test_invalid_element_type_raises_error(
    ui_detector: UIDetector,
    merged_elements: list[GUIElement],
) -> None:
    """
    Test empty type-query validation.
    """

    _print_separator("TEST: INVALID ELEMENT TYPE")

    with pytest.raises(ValueError):
        ui_detector.find_by_type(
            elements=merged_elements,
            element_type="",
        )

    with pytest.raises(ValueError):
        ui_detector.find_by_type(
            elements=merged_elements,
            element_type="   ",
        )

    print("Invalid element types correctly raised ValueError.")

    logger.info("Invalid type-query test passed.")


def test_invalid_region_raises_error(
    ui_detector: UIDetector,
    full_screen_image: np.ndarray,
    ocr_elements: list[GUIElement],
) -> None:
    """
    Test region validation.
    """

    _print_separator("TEST: INVALID REGION")

    image_height, image_width = full_screen_image.shape[:2]

    with pytest.raises(ValueError):
        ui_detector.detect_region(
            image=full_screen_image,
            left=image_width + 10,
            top=0,
            width=100,
            height=100,
            ocr_elements=ocr_elements,
        )

    with pytest.raises(ValueError):
        ui_detector.detect_region(
            image=full_screen_image,
            left=0,
            top=0,
            width=-100,
            height=100,
            ocr_elements=ocr_elements,
        )

    with pytest.raises(ValueError):
        ui_detector.detect_region(
            image=full_screen_image,
            left=0,
            top=0,
            width=image_width + 100,
            height=100,
            ocr_elements=ocr_elements,
        )

    print("Invalid UI-detection regions correctly raised ValueError.")

    logger.info("Invalid region test passed.")


def test_detector_repr(
    ui_detector: UIDetector,
) -> None:
    """
    Test the UIDetector string representation.
    """

    _print_separator("TEST: DETECTOR REPR")

    representation = repr(ui_detector)

    assert isinstance(representation, str)
    assert "UIDetector" in representation

    print(representation)

    logger.info("UIDetector repr test passed.")


# ---------------------------------------------------------------------
# Manual execution
# ---------------------------------------------------------------------

def run_manual_test() -> None:
    """
    Run the complete UI detection workflow without pytest.

    Usage
    -----
    python tests/test_ui_detector.py
    """

    _print_separator("MANUAL UI DETECTOR INTEGRATION TEST")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Initializing screen capture...")
    capture = ScreenCapture()

    print("2. Capturing current desktop...")
    image = capture.capture_screen()

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise RuntimeError("Screen capture returned an invalid image.")

    saved = cv2.imwrite(
        str(SCREENSHOT_PATH),
        image,
    )

    if not saved:
        raise IOError(
            f"Failed to save screenshot to {SCREENSHOT_PATH}"
        )

    print(f"   Screenshot saved to: {SCREENSHOT_PATH}")

    print("3. Initializing PaddleOCR...")
    ocr = PaddleOCREngine(
        lang=OCR_LANGUAGE,
        confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_gpu=False,
        sort_results=True,
    )

    print("4. Running OCR...")
    text_elements = ocr.detect(image)

    _print_elements(
        title="MANUAL OCR RESULTS",
        elements=text_elements,
    )

    print("5. Initializing UIDetector...")
    detector = UIDetector(
        min_width=12,
        min_height=12,
        min_area=120,
        rectangularity_threshold=0.45,
        duplicate_iou_threshold=0.75,
        use_adaptive_threshold=True,
    )

    print("6. Detecting graphical UI elements...")
    graphical_elements = detector.detect(
        image=image,
        ocr_elements=text_elements,
    )

    _print_elements(
        title="MANUAL GRAPHICAL UI RESULTS",
        elements=graphical_elements,
    )

    print("7. Merging OCR and graphical elements...")
    all_elements = detector.detect_and_merge(
        image=image,
        ocr_elements=text_elements,
        include_unmatched_ocr=True,
    )

    _print_elements(
        title="MANUAL MERGED RESULTS",
        elements=all_elements,
    )

    print("8. Saving merged visualization...")
    merged_visualization = detector.visualize(
        image=image,
        elements=all_elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    detector.save_visualization(
        image=merged_visualization,
        save_path=MERGED_VISUALIZATION_PATH,
    )

    print(
        "   Merged visualization saved to: "
        f"{MERGED_VISUALIZATION_PATH}"
    )

    print("9. Running region detection...")
    left, top, width, height = _validate_region_against_image(
        image
    )

    region_elements = detector.detect_region(
        image=image,
        left=left,
        top=top,
        width=width,
        height=height,
        ocr_elements=text_elements,
    )

    _print_elements(
        title="MANUAL REGION RESULTS",
        elements=region_elements,
    )

    region_visualization = detector.visualize(
        image=image,
        elements=region_elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    cv2.rectangle(
        region_visualization,
        (left, top),
        (left + width, top + height),
        (255, 255, 0),
        thickness=3,
    )

    detector.save_visualization(
        image=region_visualization,
        save_path=REGION_VISUALIZATION_PATH,
    )

    print(
        "   Region visualization saved to: "
        f"{REGION_VISUALIZATION_PATH}"
    )

    print(f"10. Searching for text: {SEARCH_QUERY!r}")

    text_matches = detector.find_by_text(
        elements=all_elements,
        query=SEARCH_QUERY,
        exact_match=False,
        case_sensitive=False,
    )

    _print_elements(
        title=f"MANUAL TEXT SEARCH RESULTS: {SEARCH_QUERY!r}",
        elements=text_matches,
    )

    type_counts: dict[str, int] = {}

    for element in all_elements:
        type_counts[element.element_type] = (
            type_counts.get(element.element_type, 0) + 1
        )

    print()
    print("=" * 100)
    print("MANUAL TEST COMPLETED")
    print("=" * 100)
    print(f"OCR element count: {len(text_elements)}")
    print(f"Graphical element count: {len(graphical_elements)}")
    print(f"Merged element count: {len(all_elements)}")
    print(f"Region element count: {len(region_elements)}")
    print(f"Search match count: {len(text_matches)}")
    print("Element type counts:")

    for element_type, count in sorted(type_counts.items()):
        print(f"  {element_type}: {count}")

    print(f"Screenshot: {SCREENSHOT_PATH}")
    print(f"Merged visualization: {MERGED_VISUALIZATION_PATH}")
    print(f"Region visualization: {REGION_VISUALIZATION_PATH}")


if __name__ == "__main__":
    run_manual_test()