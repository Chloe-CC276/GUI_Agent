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


# ---------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path('D:/GUIAgent_project/screenshots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_SCREEN_PATH = OUTPUT_DIR / "ocr_full_screen.png"
FULL_SCREEN_VISUALIZATION_PATH = (
    OUTPUT_DIR / "ocr_full_screen_visualization.png"
)

REGION_SCREEN_PATH = OUTPUT_DIR / "ocr_region_screen.png"
REGION_VISUALIZATION_PATH = (
    OUTPUT_DIR / "ocr_region_visualization.png"
)

SEARCH_QUERY = "知乎"

# OCR confidence threshold.
CONFIDENCE_THRESHOLD = 0.50

# OCR language.
OCR_LANGUAGE = "ch"

# Region used for the region OCR test.
REGION_LEFT_RATIO = 0.10
REGION_TOP_RATIO = 0.10
REGION_WIDTH_RATIO = 0.80
REGION_HEIGHT_RATIO = 0.60

REGION_LEFT = 40
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
    Create and return the OCR output directory.
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

    Model initialization may download PaddleOCR model files during the
    first execution.
    """

    return PaddleOCREngine(
        lang=OCR_LANGUAGE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_gpu=False,
        sort_results=True,
    )


@pytest.fixture(scope="session")
def full_screen_image(
    screen_capture: ScreenCapture,
    output_dir: Path,
) -> np.ndarray:
    """
    Capture and save the full desktop screenshot.
    """

    image = screen_capture.capture_screen()

    assert isinstance(image, np.ndarray)
    assert image.size > 0
    assert image.ndim in (2, 3)

    success = cv2.imwrite(str(FULL_SCREEN_PATH), image)

    assert success is True
    assert FULL_SCREEN_PATH.exists()

    logger.info(
        "Full-screen screenshot saved to: %s",
        FULL_SCREEN_PATH,
    )

    return image


@pytest.fixture(scope="session")
def full_screen_elements(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> list[GUIElement]:
    """
    Run OCR once on the full screenshot and reuse the results.
    """

    elements = ocr_engine.detect(full_screen_image)

    logger.info(
        "Full-screen OCR detected %d elements.",
        len(elements),
    )

    _print_elements(
        title="FULL-SCREEN OCR RESULTS",
        elements=elements,
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
    print("=" * 90)
    print(title)
    print("=" * 90)


def _print_elements(
    title: str,
    elements: Sequence[GUIElement],
) -> None:
    """
    Print detected GUI elements.
    """

    _print_separator(title)

    if not elements:
        print("No OCR elements detected.")
        return

    for index, element in enumerate(elements, start=1):
        print(
            f"[{index:03d}] "
            f"text={element.text!r}, "
            f"confidence={element.confidence:.4f}, "
            f"bbox={element.bbox}, "
            f"center={element.center}, "
            f"type={element.element_type!r}"
        )


def _validate_gui_element(element: GUIElement) -> None:
    """
    Validate one GUIElement object.
    """

    assert isinstance(element, GUIElement)

    assert isinstance(element.text, str)
    assert element.text.strip() != ""

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

    if element.center is not None:
        assert isinstance(element.center, tuple)
        assert len(element.center) == 2

        center_x, center_y = element.center

        assert isinstance(center_x, int)
        assert isinstance(center_y, int)

        assert x1 <= center_x <= x2
        assert y1 <= center_y <= y2


def _calculate_test_region(
    image: np.ndarray,
) -> tuple[int, int, int, int]:
    image_height, image_width = image.shape[:2]

    left = REGION_LEFT
    top = REGION_TOP
    width = REGION_WIDTH
    height = REGION_HEIGHT

    if left < 0 or top < 0:
        raise ValueError("left and top must be non-negative.")

    if left + width > image_width:
        raise ValueError(
            f"Region exceeds image width: "
            f"{left + width} > {image_width}"
        )

    if top + height > image_height:
        raise ValueError(
            f"Region exceeds image height: "
            f"{top + height} > {image_height}"
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

def test_full_screen_capture(
    full_screen_image: np.ndarray,
) -> None:
    """
    Test that the full desktop screenshot is valid and saved.
    """

    _print_separator("TEST: FULL-SCREEN CAPTURE")

    assert isinstance(full_screen_image, np.ndarray)
    assert full_screen_image.size > 0
    assert FULL_SCREEN_PATH.exists()
    assert FULL_SCREEN_PATH.stat().st_size > 0

    height, width = full_screen_image.shape[:2]

    print(f"Screenshot shape: {full_screen_image.shape}")
    print(f"Screenshot width: {width}")
    print(f"Screenshot height: {height}")
    print(f"Saved path: {FULL_SCREEN_PATH}")

    logger.info("Full-screen capture test passed.")


def test_full_screen_ocr_recognition(
    full_screen_elements: list[GUIElement],
) -> None:
    """
    Test OCR recognition on the full desktop screenshot.
    """

    _print_separator("TEST: FULL-SCREEN OCR RECOGNITION")

    assert isinstance(full_screen_elements, list)

    assert len(full_screen_elements) > 0, (
        "No OCR text was detected. Open an application containing visible "
        "text, such as Notepad, Chrome or VS Code, and rerun the test."
    )

    for element in full_screen_elements:
        _validate_gui_element(element)

    recognized_text = "\n".join(
        element.text
        for element in full_screen_elements
    )

    print(f"Detected element count: {len(full_screen_elements)}")
    print("Recognized text:")
    print(recognized_text)

    logger.info("Full-screen OCR recognition test passed.")


def test_full_screen_detect_text(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test the convenience method that returns only recognized strings.
    """

    _print_separator("TEST: DETECT TEXT")

    texts = ocr_engine.detect_text(full_screen_image)

    assert isinstance(texts, list)
    assert len(texts) > 0
    assert all(isinstance(text, str) for text in texts)
    assert all(text.strip() != "" for text in texts)

    print(f"Detected text count: {len(texts)}")

    for index, text in enumerate(texts, start=1):
        print(f"[{index:03d}] {text}")

    logger.info("detect_text test passed.")


def test_region_ocr_recognition(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test OCR on a selected region of the current screen.
    """

    _print_separator("TEST: REGION OCR RECOGNITION")

    left, top, width, height = _calculate_test_region(
        full_screen_image
    )

    print(
        "OCR region: "
        f"left={left}, "
        f"top={top}, "
        f"width={width}, "
        f"height={height}"
    )

    cropped_region = _crop_region(
        image=full_screen_image,
        left=left,
        top=top,
        width=width,
        height=height,
    )

    region_saved = cv2.imwrite(
        str(REGION_SCREEN_PATH),
        cropped_region,
    )

    assert region_saved is True
    assert REGION_SCREEN_PATH.exists()

    elements = ocr_engine.detect_region(
        image=full_screen_image,
        left=left,
        top=top,
        width=width,
        height=height,
    )

    assert isinstance(elements, list)

    _print_elements(
        title="REGION OCR RESULTS",
        elements=elements,
    )

    for element in elements:
        _validate_gui_element(element)

        x1, y1, x2, y2 = element.bbox

        assert x1 >= left
        assert y1 >= top
        assert x2 <= left + width
        assert y2 <= top + height

    print(f"Region screenshot saved to: {REGION_SCREEN_PATH}")
    print(f"Region OCR element count: {len(elements)}")

    logger.info("Region OCR recognition test passed.")


def test_find_text(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test searching OCR elements by visible text.

    Before running, change SEARCH_QUERY near the top of this file to text
    visible on your current screen.
    """

    _print_separator("TEST: FIND TEXT")

    matches = ocr_engine.find_text(
        image=full_screen_image,
        query=SEARCH_QUERY,
        case_sensitive=False,
        exact_match=False,
    )

    assert isinstance(matches, list)

    print(f"Search query: {SEARCH_QUERY!r}")
    print(f"Match count: {len(matches)}")

    _print_elements(
        title=f"TEXT SEARCH RESULTS: {SEARCH_QUERY!r}",
        elements=matches,
    )

    if not matches:
        pytest.skip(
            f"The query {SEARCH_QUERY!r} was not detected on the current "
            "screen. Change SEARCH_QUERY to visible text and rerun the test."
        )

    expected = SEARCH_QUERY.lower()

    for element in matches:
        _validate_gui_element(element)

        assert expected in element.text.lower()

    logger.info("Text search test passed.")


def test_exact_text_search(
    ocr_engine: PaddleOCREngine,
    full_screen_elements: list[GUIElement],
    full_screen_image: np.ndarray,
) -> None:
    """
    Test exact-match text search using one real OCR result.

    This avoids depending on a manually configured screen string.
    """

    _print_separator("TEST: EXACT TEXT SEARCH")

    if not full_screen_elements:
        pytest.skip("No OCR elements are available for exact search.")

    query = full_screen_elements[0].text

    matches = ocr_engine.find_text(
        image=full_screen_image,
        query=query,
        case_sensitive=True,
        exact_match=True,
    )

    assert isinstance(matches, list)
    assert len(matches) > 0

    for element in matches:
        assert element.text == query

    print(f"Exact query: {query!r}")
    print(f"Exact match count: {len(matches)}")

    logger.info("Exact text-search test passed.")


def test_full_screen_visualization_save(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
    full_screen_elements: list[GUIElement],
    output_dir: Path,
) -> None:
    """
    Test drawing and saving full-screen OCR bounding boxes.
    """

    _print_separator("TEST: FULL-SCREEN VISUALIZATION")

    visualization = ocr_engine.visualize(
        image=full_screen_image,
        elements=full_screen_elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    assert isinstance(visualization, np.ndarray)
    assert visualization.size > 0
    assert visualization.shape[:2] == full_screen_image.shape[:2]

    saved_path = ocr_engine.save_visualization(
        image=visualization,
        save_path=FULL_SCREEN_VISUALIZATION_PATH,
    )

    assert saved_path == FULL_SCREEN_VISUALIZATION_PATH
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    saved_image = cv2.imread(str(saved_path))

    assert saved_image is not None
    assert saved_image.size > 0

    print(f"Visualization saved to: {saved_path}")
    print(f"File size: {saved_path.stat().st_size} bytes")

    logger.info("Full-screen visualization test passed.")


def test_region_visualization_save(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Test drawing region OCR boxes on the original full-screen image.
    """

    _print_separator("TEST: REGION VISUALIZATION")

    left, top, width, height = _calculate_test_region(
        full_screen_image
    )

    elements = ocr_engine.detect_region(
        image=full_screen_image,
        left=left,
        top=top,
        width=width,
        height=height,
    )

    visualization = ocr_engine.visualize(
        image=full_screen_image,
        elements=elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    assert isinstance(visualization, np.ndarray)
    assert visualization.size > 0

    # Draw the selected OCR region as an additional rectangle.
    cv2.rectangle(
        visualization,
        (left, top),
        (left + width, top + height),
        (255, 255, 0),
        thickness=3,
    )

    saved_path = ocr_engine.save_visualization(
        image=visualization,
        save_path=REGION_VISUALIZATION_PATH,
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    print(f"Region visualization saved to: {saved_path}")
    print(f"Region OCR element count: {len(elements)}")

    logger.info("Region visualization test passed.")


def test_confidence_threshold_filtering(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test that a higher confidence threshold cannot return more results
    than a lower threshold.
    """

    _print_separator("TEST: CONFIDENCE FILTERING")

    low_threshold_elements = ocr_engine.detect(
        image=full_screen_image,
        confidence_threshold=0.20,
    )

    high_threshold_elements = ocr_engine.detect(
        image=full_screen_image,
        confidence_threshold=0.90,
    )

    assert len(high_threshold_elements) <= len(
        low_threshold_elements
    )

    assert all(
        element.confidence >= 0.20
        for element in low_threshold_elements
    )

    assert all(
        element.confidence >= 0.90
        for element in high_threshold_elements
    )

    print(
        "Low-threshold result count: "
        f"{len(low_threshold_elements)}"
    )
    print(
        "High-threshold result count: "
        f"{len(high_threshold_elements)}"
    )

    logger.info("Confidence filtering test passed.")


def test_ocr_result_sorting(
    full_screen_elements: list[GUIElement],
) -> None:
    """
    Test that OCR output is approximately sorted in reading order.
    """

    _print_separator("TEST: OCR RESULT SORTING")

    if len(full_screen_elements) < 2:
        pytest.skip(
            "At least two OCR elements are required for sorting test."
        )

    previous_row = None
    previous_x = None
    row_tolerance = 15

    for element in full_screen_elements:
        x1, y1, _, _ = element.bbox
        current_row = int(round(y1 / row_tolerance))

        if previous_row is not None:
            assert current_row >= previous_row

            if current_row == previous_row:
                assert x1 >= previous_x

        previous_row = current_row
        previous_x = x1

    print("OCR elements are sorted by approximate reading order.")

    logger.info("OCR result sorting test passed.")


def test_invalid_region_raises_error(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test region validation.
    """

    _print_separator("TEST: INVALID REGION")

    image_height, image_width = full_screen_image.shape[:2]

    with pytest.raises(ValueError):
        ocr_engine.detect_region(
            image=full_screen_image,
            left=image_width + 10,
            top=0,
            width=100,
            height=100,
        )

    with pytest.raises(ValueError):
        ocr_engine.detect_region(
            image=full_screen_image,
            left=0,
            top=0,
            width=-100,
            height=100,
        )

    with pytest.raises(ValueError):
        ocr_engine.detect_region(
            image=full_screen_image,
            left=0,
            top=0,
            width=image_width + 100,
            height=100,
        )

    print("Invalid OCR regions correctly raised ValueError.")

    logger.info("Invalid region test passed.")


def test_invalid_search_query_raises_error(
    ocr_engine: PaddleOCREngine,
    full_screen_image: np.ndarray,
) -> None:
    """
    Test empty search-query validation.
    """

    _print_separator("TEST: INVALID SEARCH QUERY")

    with pytest.raises(ValueError):
        ocr_engine.find_text(
            image=full_screen_image,
            query="",
        )

    with pytest.raises(ValueError):
        ocr_engine.find_text(
            image=full_screen_image,
            query="   ",
        )

    print("Empty search queries correctly raised ValueError.")

    logger.info("Invalid search-query test passed.")


# ---------------------------------------------------------------------
# Manual execution
# ---------------------------------------------------------------------

def run_manual_test() -> None:
    """
    Run the main OCR workflow without pytest.

    Usage
    -----
    python tests/test_paddle_ocr.py
    """

    _print_separator("MANUAL PADDLEOCR INTEGRATION TEST")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Initializing screen capture...")
    capture = ScreenCapture()

    print("2. Capturing full screen...")
    image = capture.capture_screen()

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise RuntimeError("Screen capture returned an invalid image.")

    saved = cv2.imwrite(str(FULL_SCREEN_PATH), image)

    if not saved:
        raise IOError(
            f"Failed to save screenshot to {FULL_SCREEN_PATH}"
        )

    print(f"   Screenshot saved to: {FULL_SCREEN_PATH}")

    print("3. Initializing PaddleOCR...")
    engine = PaddleOCREngine(
        lang=OCR_LANGUAGE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_gpu=False,
        sort_results=True,
    )

    print("4. Running full-screen OCR...")
    elements = engine.detect(image)

    _print_elements(
        title="MANUAL FULL-SCREEN OCR RESULTS",
        elements=elements,
    )

    print("5. Saving full-screen visualization...")
    visualization = engine.visualize(
        image=image,
        elements=elements,
        show_text=True,
        show_confidence=True,
        show_center=True,
    )

    engine.save_visualization(
        image=visualization,
        save_path=FULL_SCREEN_VISUALIZATION_PATH,
    )

    print(
        "   Full-screen visualization saved to: "
        f"{FULL_SCREEN_VISUALIZATION_PATH}"
    )

    print("6. Running region OCR...")
    left, top, width, height = _calculate_test_region(image)

    region_elements = engine.detect_region(
        image=image,
        left=left,
        top=top,
        width=width,
        height=height,
    )

    _print_elements(
        title="MANUAL REGION OCR RESULTS",
        elements=region_elements,
    )

    region_visualization = engine.visualize(
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

    engine.save_visualization(
        image=region_visualization,
        save_path=REGION_VISUALIZATION_PATH,
    )

    print(
        "   Region visualization saved to: "
        f"{REGION_VISUALIZATION_PATH}"
    )

    print(f"7. Searching for text: {SEARCH_QUERY!r}")

    matches = engine.find_text(
        image=image,
        query=SEARCH_QUERY,
        case_sensitive=False,
        exact_match=False,
    )

    _print_elements(
        title=f"MANUAL SEARCH RESULTS: {SEARCH_QUERY!r}",
        elements=matches,
    )

    print()
    print("=" * 90)
    print("MANUAL TEST COMPLETED")
    print("=" * 90)
    print(f"Full screenshot: {FULL_SCREEN_PATH}")
    print(
        "Full OCR visualization: "
        f"{FULL_SCREEN_VISUALIZATION_PATH}"
    )
    print(
        "Region OCR visualization: "
        f"{REGION_VISUALIZATION_PATH}"
    )
    print(f"Full-screen OCR elements: {len(elements)}")
    print(f"Region OCR elements: {len(region_elements)}")
    print(f"Search matches: {len(matches)}")


if __name__ == "__main__":
    run_manual_test()