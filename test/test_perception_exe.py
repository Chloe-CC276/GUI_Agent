"""
test_perception_execution

Perception:
- ScreenCapture
- ImageProcessor
- GUIElement
- PaddleOCREngine
- UIDetector
- PerceptionPipeline
- PerceptionResult

Execution:
- MouseController
- KeyboardController
- Action
- ActionSequence
- Executor
- ExecutionResult
- SequenceExecutionResult
"""


from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import pytest


# =====================================================================
# Project import path
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Project imports
# =====================================================================

from src.executor.action import (
    Action,
    ActionSequence,
    ActionStatus,
    ActionType,
    MouseButton,
)
from src.executor.executor import (
    ExecutionResult,
    Executor,
    SequenceExecutionResult,
)
from src.executor.keyboard import (
    KeyboardActionResult,
    KeyboardController,
)
from src.executor.mouse import (
    MouseActionResult,
    MouseController,
    MousePosition,
)
from src.perception.gui_element import GUIElement
from src.perception.image_preprocess import ImageProcessor
from src.perception.perception_pipeline import (
    PerceptionPipeline,
    PerceptionResult,
)
from src.perception.screen_capture import ScreenCapture
from src.perception.ui_detector import UIDetector


# =====================================================================
# Test configuration
# =====================================================================

OUTPUT_DIR = Path('D:/GUIAgent_project/screenshots/outputs')

RUN_REAL_TESTS = (
    os.getenv("GUI_AGENT_RUN_REAL_TESTS", "0").strip() == "1"
)

REAL_TEST_QUERY = os.getenv(
    "GUI_AGENT_REAL_TEST_QUERY",
    "File",
)

REAL_TEST_REGION = (60, 10, 880, 1010)


# =====================================================================
# Fake components 虚拟测试
# =====================================================================

class FakeScreenCapture:
    """
    Deterministic screen-capture substitute.

    It avoids taking a real screenshot during unit tests.
    """

    def __init__(
        self,
        image: np.ndarray,
    ) -> None:
        self.image = image.copy()

    def capture_screen(self) -> np.ndarray:
        return self.image.copy()

    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        if left < 0 or top < 0:
            raise ValueError("left and top must be non-negative.")

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")

        image_height, image_width = self.image.shape[:2]

        if left + width > image_width:
            raise ValueError("Region exceeds image width.")

        if top + height > image_height:
            raise ValueError("Region exceeds image height.")

        return self.image[
            top:top + height,
            left:left + width,
        ].copy()


class FakeOCREngine:
    """
    Predictable OCR substitute.

    It returns local coordinates relative to the supplied image.
    """

    def __init__(
        self,
        elements: Optional[Sequence[GUIElement]] = None,
    ) -> None:
        self.elements = list(
    [
        GUIElement(
            text="src",
            bbox=(50, 40, 120, 80),
            confidence=0.98,
            element_type="text",
            center=(85, 60),
        ),
        GUIElement(
            text="executor",
            bbox=(180, 100, 290, 140),
            confidence=0.96,
            element_type="text",
            center=(235, 120),
        ),
    ]
    if elements is None
    else elements
)
        

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> list[GUIElement]:
        threshold = (
            0.0
            if confidence_threshold is None
            else confidence_threshold
        )

        return [
            element
            for element in self.elements
            if element.confidence >= threshold
        ]

    def detect_text(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> list[str]:
        return [
            element.text
            for element in self.detect(
                image,
                confidence_threshold,
            )
        ]


class FakeUIDetector:
    """
    Predictable UI detector substitute.
    """

    def __init__(
        self,
        elements: Optional[Sequence[GUIElement]] = None,
    ) -> None:
        self.elements = list(
        [
            GUIElement(
                text="src",
                bbox=(40, 30, 135, 90),
                confidence=0.90,
                element_type="button",
                center=(87, 60),
            ),
            GUIElement(
                text="executor",
                bbox=(170, 90, 310, 155),
                confidence=0.88,
                element_type="button",
                center=(240, 122),
            ),
        ]
        if elements is None
        else elements
    )

    def detect(
        self,
        image: np.ndarray,
        ocr_elements: Optional[Sequence[GUIElement]] = None,
    ) -> list[GUIElement]:
        return list(self.elements)

    def detect_and_merge(
        self,
        image: np.ndarray,
        ocr_elements: Sequence[GUIElement],
        include_unmatched_ocr: bool = True,
    ) -> list[GUIElement]:
        result = list(self.elements)

        if include_unmatched_ocr:
            existing_texts = {
                element.text
                for element in result
            }

            result.extend(
                element
                for element in ocr_elements
                if element.text not in existing_texts
            )

        return result

    @staticmethod
    def find_by_text(
        elements: Sequence[GUIElement],
        query: str,
        exact_match: bool = False,
        case_sensitive: bool = False,
    ) -> list[GUIElement]:
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        expected = query.strip()

        if not case_sensitive:
            expected = expected.lower()

        matches: list[GUIElement] = []

        for element in elements:
            actual = element.text.strip()

            if not case_sensitive:
                actual = actual.lower()

            matched = (
                actual == expected
                if exact_match
                else expected in actual
            )

            if matched:
                matches.append(element)

        return matches

    @staticmethod
    def find_by_type(
        elements: Sequence[GUIElement],
        element_type: str,
    ) -> list[GUIElement]:
        if not element_type or not element_type.strip():
            raise ValueError("element_type must not be empty.")

        expected = element_type.strip().lower()

        return [
            element
            for element in elements
            if element.element_type.lower() == expected
        ]

    @staticmethod
    def visualize(
        image: np.ndarray,
        elements: Sequence[GUIElement],
        show_text: bool = True,
        show_confidence: bool = True,
        show_center: bool = True,
    ) -> np.ndarray:
        canvas = image.copy()

        if canvas.ndim == 2:
            canvas = cv2.cvtColor(
                canvas,
                cv2.COLOR_GRAY2BGR,
            )

        for element in elements:
            x1, y1, x2, y2 = element.bbox

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            if show_center and element.center is not None:
                cv2.circle(
                    canvas,
                    element.center,
                    3,
                    (0, 0, 255),
                    -1,
                )

        return canvas


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="session")
def output_dir() -> Path:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    return OUTPUT_DIR


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """
    Create a deterministic desktop-like test image.
    """

    image = np.full(
        (400, 640, 3),
        245,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (40, 30),
        (135, 90),
        (180, 180, 180),
        2,
    )

    cv2.putText(
        image,
        "src",
        (55, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        image,
        (170, 90),
        (310, 155),
        (180, 180, 180),
        2,
    )

    cv2.putText(
        image,
        "executor",
        (185, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        image,
        (350, 190),
        (590, 240),
        (130, 130, 130),
        2,
    )

    cv2.putText(
        image,
        "Input",
        (365, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return image


@pytest.fixture
def fake_pipeline(
    synthetic_image: np.ndarray,
) -> PerceptionPipeline:
    """
    Perception pipeline with deterministic fake components.
    """

    return PerceptionPipeline(
        screen_capture=FakeScreenCapture(
            synthetic_image
        ),
        image_processor=ImageProcessor(),
        ocr_engine=FakeOCREngine(),
        ui_detector=FakeUIDetector(),
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
        include_unmatched_ocr=True,
    )


@pytest.fixture
def dry_executor() -> Executor:
    """
    Safe executor that never controls the real desktop.
    """

    return Executor(
        dry_run=True,
        stop_on_failure=True,
        raise_on_error=False,
        default_wait_after_action=0.0,
        keep_history=True,
    )


# =====================================================================
# GUIElement and PerceptionResult tests
# =====================================================================

def test_gui_element_structure() -> None:
    element = GUIElement(
        text="Save",
        bbox=(10, 20, 100, 60),
        confidence=0.95,
        element_type="button",
        center=(55, 40),
    )

    assert element.text == "Save"
    assert element.bbox == (10, 20, 100, 60)
    assert element.confidence == pytest.approx(0.95)
    assert element.element_type == "button"
    assert element.center == (55, 40)


def test_perception_result_properties(
    synthetic_image: np.ndarray,
) -> None:
    text_element = GUIElement(
        text="src",
        bbox=(10, 10, 60, 40),
        confidence=0.90,
        element_type="text",
        center=(35, 25),
    )

    ui_element = GUIElement(
        text="Save",
        bbox=(70, 10, 130, 45),
        confidence=0.85,
        element_type="button",
        center=(100, 27),
    )

    result = PerceptionResult(
        original_image=synthetic_image,
        processed_image=synthetic_image.copy(),
        ocr_elements=[text_element],
        ui_elements=[ui_element],
        merged_elements=[
            text_element,
            ui_element,
        ],
        elapsed_time=0.1,
    )

    assert result.text_count == 1
    assert result.ui_count == 1
    assert result.element_count == 2

    assert result.get_texts() == [
        "src",
        "Save",
    ]

    buttons = result.get_elements_by_type(
        "button"
    )

    assert buttons == [ui_element]

    summary = result.summary()

    assert summary["ocr_element_count"] == 1
    assert summary["ui_element_count"] == 1
    assert summary["merged_element_count"] == 2
    assert summary["element_type_counts"]["button"] == 1


def test_perception_result_rejects_empty_type(
    synthetic_image: np.ndarray,
) -> None:
    result = PerceptionResult(
        original_image=synthetic_image,
        processed_image=synthetic_image.copy(),
    )

    with pytest.raises(ValueError):
        result.get_elements_by_type("")


# =====================================================================
# ImageProcessor tests
# =====================================================================

def test_image_processor_gray(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.gray(
        synthetic_image
    )

    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.shape == synthetic_image.shape[:2]


def test_image_processor_resize_width(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.resize(
        synthetic_image,
        width=320,
    )

    assert output.shape[1] == 320
    assert output.shape[0] == 200


def test_image_processor_gaussian(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.gaussian(
        synthetic_image,
        kernel=(5, 5),
    )

    assert output.shape == synthetic_image.shape


def test_image_processor_binary(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.binary(
        synthetic_image,
        threshold=150,
    )

    assert output.ndim == 2

    unique_values = set(
        np.unique(output).tolist()
    )

    assert unique_values.issubset(
        {0, 255}
    )


def test_image_processor_adaptive_binary(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.adaptive_binary(
        synthetic_image
    )

    assert output.ndim == 2
    assert output.dtype == np.uint8


def test_image_processor_clahe(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.clahe(
        synthetic_image
    )

    assert output.ndim == 2
    assert output.shape == synthetic_image.shape[:2]


def test_image_processor_sharpen(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.sharpen(
        synthetic_image
    )

    assert output.shape == synthetic_image.shape


def test_image_processor_complete_pipeline(
    synthetic_image: np.ndarray,
) -> None:
    processor = ImageProcessor()

    output = processor.process(
        synthetic_image,
        resize_width=320,
        use_gray=True,
        use_gaussian=True,
        use_median=False,
        use_binary=False,
        use_adaptive=False,
        use_clahe=True,
        use_sharpen=True,
    )

    assert output.ndim == 2
    assert output.shape[1] == 320


# =====================================================================
# PerceptionPipeline functional tests
# =====================================================================

def test_fake_pipeline_complete_run(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run()

    assert isinstance(result, PerceptionResult)
    assert len(result.ocr_elements) == 2
    assert len(result.ui_elements) == 2
    assert len(result.merged_elements) >= 2

    texts = result.get_texts()

    assert "src" in texts
    assert "executor" in texts


def test_fake_pipeline_ocr_only(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run(
        enable_ocr=True,
        enable_ui_detection=False,
        merge_results=True,
    )

    assert len(result.ocr_elements) == 2
    assert result.ui_elements == []
    assert result.merged_elements == result.ocr_elements


def test_fake_pipeline_ui_only(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run(
        enable_ocr=False,
        enable_ui_detection=True,
        merge_results=True,
    )

    assert result.ocr_elements == []
    assert len(result.ui_elements) == 2
    assert result.merged_elements == result.ui_elements


def test_fake_pipeline_all_detection_disabled(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run(
        enable_ocr=False,
        enable_ui_detection=False,
    )

    assert result.ocr_elements == []
    assert result.ui_elements == []
    assert result.merged_elements == []


def test_fake_pipeline_region_coordinate_mapping(
    fake_pipeline: PerceptionPipeline,
) -> None:
    region = (20, 30, 400, 250)

    result = fake_pipeline.capture_and_run(
        region=region,
    )

    assert result.capture_region == region

    for element in result.merged_elements:
        x1, y1, x2, y2 = element.bbox

        assert x1 >= 20
        assert y1 >= 30
        assert x2 > x1
        assert y2 > y1


def test_fake_pipeline_find_text(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run()

    matches = fake_pipeline.find_text(
        result=result,
        query="src",
        exact_match=True,
        case_sensitive=False,
    )

    assert len(matches) >= 1
    assert all(
        element.text.lower() == "src"
        for element in matches
    )


def test_fake_pipeline_find_best_text_match(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run()

    best = fake_pipeline.find_best_text_match(
        result=result,
        query="executor",
        exact_match=True,
        case_sensitive=False,
    )

    assert best is not None
    assert best.text == "executor"
    assert best.center is not None


def test_fake_pipeline_find_missing_text_returns_none(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run()

    best = fake_pipeline.find_best_text_match(
        result=result,
        query="not-existing-element",
        exact_match=True,
    )

    assert best is None


def test_fake_pipeline_find_by_type(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run()

    buttons = fake_pipeline.find_by_type(
        result=result,
        element_type="button",
    )

    assert len(buttons) == 2
    assert all(
        element.element_type == "button"
        for element in buttons
    )


def test_fake_pipeline_visualization(
    fake_pipeline: PerceptionPipeline,
) -> None:
    result = fake_pipeline.capture_and_run()

    image = fake_pipeline.visualize(
        result
    )

    assert isinstance(image, np.ndarray)
    assert image.size > 0


def test_fake_pipeline_save_result(
    fake_pipeline: PerceptionPipeline,
    output_dir: Path,
) -> None:
    result = fake_pipeline.capture_and_run()

    paths = fake_pipeline.save_result(
        result=result,
        output_dir=output_dir,
        prefix="fake_pipeline",
        save_original=True,
        save_processed=True,
        save_visualization=True,
    )

    assert {
        "original",
        "processed",
        "visualization",
    }.issubset(paths.keys())

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


# =====================================================================
# Perception robustness tests
# =====================================================================

@pytest.mark.parametrize(
    "invalid_region, expected_exception",
    [
        ((-1, 0, 100, 100), ValueError),
        ((0, -1, 100, 100), ValueError),
        ((0, 0, 0, 100), ValueError),
        ((0, 0, 100, 0), ValueError),
        ((0, 0, -1, 100), ValueError),
        ((0, 0, 100), ValueError),
    ],
)
def test_pipeline_invalid_regions(
    fake_pipeline: PerceptionPipeline,
    invalid_region: tuple,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        fake_pipeline.capture_and_run(
            region=invalid_region
        )


def test_pipeline_region_must_be_tuple(
    fake_pipeline: PerceptionPipeline,
) -> None:
    with pytest.raises(TypeError):
        fake_pipeline.capture_and_run(
            region=[0, 0, 100, 100]
        )


@pytest.mark.parametrize(
    "option_name",
    [
        "enable_preprocessing",
        "enable_ocr",
        "enable_ui_detection",
        "merge_results",
    ],
)
def test_pipeline_rejects_invalid_boolean_overrides(
    fake_pipeline: PerceptionPipeline,
    option_name: str,
) -> None:
    options = {
        option_name: "yes",
    }

    with pytest.raises(TypeError):
        fake_pipeline.capture_and_run(
            **options
        )


def test_pipeline_rejects_empty_image(
    fake_pipeline: PerceptionPipeline,
) -> None:
    with pytest.raises(ValueError):
        fake_pipeline.process_image(
            np.array([])
        )


def test_pipeline_rejects_invalid_image_type(
    fake_pipeline: PerceptionPipeline,
) -> None:
    with pytest.raises(TypeError):
        fake_pipeline.process_image(
            12345
        )


def test_pipeline_rejects_missing_image_file(
    fake_pipeline: PerceptionPipeline,
) -> None:
    with pytest.raises(FileNotFoundError):
        fake_pipeline.process_image(
            "missing-image-file.png"
        )


def test_pipeline_empty_ocr_result(
    synthetic_image: np.ndarray,
) -> None:
    pipeline = PerceptionPipeline(
        screen_capture=FakeScreenCapture(
            synthetic_image
        ),
        image_processor=ImageProcessor(),
        ocr_engine=FakeOCREngine(
            elements=[]
        ),
        ui_detector=FakeUIDetector(),
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    result = pipeline.capture_and_run()

    assert result.ocr_elements == []
    assert len(result.ui_elements) == 2


def test_pipeline_empty_ui_result(
    synthetic_image: np.ndarray,
) -> None:
    pipeline = PerceptionPipeline(
        screen_capture=FakeScreenCapture(
            synthetic_image
        ),
        image_processor=ImageProcessor(),
        ocr_engine=FakeOCREngine(),
        ui_detector=FakeUIDetector(
            elements=[]
        ),
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    result = pipeline.capture_and_run()

    assert result.ui_elements == []
    assert len(result.merged_elements) >= 1


# =====================================================================
# MouseController dry-run tests
# =====================================================================

def test_mouse_controller_dry_run_initialization() -> None:
    mouse = MouseController(
        dry_run=True,
        raise_on_error=False,
    )

    assert mouse.dry_run is True
    assert isinstance(
        mouse.get_screen_size(),
        tuple,
    )


def test_mouse_move_to_dry_run() -> None:
    mouse = MouseController(
        dry_run=True,
        raise_on_error=False,
    )

    width, height = mouse.get_screen_size()

    x = min(100, width - 1)
    y = min(100, height - 1)

    result = mouse.move_to(
        x=x,
        y=y,
    )

    assert result.success is True
    assert result.dry_run is True
    assert result.action == "move_to"
    assert result.end_position == MousePosition(
        x,
        y,
    )


def test_mouse_click_dry_run() -> None:
    mouse = MouseController(
        dry_run=True,
        raise_on_error=False,
    )

    width, height = mouse.get_screen_size()

    result = mouse.left_click(
        x=min(120, width - 1),
        y=min(120, height - 1),
    )

    assert result.success is True
    assert result.action == "click"
    assert result.metadata["button"] == "left"


def test_mouse_double_click_dry_run() -> None:
    mouse = MouseController(
        dry_run=True,
        raise_on_error=False,
    )

    width, height = mouse.get_screen_size()

    result = mouse.double_click(
        x=min(150, width - 1),
        y=min(150, height - 1),
    )

    assert result.success is True
    assert result.metadata["clicks"] == 2


def test_mouse_right_click_dry_run() -> None:
    mouse = MouseController(
        dry_run=True,
        raise_on_error=False,
    )

    width, height = mouse.get_screen_size()

    result = mouse.right_click(
        x=min(170, width - 1),
        y=min(170, height - 1),
    )

    assert result.success is True
    assert result.metadata["button"] == "right"


def test_mouse_scroll_dry_run() -> None:
    mouse = MouseController(
        dry_run=True,
        raise_on_error=False,
    )

    result = mouse.scroll(
        amount=-10,
    )

    assert result.success is True
    assert result.metadata["amount"] == -10


def test_mouse_drag_to_dry_run() -> None:
    mouse = MouseController(
        dry_run=True,
        raise_on_error=False,
    )

    width, height = mouse.get_screen_size()

    x = min(200, width - 1)
    y = min(200, height - 1)

    result = mouse.drag_to(
        x=x,
        y=y,
        duration=0.1,
    )

    assert result.success is True
    assert result.end_position == MousePosition(
        x,
        y,
    )


def test_mouse_normalised_coordinates() -> None:
    mouse = MouseController(
        dry_run=True,
    )

    width, height = mouse.get_screen_size()

    x, y = mouse.normalised_to_screen(
        0.5,
        0.5,
    )

    assert 0 <= x < width
    assert 0 <= y < height


@pytest.mark.parametrize(
    "x, y",
    [
        (-1, 10),
        (10, -1),
        (999999, 10),
        (10, 999999),
    ],
)
def test_mouse_rejects_out_of_screen_coordinates(
    x: int,
    y: int,
) -> None:
    mouse = MouseController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        mouse.validate_position(
            x,
            y,
        )


@pytest.mark.parametrize(
    "x, y",
    [
        (1.5, 10),
        (10, "20"),
        (True, 10),
    ],
)
def test_mouse_rejects_invalid_coordinate_types(
    x: Any,
    y: Any,
) -> None:
    mouse = MouseController(
        dry_run=True
    )

    with pytest.raises(TypeError):
        mouse.validate_position(
            x,
            y,
        )


def test_mouse_optional_position_requires_both_coordinates() -> None:
    mouse = MouseController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        mouse.click(
            x=100,
            y=None,
        )


def test_mouse_rejects_zero_scroll() -> None:
    mouse = MouseController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        mouse.scroll(0)


def test_mouse_rejects_invalid_button() -> None:
    mouse = MouseController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        mouse.click(
            button="invalid-button"
        )


# =====================================================================
# KeyboardController dry-run tests
# =====================================================================

def test_keyboard_press_dry_run() -> None:
    keyboard = KeyboardController(
        dry_run=True,
        raise_on_error=False,
    )

    result = keyboard.press(
        "enter"
    )

    assert result.success is True
    assert result.action == "press"
    assert result.metadata["key"] == "enter"


def test_keyboard_hotkey_dry_run() -> None:
    keyboard = KeyboardController(
        dry_run=True,
        raise_on_error=False,
    )

    result = keyboard.hotkey(
        "ctrl",
        "a",
    )

    assert result.success is True
    assert result.action == "hotkey"
    assert result.metadata["keys"] == (
        "ctrl",
        "a",
    )


def test_keyboard_type_text_dry_run() -> None:
    keyboard = KeyboardController(
        dry_run=True,
        raise_on_error=False,
    )

    result = keyboard.type_text(
        "GUI Agent"
    )

    assert result.success is True
    assert result.metadata["text"] == "GUI Agent"


def test_keyboard_common_shortcuts_dry_run() -> None:
    keyboard = KeyboardController(
        dry_run=True,
        raise_on_error=False,
    )

    results = [
        keyboard.copy(),
        keyboard.paste(),
        keyboard.cut(),
        keyboard.select_all(),
        keyboard.save(),
        keyboard.undo(),
        keyboard.redo(),
        keyboard.find(),
    ]

    assert all(
        result.success
        for result in results
    )


def test_keyboard_press_sequence_dry_run() -> None:
    keyboard = KeyboardController(
        dry_run=True,
        raise_on_error=False,
    )

    result = keyboard.press_sequence(
        [
            "down",
            "down",
            "enter",
        ]
    )

    assert result.success is True
    assert result.metadata["keys"] == (
        "down",
        "down",
        "enter",
    )


def test_keyboard_rejects_invalid_key() -> None:
    keyboard = KeyboardController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        keyboard.press(
            "not-a-real-key"
        )


def test_keyboard_rejects_empty_text() -> None:
    keyboard = KeyboardController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        keyboard.type_text("")


def test_keyboard_hotkey_requires_two_keys() -> None:
    keyboard = KeyboardController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        keyboard.hotkey(
            "ctrl"
        )


def test_keyboard_rejects_invalid_press_count() -> None:
    keyboard = KeyboardController(
        dry_run=True
    )

    with pytest.raises(ValueError):
        keyboard.press(
            "down",
            presses=0,
        )


# =====================================================================
# Action validation and serialization tests
# =====================================================================

def test_action_click_factory() -> None:
    action = Action.click(
        x=100,
        y=200,
        description="Click target",
    )

    assert action.type == ActionType.CLICK
    assert action.x == 100
    assert action.y == 200
    assert action.status == ActionStatus.PENDING


def test_action_from_dict() -> None:
    action = Action.from_dict(
        {
            "type": "double_click",
            "x": 100,
            "y": 200,
            "description": "Open folder",
        }
    )

    assert action.type == ActionType.DOUBLE_CLICK
    assert action.x == 100
    assert action.y == 200


def test_action_from_json() -> None:
    action = Action.from_json(
        """
        {
            "type": "press",
            "key": "enter",
            "presses": 1
        }
        """
    )

    assert action.type == ActionType.PRESS
    assert action.key == "enter"


def test_action_alias_normalisation() -> None:
    action = Action.from_dict(
        {
            "type": "doubleclick",
            "x": 10,
            "y": 20,
        }
    )

    assert action.type == ActionType.DOUBLE_CLICK


def test_action_to_json_round_trip() -> None:
    original = Action.type_text(
        text="GUI Agent",
        interval=0.05,
    )

    restored = Action.from_json(
        original.to_json()
    )

    assert restored.type == original.type
    assert restored.text == original.text
    assert restored.interval == original.interval


def test_action_unknown_fields_saved_in_metadata() -> None:
    action = Action.from_dict(
        {
            "type": "click",
            "x": 100,
            "y": 200,
            "model_reasoning": "Target detected.",
        }
    )

    assert (
        action.metadata["extra_fields"]["model_reasoning"]
        == "Target detected."
    )


def test_action_status_management() -> None:
    action = Action.click(
        x=100,
        y=100,
    )

    action.mark_running()
    assert action.status == ActionStatus.RUNNING

    action.mark_success()
    assert action.status == ActionStatus.SUCCESS

    action.mark_failed("test error")
    assert action.status == ActionStatus.FAILED
    assert action.metadata["error"] == "test error"

    action.mark_skipped("not required")
    assert action.status == ActionStatus.SKIPPED


@pytest.mark.parametrize(
    "data, exception_type",
    [
        (
            {
                "type": "move_to",
                "x": 10,
            },
            ValueError,
        ),
        (
            {
                "type": "scroll",
                "amount": 0,
            },
            ValueError,
        ),
        (
            {
                "type": "hotkey",
                "keys": ["ctrl"],
            },
            ValueError,
        ),
        (
            {
                "type": "type_text",
                "text": "",
            },
            ValueError,
        ),
        (
            {
                "type": "wait",
            },
            ValueError,
        ),
        (
            {
                "type": "unknown-action",
            },
            ValueError,
        ),
    ],
)
def test_action_rejects_invalid_parameters(
    data: dict[str, Any],
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        Action.from_dict(data)


def test_action_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        Action.from_json(
            "{not-valid-json}"
        )


# =====================================================================
# ActionSequence tests
# =====================================================================

def test_action_sequence_add_and_length() -> None:
    sequence = ActionSequence(
        description="Test sequence"
    )

    sequence.add(
        Action.click(
            x=10,
            y=20,
        )
    )

    sequence.add(
        Action.press_key(
            "enter"
        )
    )

    assert len(sequence) == 2


def test_action_sequence_serialization() -> None:
    sequence = ActionSequence(
        actions=[
            Action.click(
                x=10,
                y=20,
            ),
            Action.wait(0.1),
            Action.finish(),
        ]
    )

    restored = ActionSequence.from_json(
        sequence.to_json()
    )

    assert len(restored) == 3
    assert restored[0].type == ActionType.CLICK
    assert restored[2].type == ActionType.FINISH


def test_action_sequence_status_filters() -> None:
    first = Action.click(
        x=10,
        y=10,
    )

    second = Action.press_key(
        "enter"
    )

    third = Action.wait(0.1)

    first.mark_success()
    second.mark_failed("error")

    sequence = ActionSequence(
        actions=[
            first,
            second,
            third,
        ]
    )

    assert sequence.successful_actions() == [first]
    assert sequence.failed_actions() == [second]
    assert sequence.pending_actions() == [third]


def test_action_sequence_remove() -> None:
    action = Action.click(
        x=10,
        y=10,
    )

    sequence = ActionSequence(
        actions=[action]
    )

    removed = sequence.remove(
        action.action_id
    )

    assert removed is action
    assert len(sequence) == 0


def test_action_sequence_remove_missing_id() -> None:
    sequence = ActionSequence()

    with pytest.raises(KeyError):
        sequence.remove(
            "missing-id"
        )


# =====================================================================
# Executor dry-run tests
# =====================================================================

def test_executor_executes_mouse_action(
    dry_executor: Executor,
) -> None:
    width, height = (
        dry_executor.mouse.get_screen_size()
    )

    action = Action.click(
        x=min(100, width - 1),
        y=min(100, height - 1),
    )

    result = dry_executor.execute(
        action
    )

    assert isinstance(
        result,
        ExecutionResult,
    )

    assert result.success is True
    assert result.status == ActionStatus.SUCCESS
    assert result.mouse_result is not None
    assert action.status == ActionStatus.SUCCESS


def test_executor_executes_keyboard_action(
    dry_executor: Executor,
) -> None:
    action = Action.press_key(
        "enter"
    )

    result = dry_executor.execute(
        action
    )

    assert result.success is True
    assert result.keyboard_result is not None
    assert result.action_type == ActionType.PRESS


def test_executor_executes_dict(
    dry_executor: Executor,
) -> None:
    width, height = (
        dry_executor.mouse.get_screen_size()
    )

    result = dry_executor.execute(
        {
            "type": "move_to",
            "x": min(120, width - 1),
            "y": min(120, height - 1),
        }
    )

    assert result.success is True
    assert result.action.type == ActionType.MOVE_TO


def test_executor_executes_json(
    dry_executor: Executor,
) -> None:
    result = dry_executor.execute_json(
        """
        {
            "type": "hotkey",
            "keys": ["ctrl", "a"]
        }
        """
    )

    assert isinstance(
        result,
        ExecutionResult,
    )

    assert result.success is True


def test_executor_wait_dry_run(
    dry_executor: Executor,
) -> None:
    result = dry_executor.execute(
        Action.wait(1.0)
    )

    assert result.success is True
    assert result.metadata["dry_run"] is True
    assert result.elapsed_time < 0.5


def test_executor_finish_action(
    dry_executor: Executor,
) -> None:
    result = dry_executor.execute(
        Action.finish(
            "Complete"
        )
    )

    assert result.success is True
    assert result.message == "Complete"


def test_executor_fail_action(
    dry_executor: Executor,
) -> None:
    result = dry_executor.execute(
        Action.fail(
            "Intentional failure"
        )
    )

    assert result.success is False
    assert result.status == ActionStatus.FAILED
    assert result.error == "Intentional failure"


def test_executor_sequence_success(
    dry_executor: Executor,
) -> None:
    sequence = ActionSequence(
        actions=[
            Action.press_key("enter"),
            Action.hotkey_action(
                "ctrl",
                "a",
            ),
            Action.wait(0.1),
            Action.finish(),
        ]
    )

    result = dry_executor.execute_sequence(
        sequence
    )

    assert isinstance(
        result,
        SequenceExecutionResult,
    )

    assert result.success is True
    assert result.failed_actions == 0
    assert result.executed_actions == 4


def test_executor_sequence_stops_at_finish(
    dry_executor: Executor,
) -> None:
    sequence = ActionSequence(
        actions=[
            Action.press_key("enter"),
            Action.finish(),
            Action.press_key("escape"),
        ]
    )

    result = dry_executor.execute_sequence(
        sequence
    )

    assert result.executed_actions == 2
    assert result.stopped_early is True


def test_executor_sequence_stops_at_fail(
    dry_executor: Executor,
) -> None:
    sequence = ActionSequence(
        actions=[
            Action.press_key("enter"),
            Action.fail("Stop test"),
            Action.press_key("escape"),
        ]
    )

    result = dry_executor.execute_sequence(
        sequence
    )

    assert result.success is False
    assert result.executed_actions == 2
    assert result.stopped_early is True


def test_executor_history(
    dry_executor: Executor,
) -> None:
    dry_executor.clear_history()

    dry_executor.execute(
        Action.press_key("enter")
    )

    dry_executor.execute(
        Action.press_key("escape")
    )

    assert len(dry_executor.history) == 2

    summary = dry_executor.history_summary()

    assert summary["total_actions"] == 2
    assert summary["successful_actions"] == 2


def test_executor_stop_request(
    dry_executor: Executor,
) -> None:
    dry_executor.request_stop()

    result = dry_executor.execute(
        Action.press_key("enter")
    )

    assert result.success is False
    assert result.status == ActionStatus.SKIPPED

    dry_executor.reset_stop()


def test_executor_dry_run_switch(
    dry_executor: Executor,
) -> None:
    dry_executor.set_dry_run(False)

    assert dry_executor.dry_run is False
    assert dry_executor.mouse.dry_run is False
    assert dry_executor.keyboard.dry_run is False

    # Immediately restore safe mode.
    dry_executor.set_dry_run(True)

    assert dry_executor.dry_run is True


def test_executor_rejects_invalid_action_input(
    dry_executor: Executor,
) -> None:
    with pytest.raises(TypeError):
        dry_executor.execute(
            12345
        )


def test_executor_rejects_invalid_json(
    dry_executor: Executor,
) -> None:
    with pytest.raises(ValueError):
        dry_executor.execute_json(
            "{invalid}"
        )


# =====================================================================
# Perception-to-execution integration tests
# =====================================================================

def test_perception_target_to_click_action(
    fake_pipeline: PerceptionPipeline,
    dry_executor: Executor,
) -> None:
    perception_result = (
        fake_pipeline.capture_and_run()
    )

    target = (
        fake_pipeline.find_best_text_match(
            result=perception_result,
            query="src",
            exact_match=True,
            case_sensitive=False,
        )
    )

    assert target is not None
    assert target.center is not None

    action = Action.double_click(
        x=target.center[0],
        y=target.center[1],
        description="Open src folder",
    )

    execution_result = dry_executor.execute(
        action
    )

    assert execution_result.success is True
    assert execution_result.mouse_result is not None

    metadata = execution_result.mouse_result.metadata

    assert metadata["x"] == target.center[0]
    assert metadata["y"] == target.center[1]
    assert metadata["clicks"] == 2


def test_perception_target_to_right_click(
    fake_pipeline: PerceptionPipeline,
    dry_executor: Executor,
) -> None:
    perception_result = (
        fake_pipeline.capture_and_run()
    )

    target = (
        fake_pipeline.find_best_text_match(
            result=perception_result,
            query="executor",
            exact_match=True,
        )
    )

    assert target is not None
    assert target.center is not None

    action = Action.right_click(
        x=target.center[0],
        y=target.center[1],
        description="Open executor context menu",
    )

    result = dry_executor.execute(
        action
    )

    assert result.success is True
    assert result.mouse_result is not None
    assert (
        result.mouse_result.metadata["button"]
        == "right"
    )


def test_missing_perception_target_prevents_execution(
    fake_pipeline: PerceptionPipeline,
    dry_executor: Executor,
) -> None:
    perception_result = (
        fake_pipeline.capture_and_run()
    )

    target = (
        fake_pipeline.find_best_text_match(
            result=perception_result,
            query="missing-folder",
            exact_match=True,
        )
    )

    history_before = len(
        dry_executor.history
    )

    assert target is None

    # No action must be created or executed.
    history_after = len(
        dry_executor.history
    )

    assert history_after == history_before


def test_perception_action_sequence(
    fake_pipeline: PerceptionPipeline,
    dry_executor: Executor,
) -> None:
    perception_result = (
        fake_pipeline.capture_and_run()
    )

    src = fake_pipeline.find_best_text_match(
        result=perception_result,
        query="src",
        exact_match=True,
    )

    executor_target = (
        fake_pipeline.find_best_text_match(
            result=perception_result,
            query="executor",
            exact_match=True,
        )
    )

    assert src is not None
    assert executor_target is not None
    assert src.center is not None
    assert executor_target.center is not None

    sequence = ActionSequence(
        description=(
            "Synthetic perception-execution workflow"
        ),
        actions=[
            Action.double_click(
                x=src.center[0],
                y=src.center[1],
                description="Open src",
            ),
            Action.wait(0.1),
            Action.right_click(
                x=executor_target.center[0],
                y=executor_target.center[1],
                description="Open context menu",
            ),
            Action.press_key(
                "esc",
                description="Close context menu",
            ),
            Action.finish(),
        ],
    )

    result = dry_executor.execute_sequence(
        sequence
    )

    assert result.success is True
    assert result.executed_actions == 5


def test_region_perception_coordinates_to_action(
    fake_pipeline: PerceptionPipeline,
    dry_executor: Executor,
) -> None:
    region = (20, 30, 400, 250)

    perception_result = (
        fake_pipeline.capture_and_run(
            region=region
        )
    )

    target = (
        fake_pipeline.find_best_text_match(
            result=perception_result,
            query="src",
            exact_match=True,
        )
    )

    assert target is not None
    assert target.center is not None

    # Coordinates should include the region offset.
    assert target.center[0] >= region[0]
    assert target.center[1] >= region[1]

    action = Action.click(
        x=target.center[0],
        y=target.center[1],
    )

    result = dry_executor.execute(
        action
    )

    assert result.success is True


# =====================================================================
# Real desktop tests — disabled by default
# =====================================================================

@pytest.mark.real_gui
@pytest.mark.skipif(
    not RUN_REAL_TESTS,
    reason=(
        "Real desktop tests are disabled. "
        "Set GUI_AGENT_RUN_REAL_TESTS=1 to enable."
    ),
)
def test_real_screen_capture() -> None:
    capture = ScreenCapture()

    image = capture.capture_screen()

    assert isinstance(image, np.ndarray)
    assert image.size > 0
    assert image.ndim == 3


@pytest.mark.real_gui
@pytest.mark.skipif(
    not RUN_REAL_TESTS,
    reason=(
        "Real desktop tests are disabled. "
        "Set GUI_AGENT_RUN_REAL_TESTS=1 to enable."
    ),
)
def test_real_region_capture() -> None:
    capture = ScreenCapture()

    full_image = capture.capture_screen()

    image_height, image_width = (
        full_image.shape[:2]
    )

    left, top, width, height = (
        REAL_TEST_REGION
    )

    width = min(
        width,
        image_width - left,
    )

    height = min(
        height,
        image_height - top,
    )

    image = capture.capture_region(
        left=left,
        top=top,
        width=width,
        height=height,
    )

    assert image.shape[0] == height
    assert image.shape[1] == width


@pytest.mark.real_gui
@pytest.mark.skipif(
    not RUN_REAL_TESTS,
    reason=(
        "Real desktop tests are disabled. "
        "Set GUI_AGENT_RUN_REAL_TESTS=1 to enable."
    ),
)
def test_real_perception_pipeline() -> None:
    pipeline = PerceptionPipeline(
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
        include_unmatched_ocr=True,
    )

    result = pipeline.capture_and_run()

    assert isinstance(
        result,
        PerceptionResult,
    )

    assert result.original_image.size > 0
    assert isinstance(
        result.merged_elements,
        list,
    )

    print(
        "Real perception summary:",
        result.summary(),
    )


@pytest.mark.real_gui
@pytest.mark.skipif(
    not RUN_REAL_TESTS,
    reason=(
        "Real desktop tests are disabled. "
        "Set GUI_AGENT_RUN_REAL_TESTS=1 to enable."
    ),
)
def test_real_perception_text_query() -> None:
    pipeline = PerceptionPipeline(
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    result = pipeline.capture_and_run()

    matches = pipeline.find_text(
        result=result,
        query=REAL_TEST_QUERY,
        exact_match=False,
        case_sensitive=False,
    )

    print(
        f"Query={REAL_TEST_QUERY!r}, "
        f"matches={len(matches)}"
    )

    for element in matches:
        print(
            element.text,
            element.bbox,
            element.center,
            element.confidence,
        )


@pytest.mark.real_gui
@pytest.mark.skipif(
    not RUN_REAL_TESTS,
    reason=(
        "Real desktop tests are disabled. "
        "Set GUI_AGENT_RUN_REAL_TESTS=1 to enable."
    ),
)
def test_real_mouse_safe_move() -> None:
    """
    Real mouse movement only.

    It does not click, drag or modify files.
    """

    mouse = MouseController(
        dry_run=False,
        fail_safe=True,
        raise_on_error=True,
    )

    original = mouse.get_position()

    width, height = mouse.get_screen_size()

    target_x = min(
        max(100, width // 2),
        width - 1,
    )

    target_y = min(
        max(100, height // 2),
        height - 1,
    )

    move_result = mouse.move_to(
        x=target_x,
        y=target_y,
        duration=0.2,
    )

    assert move_result.success is True

    # Restore original position.
    mouse.move_to(
        x=original.x,
        y=original.y,
        duration=0.2,
    )


# =====================================================================
# Manual summary helper
# =====================================================================

def print_test_configuration() -> None:
    print()
    print("=" * 90)
    print("GUI Agent perception/executor test configuration")
    print("=" * 90)
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print(f"Real tests enabled: {RUN_REAL_TESTS}")
    print(f"Real query        : {REAL_TEST_QUERY}")
    print()


if __name__ == "__main__":
    print_test_configuration()

    raise SystemExit(
        pytest.main(
            [
                str(Path(__file__).resolve()),
                "-v",
                "-s",
            ]
        )
    )