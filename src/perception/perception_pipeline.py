"""
perception_pipeline
截取实时屏幕、OpenCV图片预处理、PaddleOCR文本检测与识别、
UI组件识别、UI组件和OCR文本合并、保存截图和处理留痕
"""


from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .gui_element import GUIElement
from .image_preprocess import ImageProcessor
from .paddle_ocr import PaddleOCREngine
from .screen_capture import ScreenCapture
from .ui_detector import UIDetector


logger = logging.getLogger(__name__)


@dataclass
class PerceptionResult:

    original_image: np.ndarray
    processed_image: np.ndarray
    ocr_elements: list[GUIElement] = field(default_factory=list)
    ui_elements: list[GUIElement] = field(default_factory=list)
    merged_elements: list[GUIElement] = field(default_factory=list)
    capture_region: Optional[tuple[int, int, int, int]] = None
    elapsed_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # 返回merged GUI elements数量
    @property
    def element_count(self) -> int:
        return len(self.merged_elements)

    # 返回OCR识别元素数量
    @property
    def text_count(self) -> int:
        return len(self.ocr_elements)

    # 返回UI组件数量
    @property
    def ui_count(self) -> int:
        return len(self.ui_elements)

    #返回非空文本
    def get_texts(self) -> list[str]:

        return [
            element.text
            for element in self.merged_elements
            if element.text.strip()
        ]

    # 根据组件类型筛选元素
    def get_elements_by_type(
        self,
        element_type: str,
    ) -> list[GUIElement]:
        
        if not element_type or not element_type.strip():
            raise ValueError("element_type must not be empty.")

        expected = element_type.strip().lower()

        return [
            element
            for element in self.merged_elements
            if element.element_type.lower() == expected
        ]

    def summary(self) -> dict[str, Any]:

        type_counts: dict[str, int] = {}

        for element in self.merged_elements:
            type_counts[element.element_type] = (
                type_counts.get(element.element_type, 0) + 1
            )

        return {
            "ocr_element_count": self.text_count,
            "ui_element_count": self.ui_count,
            "merged_element_count": self.element_count,
            "element_type_counts": type_counts,
            "capture_region": self.capture_region,
            "elapsed_time_seconds": self.elapsed_time,
            **self.metadata,
        }

    def target_candidates(self) -> list[dict[str, Any]]:
        """Return serializable element data used by Planner target matching."""
        candidates: list[dict[str, Any]] = []
        for index, element in enumerate(self.merged_elements):
            element_id = getattr(element, "element_id", None)
            candidates.append(
                {
                    "element_id": index if element_id is None else int(element_id),
                    "text": str(getattr(element, "text", "") or ""),
                    "source": str(getattr(element, "source", "") or ""),
                    "element_type": str(
                        getattr(element, "element_type", "") or ""
                    ),
                    "bbox": [
                        int(round(float(value)))
                        for value in element.bbox
                    ],
                    "center": [
                        int(round(float(value)))
                        for value in element.center
                    ],
                    "confidence": float(
                        getattr(element, "confidence", 0.0) or 0.0
                    ),
                }
            )
        return candidates
    

class PerceptionPipeline:
    def __init__(
        self,
        screen_capture: Optional[ScreenCapture] = None,
        image_processor: Optional[ImageProcessor] = None,
        ocr_engine: Optional[PaddleOCREngine] = None,
        ui_detector: Optional[UIDetector] = None,
        enable_preprocessing: bool = False,
        enable_ocr: bool = True,
        enable_ui_detection: bool = True,
        merge_results: bool = True,
        include_unmatched_ocr: bool = True,
        preprocess_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.screen_capture = screen_capture or ScreenCapture()
        self.image_processor = image_processor or ImageProcessor()

        self.ocr_engine = ocr_engine
        self.ui_detector = ui_detector

        self.enable_preprocessing = enable_preprocessing
        self.enable_ocr = enable_ocr
        self.enable_ui_detection = enable_ui_detection
        self.merge_results = merge_results
        self.include_unmatched_ocr = include_unmatched_ocr

        self.preprocess_options = preprocess_options or {
            "use_gray": False,
            "use_gaussian": False,
            "use_median": False,
            "use_binary": False,
            "use_adaptive": False,
            "use_clahe": False,
            "use_sharpen": False,
        }


    # ------------------------------------------------------------------
    # Main pipeline API 输出image和region,输出Perception Result
    # ------------------------------------------------------------------

    # 执行'图像获取->preprocessing->OCR->UI detection->merge result->result'
    def run(
        self,
        image: Optional[np.ndarray | str | Path] = None,
        region: Optional[tuple[int, int, int, int]] = None,
        enable_preprocessing: Optional[bool] = None,
        enable_ocr: Optional[bool] = None,
        enable_ui_detection: Optional[bool] = None,
        merge_results: Optional[bool] = None,
        preprocess_options: Optional[dict[str, Any]] = None,
    ) -> PerceptionResult:
        start_time = time.perf_counter()

        use_preprocessing = self._resolve_boolean(
            enable_preprocessing,
            self.enable_preprocessing,
        )
        use_ocr = self._resolve_boolean(
            enable_ocr,
            self.enable_ocr,
        )
        use_ui_detection = self._resolve_boolean(
            enable_ui_detection,
            self.enable_ui_detection,
        )
        use_merge = self._resolve_boolean(
            merge_results,
            self.merge_results,
        )

        original_image, coordinate_offset = self._obtain_image(
            image=image,
            region=region,
        )

        processed_image = self._process_image(
            image=original_image,
            enabled=use_preprocessing,
            preprocess_options=preprocess_options,
        )

        ocr_elements: list[GUIElement] = []

        if use_ocr:
            if self.ocr_engine is None:
                self.ocr_engine = PaddleOCREngine(
                    lang="ch",
                    confidence_threshold=0.50,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    use_gpu=False,
                    sort_results=True,
                )
            ocr_elements = self.ocr_engine.detect(processed_image)
            self._tag_source(ocr_elements, "ocr")

            if coordinate_offset != (0, 0):
                ocr_elements = self._offset_elements(
                    elements=ocr_elements,
                    offset_x=coordinate_offset[0],
                    offset_y=coordinate_offset[1],
                )

        ui_elements: list[GUIElement] = []

        if use_ui_detection:
            if self.ui_detector is None:
                self.ui_detector = UIDetector()
            local_ocr_elements = ocr_elements

            if coordinate_offset != (0, 0):
                local_ocr_elements = self._offset_elements(
                    elements=ocr_elements,
                    offset_x=-coordinate_offset[0],
                    offset_y=-coordinate_offset[1],
                )

            ui_elements = self.ui_detector.detect(
                image=processed_image,
                ocr_elements=local_ocr_elements,
            )
            self._tag_source(ui_elements, "ui")

            if coordinate_offset != (0, 0):
                ui_elements = self._offset_elements(
                    elements=ui_elements,
                    offset_x=coordinate_offset[0],
                    offset_y=coordinate_offset[1],
                )

        merged_elements = self._build_merged_elements(
            image=processed_image,
            ocr_elements=ocr_elements,
            ui_elements=ui_elements,
            coordinate_offset=coordinate_offset,
            use_ocr=use_ocr,
            use_ui_detection=use_ui_detection,
            use_merge=use_merge,
        )

        elapsed_time = time.perf_counter() - start_time

        metadata = {
            "preprocessing_enabled": use_preprocessing,
            "ocr_enabled": use_ocr,
            "ui_detection_enabled": use_ui_detection,
            "merge_enabled": use_merge,
            "image_shape": tuple(original_image.shape),
            "processed_image_shape": tuple(processed_image.shape),
        }

        result = PerceptionResult(
            original_image=original_image,
            processed_image=processed_image,
            ocr_elements=ocr_elements,
            ui_elements=ui_elements,
            merged_elements=merged_elements,
            capture_region=region,
            elapsed_time=elapsed_time,
            metadata=metadata,
        )
        result.metadata["target_candidates"] = result.target_candidates()

        logger.info(
            "Perception pipeline completed: OCR=%d, UI=%d, merged=%d, "
            "elapsed=%.3fs",
            len(ocr_elements),
            len(ui_elements),
            len(merged_elements),
            elapsed_time,
        )

        return result

    # 截取当前桌面并感知
    def capture_and_run(
        self,
        region: Optional[tuple[int, int, int, int]] = None,
        **run_options: Any,
    ) -> PerceptionResult:

        return self.run(
            image=None,
            region=region,
            **run_options,
        )


    def process_image(
        self,
        image: np.ndarray | str | Path,
        **run_options: Any,
    ) -> PerceptionResult:
        """
        Execute the pipeline on an existing image.
        """

        return self.run(
            image=image,
            **run_options,
        )
    

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def find_text(
        self,
        result: PerceptionResult,
        query: str,
        exact_match: bool = False,
        case_sensitive: bool = False,
    ) -> list[GUIElement]:
        """
        Find GUI elements by recognized text.
        """

        if not query or not query.strip():
            return []
        expected = (
            query.strip()
            if case_sensitive
            else query.strip().casefold()
        )
        matches: list[GUIElement] = []
        for element in result.merged_elements:
            labels = (
                getattr(element, "text", ""),
                getattr(element, "label", ""),
                getattr(element, "name", ""),
                getattr(element, "element_type", ""),
            )
            for label in labels:
                candidate = str(label or "").strip()
                if not case_sensitive:
                    candidate = candidate.casefold()
                matched = (
                    candidate == expected
                    if exact_match
                    else expected in candidate or candidate in expected
                )
                if candidate and matched:
                    matches.append(element)
                    break
        return matches

    def find_by_type(
        self,
        result: PerceptionResult,
        element_type: str,
    ) -> list[GUIElement]:
        """
        Find GUI elements by element type.
        """

        source_elements = (
            result.ui_elements
            if result.ui_elements
            else result.merged_elements
        )

        expected = element_type.strip().casefold()
        return [
            element
            for element in source_elements
            if str(getattr(element, "element_type", "")).casefold()
            == expected
        ]

    # 根据文本匹配与最高置信度寻找元素
    def find_best_text_match(
        self,
        result: PerceptionResult,
        query: str,
        exact_match: bool = False,
        case_sensitive: bool = False,
    ) -> Optional[GUIElement]:
        """
        Return the highest-confidence text match.

        Returns None when no result matches.
        """

        matches = self.find_text(
            result=result,
            query=query,
            exact_match=exact_match,
            case_sensitive=case_sensitive,
        )

        if not matches:
            return None

        return max(
            matches,
            key=lambda element: element.confidence,
        )


    # ------------------------------------------------------------------
    # Visualization API
    # ------------------------------------------------------------------

    def visualize(
        self,
        result: PerceptionResult,
        use_processed_image: bool = False,
        show_text: bool = True,
        show_confidence: bool = True,
        show_center: bool = True,
    ) -> np.ndarray:
        base_image = (
            result.processed_image
            if use_processed_image
            else result.original_image
        )

        elements = result.merged_elements

        if result.capture_region is not None:
            left, top, _, _ = result.capture_region

            elements = self._offset_elements(
                elements=elements,
                offset_x=-left,
                offset_y=-top,
            )

        return self.ui_detector.visualize(
            image=base_image,
            elements=elements,
            show_text=show_text,
            show_confidence=show_confidence,
            show_center=show_center,
        )
    

    def save_result(
        self,
        result: PerceptionResult,
        output_dir: str | Path,
        prefix: str = "perception",
        save_original: bool = True,
        save_processed: bool = True,
        save_visualization: bool = True,
    ) -> dict[str, Path]:
        
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}

        if save_original:
            original_path = destination / f"{prefix}_original.png"

            self._save_image(
                image=result.original_image,
                save_path=original_path,
            )

            saved_paths["original"] = original_path

        if save_processed:
            processed_path = destination / f"{prefix}_processed.png"

            self._save_image(
                image=result.processed_image,
                save_path=processed_path,
            )

            saved_paths["processed"] = processed_path

        if save_visualization:
            visualization = self.visualize(result)

            visualization_path = (
                destination / f"{prefix}_visualization.png"
            )

            self._save_image(
                image=visualization,
                save_path=visualization_path,
            )

            saved_paths["visualization"] = visualization_path

        return saved_paths


    # ------------------------------------------------------------------
    # Internal image workflow
    # ------------------------------------------------------------------

    def _obtain_image(
        self,
        image: Optional[np.ndarray | str | Path],
        region: Optional[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        
        if image is None:
            if region is None:
                captured = self.screen_capture.capture_screen()
                return self._prepare_image(captured), (0, 0)

            left, top, width, height = self._validate_region_tuple(region)

            captured = self.screen_capture.capture_region(
                left=left,
                top=top,
                width=width,
                height=height,
            )

            return self._prepare_image(captured), (left, top)

        loaded_image = self._prepare_image(image)

        if region is None:
            return loaded_image, (0, 0)

        left, top, width, height = self._validate_region_tuple(region)

        self._validate_region_inside_image(
            image=loaded_image,
            left=left,
            top=top,
            width=width,
            height=height,
        )

        right = left + width
        bottom = top + height

        cropped = loaded_image[top:bottom, left:right]

        if cropped.size == 0:
            raise ValueError("Selected pipeline region is empty.")

        return np.ascontiguousarray(cropped), (left, top)

    def _process_image(
        self,
        image: np.ndarray,
        enabled: bool,
        preprocess_options: Optional[dict[str, Any]],
    ) -> np.ndarray:
        """
        Apply optional OpenCV preprocessing.
        """

        if not enabled:
            return image.copy()

        options = dict(self.preprocess_options)

        if preprocess_options is not None:
            options.update(preprocess_options)

        return self.image_processor.process(
            image=image,
            **options,
        )

    def _build_merged_elements(
        self,
        image: np.ndarray,
        ocr_elements: Sequence[GUIElement],
        ui_elements: Sequence[GUIElement],
        coordinate_offset: tuple[int, int],
        use_ocr: bool,
        use_ui_detection: bool,
        use_merge: bool,
    ) -> list[GUIElement]:
        """
        Construct the final GUIElement result.
        """

        merged_elements = self._select_merge_sources(
            ocr_elements=ocr_elements,
            ui_elements=ui_elements,
            use_ocr=use_ocr,
            use_ui_detection=use_ui_detection,
            use_merge=use_merge,
        )
        return self._assign_element_ids(merged_elements)

    def _select_merge_sources(
        self,
        ocr_elements: Sequence[GUIElement],
        ui_elements: Sequence[GUIElement],
        use_ocr: bool,
        use_ui_detection: bool,
        use_merge: bool,
    ) -> list[GUIElement]:
        if not use_merge:
            if use_ui_detection:
                return list(ui_elements)

            if use_ocr:
                return list(ocr_elements)

            return []

        if use_ocr and use_ui_detection:
            merged_elements = list(ui_elements)

            if self.include_unmatched_ocr:
                existing_texts = {
                    element.text
                    for element in merged_elements
                    if element.text.strip()
                }

                merged_elements.extend(
                    element
                    for element in ocr_elements
                    if element.text.strip()
                    and element.text not in existing_texts
                )

            return merged_elements

        if use_ocr:
            return list(ocr_elements)

        if use_ui_detection:
            return list(ui_elements)

        return []

    @staticmethod
    def _tag_source(
        elements: Sequence[GUIElement],
        source: str,
    ) -> None:
        for element in elements:
            element.source = source

    @staticmethod
    def _assign_element_ids(
        elements: list[GUIElement],
    ) -> list[GUIElement]:
        """Number the final elements so Planner and prompt agree on the ids."""

        for index, element in enumerate(elements):
            element.element_id = index

        return elements
    

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _offset_elements(
        elements: Sequence[GUIElement],
        offset_x: int,
        offset_y: int,
    ) -> list[GUIElement]:
        """
        Add an x/y offset to GUIElement coordinates.
        """

        shifted: list[GUIElement] = []

        for element in elements:
            x1, y1, x2, y2 = element.bbox

            center = element.center

            shifted_center = (
                center[0] + offset_x,
                center[1] + offset_y,
            ) if center is not None else None

            shifted.append(
                GUIElement(
                    text=element.text,
                    bbox=(
                        x1 + offset_x,
                        y1 + offset_y,
                        x2 + offset_x,
                        y2 + offset_y,
                    ),
                    confidence=element.confidence,
                    element_type=element.element_type,
                    center=shifted_center,
                    element_id=element.element_id,
                    source=element.source,
                )
            )

        return shifted

    @staticmethod
    def _prepare_image(
        image: np.ndarray | str | Path,
    ) -> np.ndarray:
        """
        Load and validate an input image.
        """

        if isinstance(image, (str, Path)):
            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image does not exist: {image_path}"
                )

            loaded = cv2.imread(str(image_path))

            if loaded is None:
                raise ValueError(
                    f"OpenCV failed to load image: {image_path}"
                )

            return loaded

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray, str or pathlib.Path."
            )

        if image.size == 0:
            raise ValueError("image must not be empty.")

        if image.ndim not in (2, 3):
            raise ValueError(
                "image must be a two-dimensional or three-dimensional "
                "array."
            )

        prepared = image

        if prepared.dtype != np.uint8:
            prepared = np.nan_to_num(
                prepared,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            if np.issubdtype(prepared.dtype, np.floating):
                if float(np.max(prepared)) <= 1.0:
                    prepared = prepared * 255.0

            prepared = np.clip(
                prepared,
                0,
                255,
            ).astype(np.uint8)

        if prepared.ndim == 3 and prepared.shape[2] == 4:
            prepared = cv2.cvtColor(
                prepared,
                cv2.COLOR_BGRA2BGR,
            )

        if prepared.ndim == 3 and prepared.shape[2] == 1:
            prepared = prepared[:, :, 0]

        return np.ascontiguousarray(prepared)

    @staticmethod
    def _save_image(
        image: np.ndarray,
        save_path: str | Path,
    ) -> Path:
        """
        Save an image and validate the operation.
        """

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(path), image)

        if not success:
            raise IOError(f"Failed to save image to: {path}")

        return path

    @staticmethod
    def _resolve_boolean(
        override: Optional[bool],
        default: bool,
    ) -> bool:
        """
        Resolve an optional Boolean override.
        """

        if override is None:
            return default

        if not isinstance(override, bool):
            raise TypeError("Boolean pipeline options must be bool values.")

        return override

    @staticmethod
    def _validate_region_tuple(
        region: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """
        Validate a region tuple.
        """

        if not isinstance(region, tuple):
            raise TypeError(
                "region must be a tuple: "
                "(left, top, width, height)."
            )

        if len(region) != 4:
            raise ValueError(
                "region must contain exactly four values: "
                "(left, top, width, height)."
            )

        left, top, width, height = region

        for name, value in {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

        if left < 0 or top < 0:
            raise ValueError("left and top must be non-negative.")

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")

        return left, top, width, height

    @staticmethod
    def _validate_region_inside_image(
        image: np.ndarray,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> None:
        """
        Validate a region against image dimensions.
        """

        image_height, image_width = image.shape[:2]

        if left >= image_width or top >= image_height:
            raise ValueError(
                "Selected region starts outside the image."
            )

        if left + width > image_width:
            raise ValueError(
                "Selected region exceeds image width."
            )

        if top + height > image_height:
            raise ValueError(
                "Selected region exceeds image height."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"enable_preprocessing={self.enable_preprocessing}, "
            f"enable_ocr={self.enable_ocr}, "
            f"enable_ui_detection={self.enable_ui_detection}, "
            f"merge_results={self.merge_results}, "
            f"include_unmatched_ocr={self.include_unmatched_ocr}"
            f")"
        )