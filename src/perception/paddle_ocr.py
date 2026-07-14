"""
paddle_ocr
文本检测与识别、识别GUI元素、结构化输出

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

from paddleocr import PaddleOCR

from .base_ocr import BaseOCREngine
from .gui_element import GUIElement

logger = logging.getLogger(__name__)

class PaddleOCREngine(BaseOCREngine):
    """
    参数：
    lang: ch/ en,
    confidence_threshold:识别最小可返回置信区间
    use_doc_orientation_classify:纠正文档方向
    use_doc_unwarping:文档去畸变
    use_textline_orientation:检测文本方向
    use_gpu
    device: cpu/ gpu
    sort_results:检测到的元素是否应按大致的阅读顺序排序，从上到下，从左到右
    extra_options
    """
    
    def __init__(
        self,
        lang: str = "ch",
        confidence_threshold: float = 0.50,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        use_gpu: bool = False,
        device: Optional[str] = None,
        sort_results: bool = True,
        **extra_options: Any,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )

        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self.sort_results = sort_results

        self._ocr = self._create_engine(
            lang=lang,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_gpu=use_gpu,
            device=device,
            extra_options=extra_options,
        )


    # 引擎初始化，先配置最新版本，失败切换旧版本
    @staticmethod
    def _create_engine(
        lang: str,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
        use_gpu: bool,
        device: Optional[str],
        extra_options: dict[str, Any],
    ) -> PaddleOCR:

        resolved_device = device or ("gpu:0" if use_gpu else "cpu")

        modern_options: dict[str, Any] = {
            "lang": lang,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
            "device": resolved_device,
        }
        modern_options.update(extra_options)

        try:
            logger.info(
                "Initializing PaddleOCR with PaddleOCR 3.x options: %s",
                modern_options,
            )
            return PaddleOCR(**modern_options)

        except (TypeError, ValueError) as modern_error:
            logger.warning(
                "PaddleOCR 3.x initialization failed. "
                "Trying legacy PaddleOCR options. Reason: %s",
                modern_error,
            )

        legacy_options: dict[str, Any] = {
            "lang": lang,
            "use_angle_cls": use_textline_orientation,
            "use_gpu": use_gpu,
            "show_log": False,
        }

        # Remove options that only belong to the newer pipeline.
        incompatible_keys = {
            "device",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
        }

        for key, value in extra_options.items():
            if key not in incompatible_keys:
                legacy_options[key] = value

        try:
            logger.info(
                "Initializing PaddleOCR with legacy options: %s",
                legacy_options,
            )
            return PaddleOCR(**legacy_options)

        except Exception as legacy_error:
            raise RuntimeError(
                "Failed to initialize PaddleOCR using both the modern "
                "and legacy APIs. Check that paddleocr and paddlepaddle "
                "are installed with compatible versions."
            ) from legacy_error
    

    # ------------------------------------------------------------------
    # Public OCR API
    # ------------------------------------------------------------------     
    # 检测模块，传入image array和置信阈，return list[GUIElement]
    def detect(
        self,
        image: np.ndarray | str | Path,
        confidence_threshold: Optional[float] = None,
    ) -> list[GUIElement]:

        threshold = self._resolve_threshold(confidence_threshold)
        prepared_image = self._prepare_input(image)

        raw_result = self._run_paddleocr(prepared_image)

        elements = self._parse_result(
            raw_result=raw_result,
            confidence_threshold=threshold,
            offset_x=0,
            offset_y=0,
        )

        if self.sort_results:
            elements = self._sort_elements(elements)

        return elements


    # 识别模块，传入image array和置信阈，return list[GUIElement]
    def recognize(
        self,
        image: np.ndarray | str | Path,
        confidence_threshold: Optional[float] = None,
    ) -> list[GUIElement]:

        return self.detect(
            image=image,
            confidence_threshold=confidence_threshold,
        )
    

    # 检测指定区间，传入image array和区间信息，return list[GUIElement]
    def detect_region(
        self,
        image: np.ndarray | str | Path,
        left: int,
        top: int,
        width: int,
        height: int,
        confidence_threshold: Optional[float] = None,
    ) -> list[GUIElement]:

        full_image = self._prepare_input(image)
        self._validate_region(
            image=full_image,
            left=left,
            top=top,
            width=width,
            height=height,
        )

        right = left + width
        bottom = top + height

        cropped = full_image[top:bottom, left:right]

        if cropped.size == 0:
            raise ValueError("The selected OCR region is empty.")

        threshold = self._resolve_threshold(confidence_threshold)
        raw_result = self._run_paddleocr(cropped)

        elements = self._parse_result(
            raw_result=raw_result,
            confidence_threshold=threshold,
            offset_x=left,
            offset_y=top,
        )

        if self.sort_results:
            elements = self._sort_elements(elements)

        return elements
    
    # 识别文本，传入image array和置信阈，return list[str]
    def detect_text(
        self,
        image: np.ndarray | str | Path,
        confidence_threshold: Optional[float] = None,
    ) -> list[str]:

        return [
            element.text
            for element in self.detect(
                image=image,
                confidence_threshold=confidence_threshold,
            )
        ]
    

    # 搜索查询文本，传入image array/ query/ case_sensitive/ exact_match,return list[GUIElement]
    def find_text(
        self,
        image: np.ndarray | str | Path,
        query: str,
        confidence_threshold: Optional[float] = None,
        case_sensitive: bool = False,
        exact_match: bool = False,
    ) -> list[GUIElement]:
        """
        Find OCR elements whose text matches a query.

        Parameters
        ----------
        query:
            Text to search for.

        case_sensitive:
            Whether character case must match.

        exact_match:
            If True, use exact equality. If False, use substring matching.

        Returns
        -------
        list[GUIElement]
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        elements = self.detect(
            image=image,
            confidence_threshold=confidence_threshold,
        )

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

       
    # ------------------------------------------------------------------
    # PaddleOCR inference
    # ------------------------------------------------------------------
    def _run_paddleocr(self, image: np.ndarray) -> Any:
        """
        Run PaddleOCR using the new ``predict`` API where available.

        If the installed version does not expose a usable predict method,
        it falls back to the legacy ``ocr`` method.
        """

        predict_method = getattr(self._ocr, "predict", None)

        if callable(predict_method):
            try:
                return list(predict_method(image))
            except Exception as predict_error:
                logger.warning(
                    "PaddleOCR predict() failed; falling back to ocr(). "
                    "Reason: %s",
                    predict_error,
                )

        ocr_method = getattr(self._ocr, "ocr", None)

        if not callable(ocr_method):
            raise RuntimeError(
                "The installed PaddleOCR object provides neither a usable "
                "predict() method nor an ocr() method."
            )

        try:
            return ocr_method(image, cls=False)
        except TypeError:
            # Some versions no longer accept cls.
            return ocr_method(image)
        

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------
    def _parse_result(
        self,
        raw_result: Any,
        confidence_threshold: float,
        offset_x: int,
        offset_y: int,
    ) -> list[GUIElement]:
        """
        Supported patterns:

        1.PaddleOCR 3.x result dictionaries:
            {
                "rec_texts": [...],
                "rec_scores": [...],
                "dt_polys": [...]
            }

        2.Legacy PaddleOCR:
            [
                [
                    [polygon, (text, score)],
                    ...
                ]
            ]
        """

        if raw_result is None:
            return []

        elements: list[GUIElement] = []

        for result_item in self._iter_top_level_results(raw_result):
            parsed = self._parse_modern_result(
                result_item=result_item,
                confidence_threshold=confidence_threshold,
                offset_x=offset_x,
                offset_y=offset_y,
            )

            if parsed is not None:
                elements.extend(parsed)
                continue

            legacy_elements = self._parse_legacy_result(
                result_item=result_item,
                confidence_threshold=confidence_threshold,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            elements.extend(legacy_elements)

        return elements


    # 确保以list/ tuple返回可迭代对象
    @staticmethod
    def _iter_top_level_results(raw_result: Any) -> Iterable[Any]:

        if isinstance(raw_result, (list, tuple)):
            return raw_result

        return [raw_result]


    # 解析现代化版本
    def _parse_modern_result(
        self,
        result_item: Any,
        confidence_threshold: float,
        offset_x: int,
        offset_y: int,
    ) -> Optional[list[GUIElement]]:
        """
        Parse PaddleOCR 3.x-style structured results.

        Returns None when the item does not look like a modern result.
        """

        data = self._extract_result_mapping(result_item)

        if data is None:
            return None

        # 同义词映射
        texts = self._first_available(
            data,
            "rec_texts",
            "texts",
            "text",
        )
        scores = self._first_available(
            data,
            "rec_scores",
            "scores",
            "confidence",
            "confidences",
        )
        polygons = self._first_available(
            data,
            "dt_polys",
            "rec_polys",
            "polys",
            "boxes",
            "text_boxes",
        )

        if texts is None or polygons is None:
            return None

        # 规范化，补全默认值
        texts = self._to_list(texts)
        polygons = self._to_list(polygons)

        if scores is None:
            scores = [1.0] * len(texts)
        else:
            scores = self._to_list(scores)


        # min(len())安全对齐，confidence_threshold过滤杂质，offset_x/y坐标平移
        item_count = min(len(texts), len(scores), len(polygons))
        elements: list[GUIElement] = []

        for index in range(item_count):
            text = str(texts[index]).strip()
            score = self._safe_float(scores[index], default=0.0)

            if not text or score < confidence_threshold:
                continue

            polygon = polygons[index]
            element = self._build_gui_element(
                text=text,
                confidence=score,
                polygon=polygon,
                offset_x=offset_x,
                offset_y=offset_y,
            )

            if element is not None:
                elements.append(element)

        return elements
    

    # 解析旧版本
    def _parse_legacy_result(
        self,
        result_item: Any,
        confidence_threshold: float,
        offset_x: int,
        offset_y: int,
    ) -> list[GUIElement]:
        """
        Parse legacy PaddleOCR list output.
        """

        # 展平嵌套列表->一维单行列表
        lines = self._flatten_legacy_lines(result_item)
        elements: list[GUIElement] = []

        for line in lines:
            parsed_line = self._parse_legacy_line(line)

            if parsed_line is None:
                continue

            polygon, text, confidence = parsed_line

            if not text or confidence < confidence_threshold:
                continue
            
            # 解析GUI元素
            element = self._build_gui_element(
                text=text,
                confidence=confidence,
                polygon=polygon,
                offset_x=offset_x,
                offset_y=offset_y,
            )

            if element is not None:
                elements.append(element)

        return elements
    


    @staticmethod
    def _extract_result_mapping(
        result_item: Any,
    ) -> Optional[dict[str, Any]]:
        """
        Extract a dictionary from a PaddleOCR 3.x result object.
        """

        if isinstance(result_item, dict):
            return result_item

        json_attribute = getattr(result_item, "json", None)

        if isinstance(json_attribute, dict):
            # Some result objects expose {"res": {...}}.
            if "res" in json_attribute and isinstance(
                json_attribute["res"], dict
            ):
                return json_attribute["res"]

            return json_attribute

        if callable(json_attribute):
            try:
                value = json_attribute()

                if isinstance(value, dict):
                    if "res" in value and isinstance(value["res"], dict):
                        return value["res"]

                    return value
            except Exception:
                pass

        result_attribute = getattr(result_item, "res", None)

        if isinstance(result_attribute, dict):
            return result_attribute

        to_dict_method = getattr(result_item, "to_dict", None)

        if callable(to_dict_method):
            try:
                value = to_dict_method()

                if isinstance(value, dict):
                    return value
            except Exception:
                pass

        return None


    @staticmethod
    def _flatten_legacy_lines(result_item: Any) -> list[Any]:
        """
        Flatten common legacy PaddleOCR output nesting.
        """

        if result_item is None:
            return []

        if not isinstance(result_item, (list, tuple)):
            return []

        if PaddleOCREngine._looks_like_legacy_line(result_item):
            return [result_item]

        lines: list[Any] = []

        for child in result_item:
            if PaddleOCREngine._looks_like_legacy_line(child):
                lines.append(child)

            elif isinstance(child, (list, tuple)):
                lines.extend(
                    PaddleOCREngine._flatten_legacy_lines(child)
                )

        return lines


    # 判断是否符合识别标准
    @staticmethod
    def _looks_like_legacy_line(value: Any) -> bool:
        """
        Determine whether a value resembles:

        [polygon, (text, confidence)]
        """

        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False

        polygon = value[0]
        text_info = value[1]

        polygon_array = np.asarray(polygon)

        if polygon_array.ndim != 2 or polygon_array.shape[1] < 2:
            return False

        if not isinstance(text_info, (list, tuple)):
            return False

        return len(text_info) >= 2


    # 格式化抽离数据
    @staticmethod
    def _parse_legacy_line(
        line: Any,
    ) -> Optional[tuple[Any, str, float]]:
        """
        Parse one legacy PaddleOCR line.
        """

        try:
            polygon = line[0]
            text_info = line[1]

            text = str(text_info[0]).strip()
            confidence = float(text_info[1])

            return polygon, text, confidence

        except (IndexError, TypeError, ValueError):
            return None



    # ------------------------------------------------------------------
    # GUIElement conversion
    # ------------------------------------------------------------------
    # 多边形定位点输出成GUIElement
    @staticmethod
    def _build_gui_element(
        text: str,
        confidence: float,
        polygon: Any,
        offset_x: int,
        offset_y: int,
    ) -> Optional[GUIElement]:
        """
        Convert polygon coordinates into a GUIElement.
        """

        try:
            points = np.asarray(polygon, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if points.ndim != 2 or points.shape[0] == 0:
            return None

        if points.shape[1] < 2:
            return None

        x_values = points[:, 0]
        y_values = points[:, 1]

        x1 = int(np.floor(np.min(x_values))) + offset_x
        y1 = int(np.floor(np.min(y_values))) + offset_y
        x2 = int(np.ceil(np.max(x_values))) + offset_x
        y2 = int(np.ceil(np.max(y_values))) + offset_y

        if x2 <= x1 or y2 <= y1:
            return None

        center_x = int(round((x1 + x2) / 2))
        center_y = int(round((y1 + y2) / 2))

        return GUIElement(
            text=text,
            bbox=(x1, y1, x2, y2),
            confidence=float(confidence),
            element_type="text",
            center=(center_x, center_y),
        )
    

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        image: np.ndarray | str | Path,
        elements: Sequence[GUIElement],
        show_text: bool = True,
        show_confidence: bool = True,
        show_center: bool = True,
        box_color: tuple[int, int, int] = (0, 255, 0),
        text_color: tuple[int, int, int] = (0, 0, 255),
        center_color: tuple[int, int, int] = (255, 0, 0),
        line_thickness: int = 2,
        font_scale: float = 0.55,
    ) -> np.ndarray:
        """
        Draw OCR results on an image.

        Returns a new image and does not modify the input image.
        """

        canvas = self._prepare_input(image).copy()

        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for element in elements:
            x1, y1, x2, y2 = element.bbox

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                box_color,
                line_thickness,
            )

            if show_center:
                center = element.center

                if center is None:
                    center = (
                        int((x1 + x2) / 2),
                        int((y1 + y2) / 2),
                    )

                cv2.circle(
                    canvas,
                    center,
                    radius=4,
                    color=center_color,
                    thickness=-1,
                )

            if show_text:
                label = element.text

                if show_confidence:
                    label = (
                        f"{element.text} "
                        f"({element.confidence:.2f})"
                    )

                text_y = max(y1 - 8, 20)

                cv2.putText(
                    canvas,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        return canvas
    

    @staticmethod
    def save_visualization(
        image: np.ndarray,
        save_path: str | Path,
    ) -> Path:
        """
        Save a visualization image.
        """

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), image)

        if not success:
            raise IOError(
                f"OpenCV failed to save image to: {output_path}"
            )

        return output_path
    

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    # image清洗与标准化
    @staticmethod
    def _prepare_input(
        image: np.ndarray | str | Path,
    ) -> np.ndarray:
        """
        Validate and normalize an OCR input image.
        """

        if isinstance(image, (str, Path)):
            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image file does not exist: {image_path}"
                )

            loaded = cv2.imread(str(image_path))

            if loaded is None:
                raise ValueError(
                    f"OpenCV could not read image: {image_path}"
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
                "image must be a 2-D grayscale or 3-D colour array."
            )

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(
                "colour image must have 1, 3 or 4 channels."
            )

        result = image

        if result.dtype != np.uint8:
            result = PaddleOCREngine._to_uint8(result)

        if result.ndim == 3 and result.shape[2] == 4:
            result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        elif result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]

        return np.ascontiguousarray(result)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """
        Convert numeric image data into uint8.
        """

        if not np.issubdtype(image.dtype, np.number):
            raise TypeError("image array must contain numeric values.")

        finite_image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        if np.issubdtype(finite_image.dtype, np.floating):
            maximum = float(np.max(finite_image))

            if maximum <= 1.0:
                finite_image = finite_image * 255.0

        return np.clip(finite_image, 0, 255).astype(np.uint8)


    # 置信阈校对
    def _resolve_threshold(
        self,
        confidence_threshold: Optional[float],
    ) -> float:
        """
        Resolve and validate a confidence threshold.
        """

        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )

        return float(threshold)
    

    @staticmethod
    def _validate_region(
        image: np.ndarray,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> None:
        """
        Validate a crop region against an image.
        """

        values = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        for name, value in values.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

        if left < 0 or top < 0:
            raise ValueError("left and top must be non-negative.")

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")

        image_height, image_width = image.shape[:2]

        if left >= image_width or top >= image_height:
            raise ValueError(
                "The region starts outside the image boundaries."
            )

        if left + width > image_width:
            raise ValueError(
                "The region extends beyond the image width."
            )

        if top + height > image_height:
            raise ValueError(
                "The region extends beyond the image height."
            )


    @staticmethod
    def _sort_elements(
        elements: Sequence[GUIElement],
        row_tolerance: int = 15,
    ) -> list[GUIElement]:
        """
        Sort elements approximately from top to bottom and left to right.

        The y coordinate is grouped using a configurable tolerance so that
        text elements on the same row are primarily sorted by x position.
        """

        def sort_key(element: GUIElement) -> tuple[int, int]:
            x1, y1, _, _ = element.bbox
            row_group = int(round(y1 / max(row_tolerance, 1)))
            return row_group, x1

        return sorted(elements, key=sort_key)
    

    @staticmethod
    def _first_available(
        mapping: dict[str, Any],
        *keys: str,
    ) -> Any:
        """
        Return the first present non-None mapping value.
        """

        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]

        return None


    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        """
        Convert arrays and sequences into a Python list.
        """

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        return [value]


    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value into float.
        """

        try:
            return float(value)
        except (TypeError, ValueError):
            return default


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lang={self.lang!r}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"use_gpu={self.use_gpu}, "
            f"sort_results={self.sort_results}"
            f")"
        )