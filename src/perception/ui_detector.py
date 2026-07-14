"""
ui_detector
检测GUI元素（OPenCV）
检测按钮、输入框、复选框、图标...

"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .gui_element import GUIElement


logger = logging.getLogger(__name__)


class UIDetector:
    """
    参数：
    min_width：检测框的最小宽度
    min_height
    max_width_ratio：根据image宽度的最大宽度
    max_height_ratio
    min_area
    max_area_ratio
    rectangularity_threshold：轮廓面积与边框面积的最小比率
    duplicate_iou_threshold
    use_adaptive_threshold：边缘检测
    """

    def __init__(
        self,
        min_width: int = 12,
        min_height: int = 12,
        max_width_ratio: float = 0.95,
        max_height_ratio: float = 0.95,
        min_area: int = 120,
        max_area_ratio: float = 0.80,
        rectangularity_threshold: float = 0.45,
        duplicate_iou_threshold: float = 0.75,
        use_adaptive_threshold: bool = True,
    ) -> None:
        if min_width <= 0 or min_height <= 0:
            raise ValueError("min_width and min_height must be positive.")

        if min_area <= 0:
            raise ValueError("min_area must be positive.")

        for name, value in {
            "max_width_ratio": max_width_ratio,
            "max_height_ratio": max_height_ratio,
            "max_area_ratio": max_area_ratio,
            "rectangularity_threshold": rectangularity_threshold,
            "duplicate_iou_threshold": duplicate_iou_threshold,
        }.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in the range (0, 1].")

        self.min_width = min_width
        self.min_height = min_height
        self.max_width_ratio = max_width_ratio
        self.max_height_ratio = max_height_ratio
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.rectangularity_threshold = rectangularity_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
    

     # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # 参数：image，ocr_element,返回list[GUIElement]
    def detect(
        self,
        image: np.ndarray | str | Path,
        ocr_elements: Optional[Sequence[GUIElement]] = None,
    ) -> list[GUIElement]:
        prepared_image = self._prepare_image(image)
        height, width = prepared_image.shape[:2]

        candidate_boxes = self._detect_candidate_boxes(prepared_image)

        elements: list[GUIElement] = []

        for bbox in candidate_boxes:
            x1, y1, x2, y2 = bbox
            box_width = x2 - x1
            box_height = y2 - y1

            related_ocr = self._find_ocr_inside_box(
                bbox=bbox,
                ocr_elements=ocr_elements or [],
            )

            text = self._merge_ocr_text(related_ocr)

            element_type = self._classify_element(
                bbox=bbox,
                image_width=width,
                image_height=height,
                related_ocr=related_ocr,
                image=prepared_image,
            )

            confidence = self._calculate_confidence(
                bbox=bbox,
                element_type=element_type,
                related_ocr=related_ocr,
                image_width=width,
                image_height=height,
            )

            center = (
                int(round((x1 + x2) / 2)),
                int(round((y1 + y2) / 2)),
            )

            elements.append(
                GUIElement(
                    text=text,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    element_type=element_type,
                    center=center,
                )
            )

        elements = self._remove_duplicates(elements)
        elements = self._remove_boxes_containing_only_other_boxes(elements)
        elements = self._sort_elements(elements)

        return elements
    

    # 检测UI元素与OCR结果合并
    def detect_and_merge(
        self,
        image: np.ndarray | str | Path,
        ocr_elements: Sequence[GUIElement],
        include_unmatched_ocr: bool = True,
    ) -> list[GUIElement]:
        ui_elements = self.detect(
            image=image,
            ocr_elements=ocr_elements,
        )

        if not include_unmatched_ocr:
            return ui_elements

        unmatched_ocr: list[GUIElement] = []

        for ocr_element in ocr_elements:
            matched = any(
                self._contains(
                    outer=ui_element.bbox,
                    inner=ocr_element.bbox,
                    tolerance=4,
                )
                for ui_element in ui_elements
            )

            if not matched:
                unmatched_ocr.append(ocr_element)

        combined = [*ui_elements, *unmatched_ocr]
        combined = self._remove_duplicates(combined)
        combined = self._sort_elements(combined)

        return combined
    

    def detect_region(
        self,
        image: np.ndarray | str | Path,
        left: int,
        top: int,
        width: int,
        height: int,
        ocr_elements: Optional[Sequence[GUIElement]] = None,
    ) -> list[GUIElement]:
        full_image = self._prepare_image(image)

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

        local_ocr: list[GUIElement] = []

        for element in ocr_elements or []:
            if self._contains(
                outer=(left, top, right, bottom),
                inner=element.bbox,
                tolerance=0,
            ):
                x1, y1, x2, y2 = element.bbox

                local_ocr.append(
                    replace(
                        element,
                        bbox=(
                            x1 - left,
                            y1 - top,
                            x2 - left,
                            y2 - top,
                        ),
                        center=(
                            element.center[0] - left,
                            element.center[1] - top,
                        )
                        if element.center is not None
                        else None,
                    )
                )

        local_elements = self.detect(
            image=cropped,
            ocr_elements=local_ocr,
        )

        mapped_elements: list[GUIElement] = []

        for element in local_elements:
            x1, y1, x2, y2 = element.bbox

            mapped_bbox = (
                x1 + left,
                y1 + top,
                x2 + left,
                y2 + top,
            )

            mapped_center = (
                element.center[0] + left,
                element.center[1] + top,
            ) if element.center is not None else None

            mapped_elements.append(
                replace(
                    element,
                    bbox=mapped_bbox,
                    center=mapped_center,
                )
            )

        return mapped_elements
    

    # 根据文本示意识别UI
    def find_by_text(
        self,
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

            if exact_match:
                matched = actual == expected
            else:
                matched = expected in actual

            if matched:
                matches.append(element)

        return matches
    

    def find_by_type(
        self,
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
    

    # ------------------------------------------------------------------
    # Candidate detection 候选框检测
    # 边缘检测、形态学操作、轮廓筛选提取
    # ------------------------------------------------------------------

    def _detect_candidate_boxes(
        self,
        image: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        
        # preprocessing 去噪和边缘提取
        gray = self._to_gray(image)

        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            sigmaX=0,
        )

        # 边缘检测
        edge_map = cv2.Canny(
            blurred,
            threshold1=50,
            threshold2=150,
        )

        # 连接碎边缘为封闭矩形
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 1),
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 5),
        )

        horizontal_lines = cv2.morphologyEx(
            edge_map,
            cv2.MORPH_CLOSE,
            horizontal_kernel,
            iterations=1,
        )

        vertical_lines = cv2.morphologyEx(
            edge_map,
            cv2.MORPH_CLOSE,
            vertical_kernel,
            iterations=1,
        )

        combined = cv2.bitwise_or(
            horizontal_lines,
            vertical_lines,
        )

        # 光照补偿，自适应二值化：对低对比度、渐变、阴影进行阈值分割
        if self.use_adaptive_threshold:
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # 反向二值化：背景变黑，文字和边框变白
                21,
                7,
            )

            adaptive = cv2.morphologyEx(
                adaptive,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (3, 3),
                ),
                iterations=1,
            )

            combined = cv2.bitwise_or(combined, adaptive)

        # 轮廓坐标提取
        combined = cv2.dilate(
            combined,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3),
            ),
            iterations=1,
        )

        contours, _ = cv2.findContours(
            combined,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        height, width = image.shape[:2]
        image_area = width * height

        boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            contour_area = cv2.contourArea(contour)

            if contour_area < self.min_area:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)

            if box_width < self.min_width:
                continue

            if box_height < self.min_height:
                continue

            if box_width > width * self.max_width_ratio:
                continue

            if box_height > height * self.max_height_ratio:
                continue

            box_area = box_width * box_height

            if box_area <= 0:
                continue

            if box_area > image_area * self.max_area_ratio:
                continue

            rectangularity = contour_area / box_area

            if rectangularity < self.rectangularity_threshold:
                continue

            boxes.append(
                (
                    int(x),
                    int(y),
                    int(x + box_width),
                    int(y + box_height),
                )
            )

        return boxes
    

    # ------------------------------------------------------------------
    # Element classification 元素分类器
    # ------------------------------------------------------------------

    def _classify_element(
        self,
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
        related_ocr: Sequence[GUIElement],
        image: np.ndarray,
    ) -> str:

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        aspect_ratio = width / max(height, 1)
        area_ratio = (width * height) / (image_width * image_height)

        has_text = bool(related_ocr)
        merged_text = self._merge_ocr_text(related_ocr)

        lower_text = merged_text.lower()

        checkbox_words = {
            "yes",
            "no",
            "enable",
            "disable",
            "remember",
            "agree",
            "accept",
        }

        button_words = {
            "ok",
            "cancel",
            "save",
            "open",
            "close",
            "submit",
            "send",
            "next",
            "back",
            "login",
            "search",
            "confirm",
            "apply",
            "delete",
            "edit",
            "确定",
            "取消",
            "保存",
            "打开",
            "关闭",
            "提交",
            "发送",
            "下一步",
            "返回",
            "登录",
            "搜索",
            "确认",
            "应用",
            "删除",
            "编辑",
        }

        if 12 <= width <= 40 and 12 <= height <= 40:
            if 0.70 <= aspect_ratio <= 1.30:
                if any(word in lower_text for word in checkbox_words):
                    return "checkbox"

                if not has_text:
                    return "icon"

                return "checkbox"

        if has_text:
            if any(word == lower_text.strip() for word in button_words):
                return "button"

            if 1.4 <= aspect_ratio <= 8.0 and height <= 80:
                if self._looks_like_input_box(
                    image=image,
                    bbox=bbox,
                ):
                    return "input"

                return "button"

        if not has_text:
            if 1.8 <= aspect_ratio <= 15.0 and 20 <= height <= 100:
                if self._looks_like_input_box(
                    image=image,
                    bbox=bbox,
                ):
                    return "input"

            if 0.70 <= aspect_ratio <= 1.30 and max(width, height) <= 96:
                return "icon"

        if area_ratio >= 0.08:
            return "panel"

        if aspect_ratio >= 4.0 and height <= 70:
            return "toolbar"

        return "container"
    

    # 判断是否是输入框
    @staticmethod
    def _looks_like_input_box(
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> bool:
        
        x1, y1, x2, y2 = bbox

        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return False

        gray = UIDetector._to_gray(region)

        height, width = gray.shape[:2]

        if height < 6 or width < 12:
            return False

        inner_margin = max(2, min(width, height) // 8)

        inner = gray[
            inner_margin:height - inner_margin,
            inner_margin:width - inner_margin,
        ]

        if inner.size == 0:
            return False

        border_pixels = np.concatenate(
            [
                gray[0, :],
                gray[-1, :],
                gray[:, 0],
                gray[:, -1],
            ]
        )

        inner_std = float(np.std(inner))
        border_mean = float(np.mean(border_pixels))
        inner_mean = float(np.mean(inner))

        border_contrast = abs(border_mean - inner_mean)

        return inner_std < 55.0 and border_contrast >= 4.0
    

    # ------------------------------------------------------------------
    # OCR association
    # ------------------------------------------------------------------
    
    # 判断边界框里的OCR文本
    @staticmethod
    def _find_ocr_inside_box(
        bbox: tuple[int, int, int, int],
        ocr_elements: Sequence[GUIElement],
    ) -> list[GUIElement]:
        
        x1, y1, x2, y2 = bbox

        related: list[GUIElement] = []

        for element in ocr_elements:
            if element.center is not None:
                center_x, center_y = element.center
            else:
                ox1, oy1, ox2, oy2 = element.bbox
                center_x = int((ox1 + ox2) / 2)
                center_y = int((oy1 + oy2) / 2)

            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                related.append(element)

        return UIDetector._sort_elements(related)
    

    @staticmethod
    def _merge_ocr_text(
        elements: Sequence[GUIElement],
    ) -> str:
        
        return " ".join(
            element.text.strip()
            for element in elements
            if element.text.strip()
        )
    

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_confidence(
        bbox: tuple[int, int, int, int],
        element_type: str,
        related_ocr: Sequence[GUIElement],
        image_width: int,
        image_height: int,
    ) -> float:
        
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        area_ratio = (width * height) / max(
            image_width * image_height,
            1,
        )

        # 先验基准分
        base_scores = {
            "button": 0.72,
            "input": 0.70,
            "checkbox": 0.68,
            "icon": 0.60,
            "toolbar": 0.58,
            "panel": 0.55,
            "container": 0.50,
        }

        score = base_scores.get(element_type, 0.50)

        # 候选框里有文本，加权
        if related_ocr:
            mean_ocr_confidence = float(
                np.mean(
                    [
                        element.confidence
                        for element in related_ocr
                    ]
                )
            )

            # 60%基准分+40%OCR平均置信度
            score = 0.60 * score + 0.40 * mean_ocr_confidence

        # 不合理空间尺寸惩罚
        if area_ratio < 0.00005:
            score -= 0.08

        if area_ratio > 0.50:
            score -= 0.10

        return float(np.clip(score, 0.0, 1.0))
    

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    # 去除重叠
    def _remove_duplicates(
        self,
        elements: Sequence[GUIElement],
    ) -> list[GUIElement]:
        
        sorted_elements = sorted(
            elements,
            key=lambda item: item.confidence,
            reverse=True,
        )

        kept: list[GUIElement] = []

        for candidate in sorted_elements:
            duplicate = False

            for selected in kept:
                iou = self._calculate_iou(
                    candidate.bbox,
                    selected.bbox,
                )

                if iou >= self.duplicate_iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(candidate)

        return kept
    

    # 移除较小的冗余容器，panel和toolbar保留
    @staticmethod
    def _remove_boxes_containing_only_other_boxes(
        elements: Sequence[GUIElement],
    ) -> list[GUIElement]:
        
        result: list[GUIElement] = []

        for index, element in enumerate(elements):
            if element.element_type in {"panel", "toolbar"}:
                result.append(element)
                continue

            redundant = False

            for other_index, other in enumerate(elements):
                if index == other_index:
                    continue

                if UIDetector._contains(
                    outer=element.bbox,
                    inner=other.bbox,
                    tolerance=2,
                ):
                    element_area = UIDetector._box_area(element.bbox)
                    other_area = UIDetector._box_area(other.bbox)

                    if other_area > 0:
                        area_ratio = element_area / other_area

                        if area_ratio <= 1.25:
                            if element.confidence <= other.confidence:
                                redundant = True
                                break

            if not redundant:
                result.append(element)

        return result
    

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
        line_thickness: int = 2,
    ) -> np.ndarray:
        
        canvas = self._prepare_image(image).copy()

        type_colors: dict[str, tuple[int, int, int]] = {
            "button": (0, 255, 0),
            "input": (255, 0, 0),
            "checkbox": (0, 255, 255),
            "icon": (255, 0, 255),
            "toolbar": (255, 255, 0),
            "panel": (0, 128, 255),
            "container": (180, 180, 180),
            "text": (0, 0, 255),
        }

        for element in elements:
            color = type_colors.get(
                element.element_type,
                (255, 255, 255),
            )

            x1, y1, x2, y2 = element.bbox

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                line_thickness,
            )

            if show_center:
                center = element.center or (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                )

                cv2.circle(
                    canvas,
                    center,
                    radius=4,
                    color=color,
                    thickness=-1,
                )

            if show_text:
                label_parts = [element.element_type]

                if element.text:
                    label_parts.append(element.text)

                if show_confidence:
                    label_parts.append(
                        f"{element.confidence:.2f}"
                    )

                label = " | ".join(label_parts)

                text_y = max(y1 - 7, 20)

                cv2.putText(
                    canvas,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        return canvas
    

    @staticmethod
    def save_visualization(
        image: np.ndarray,
        save_path: str | Path,
    ) -> Path:
        
        output_path = Path(save_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            raise IOError(
                f"Failed to save visualization to {output_path}"
            )

        return output_path
    

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    # 两个矩形框的交并比，IoU = intersection area/ union area
    @staticmethod
    def _calculate_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1,
        )
        intersection_height = max(
            0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = UIDetector._box_area(box_a)
        area_b = UIDetector._box_area(box_b)

        union_area = area_a + area_b - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area
    

    @staticmethod
    def _box_area(
        bbox: tuple[int, int, int, int],
    ) -> int:
        
        x1, y1, x2, y2 = bbox

        return max(0, x2 - x1) * max(0, y2 - y1)
    

    # 判断包含关系
    @staticmethod
    def _contains(
        outer: tuple[int, int, int, int],
        inner: tuple[int, int, int, int],
        tolerance: int = 0,
    ) -> bool:
        
        ox1, oy1, ox2, oy2 = outer
        ix1, iy1, ix2, iy2 = inner

        return (
            ix1 >= ox1 - tolerance
            and iy1 >= oy1 - tolerance
            and ix2 <= ox2 + tolerance
            and iy2 <= oy2 + tolerance
        )

    # 视觉元素排序，row_tolerance行容差，对微小水平不齐进行容错处理
    @staticmethod
    def _sort_elements(
        elements: Sequence[GUIElement],
        row_tolerance: int = 15,
    ) -> list[GUIElement]:
        
        def sort_key(
            element: GUIElement,
        ) -> tuple[int, int]:
            x1, y1, _, _ = element.bbox
            row = int(round(y1 / max(row_tolerance, 1)))
            return row, x1

        return sorted(
            elements,
            key=sort_key,
        )
    

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    # image preprocessing
    @staticmethod
    def _prepare_image(
        image: np.ndarray | str | Path,
    ) -> np.ndarray:
        
        if isinstance(image, (str, Path)):
            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image does not exist: {image_path}"
                )

            loaded = cv2.imread(str(image_path))

            if loaded is None:
                raise ValueError(
                    f"OpenCV failed to read image: {image_path}"
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
                "image must be a 2-D or 3-D array."
            )

        result = image

        if result.dtype != np.uint8:
            finite = np.nan_to_num(
                result,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            if np.issubdtype(
                finite.dtype,
                np.floating,
            ):
                if float(np.max(finite)) <= 1.0:
                    finite = finite * 255.0

            result = np.clip(
                finite,
                0,
                255,
            ).astype(np.uint8)

        if result.ndim == 3 and result.shape[2] == 4:
            result = cv2.cvtColor(
                result,
                cv2.COLOR_BGRA2BGR,
            )

        return np.ascontiguousarray(result)


    @staticmethod
    def _to_gray(
        image: np.ndarray,
    ) -> np.ndarray:

        if image.ndim == 2:
            return image

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY,
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )


    # 剪裁区域检查
    @staticmethod
    def _validate_region(
        image: np.ndarray,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> None:
        
        for name, value in {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

        if left < 0 or top < 0:
            raise ValueError(
                "left and top must be non-negative."
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                "width and height must be positive."
            )

        image_height, image_width = image.shape[:2]

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
            f"min_width={self.min_width}, "
            f"min_height={self.min_height}, "
            f"min_area={self.min_area}, "
            f"rectangularity_threshold="
            f"{self.rectangularity_threshold}, "
            f"duplicate_iou_threshold="
            f"{self.duplicate_iou_threshold}"
            f")"
        )