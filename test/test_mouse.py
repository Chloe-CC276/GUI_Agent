"""
test mouse perception integration
完成以下动作：
1.双击打开scr文件夹
2.在scr文件夹中右键executor文件夹
3.在右键菜单中点击属性
4.鼠标移动到文件列表区域
5.向上滚动10格，再向下滚动10格
6.将executor文件夹拖拽到桌面
"""


from __future__ import annotations

import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from src.executor.mouse import (
    MouseActionResult,
    MouseController,
)
from src.perception.gui_element import GUIElement
from src.perception.perception_pipeline import (
    PerceptionPipeline,
    PerceptionResult,
)

# ================================================================
# Test configuration
# ================================================================

SRC_FOLDER_TEXT = "src"
EXECUTOR_FOLDER_TEXT = "test"

# Windows Chinese and English menu labels.
PROPERTY_TEXT_CANDIDATES = (
    "属性",
    "Properties",
)

# Windows property dialog close-button candidates.
CLOSE_TEXT_CANDIDATES = (
    "确定",
    "OK",
    "取消",
    "Cancel",
)

# Wait times for File Explorer and context menu refresh.
AFTER_DOUBLE_CLICK_WAIT = 2.0
AFTER_RIGHT_CLICK_WAIT = 1.0
AFTER_PROPERTY_CLICK_WAIT = 2.0
AFTER_CLOSE_WAIT = 1.0
AFTER_SCROLL_WAIT = 0.5

# Start with True.
# After checking the recognised coordinates, change to False.
DRY_RUN = False

# Restrict perception to a selected region if required.
# Format: (left, top, width, height)
# Set to None to process the complete screen.
PERCEPTION_REGION: Optional[tuple[int, int, int, int]] = None

# The scrolling point is selected relative to screen size.
# 0.25 means 25% from the left side.
SCROLL_AREA_X_RATIO = 0.25
SCROLL_AREA_Y_RATIO = 0.55

# 是否真实执行“移动 executor 到桌面”
# 建议先使用 executor_test 文件夹进行测试。
MOVE_EXECUTOR_TO_DESKTOP = True

DESKTOP_TEXT_CANDIDATES = (
    "桌面",
    "Desktop",
)

AFTER_DRAG_WAIT = 2.0


# ================================================================
# Logging
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ================================================================
# Formatting helpers
# ================================================================

def print_separator(
    title: str,
    character: str = "=",
    width: int = 100,
) -> None:
    """
    Print a formatted section heading.
    """

    print()
    print(character * width)
    print(title)
    print(character * width)


def print_gui_element(
    title: str,
    element: GUIElement,
) -> None:
    """
    Print one GUIElement in a readable format.
    """

    print_separator(title, "-")

    print(f"文字内容       : {element.text!r}")
    print(f"元素类型       : {element.element_type}")
    print(f"识别置信度     : {element.confidence:.4f}")
    print(f"边界框         : {element.bbox}")
    print(f"中心坐标       : {element.center}")


def print_perception_summary(
    title: str,
    result: PerceptionResult,
) -> None:
    """
    Print perception statistics.
    """

    print_separator(title, "-")

    summary = result.summary()

    print(f"OCR文字数量    : {summary['ocr_element_count']}")
    print(f"UI元素数量     : {summary['ui_element_count']}")
    print(f"融合元素数量   : {summary['merged_element_count']}")
    print(f"识别区域       : {summary['capture_region']}")
    print(
        "感知执行时间   : "
        f"{summary['elapsed_time_seconds']:.4f} 秒"
    )
    print(
        "元素类型统计   : "
        f"{summary['element_type_counts']}"
    )


def print_mouse_result(
    step_number: int,
    title: str,
    result: MouseActionResult,
) -> None:
    """
    Print one MouseActionResult using a fixed format.
    """

    print_separator(
        f"步骤 {step_number}：{title}",
        "-",
    )

    status = "成功" if result.success else "失败"

    print(f"执行状态       : {status}")
    print(f"执行动作       : {result.action}")
    print(f"动作描述       : {result.message}")
    print(f"起始坐标       : {result.start_position.as_tuple()}")
    print(f"结束坐标       : {result.end_position.as_tuple()}")
    print(f"坐标变化       : {_calculate_position_change(result)}")
    print(f"执行耗时       : {result.elapsed_time:.4f} 秒")
    print(f"Dry-run模式    : {result.dry_run}")
    print(f"元数据         : {result.metadata}")

    if result.error:
        print(f"错误信息       : {result.error}")


def _calculate_position_change(
    result: MouseActionResult,
) -> tuple[int, int]:
    """
    Calculate mouse coordinate displacement.
    """

    delta_x = (
        result.end_position.x
        - result.start_position.x
    )

    delta_y = (
        result.end_position.y
        - result.start_position.y
    )

    return delta_x, delta_y


# ================================================================
# Perception helpers
# ================================================================

def capture_perception(
    pipeline: PerceptionPipeline,
    title: str,
) -> PerceptionResult:
    """
    Capture and analyse the current desktop.
    """

    print_separator(title)

    result = pipeline.capture_and_run(
        region=PERCEPTION_REGION,
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
    )

    print_perception_summary(
        title=f"{title}：感知结果统计",
        result=result,
    )

    return result


def find_text_candidates(
    pipeline: PerceptionPipeline,
    result: PerceptionResult,
    texts: Sequence[str],
    exact_match: bool = False,
) -> list[GUIElement]:
    """
    Search several possible text labels.
    """

    matches: list[GUIElement] = []

    for text in texts:
        current_matches = pipeline.find_text(
            result=result,
            query=text,
            exact_match=exact_match,
            case_sensitive=False,
        )

        matches.extend(current_matches)

    return matches


def select_best_element(
    elements: Sequence[GUIElement],
) -> Optional[GUIElement]:
    """
    Select the highest-confidence element with a valid centre.
    """

    valid_elements = [
        element
        for element in elements
        if element.center is not None
    ]

    if not valid_elements:
        return None

    return max(
        valid_elements,
        key=lambda element: element.confidence,
    )


def require_text_element(
    pipeline: PerceptionPipeline,
    result: PerceptionResult,
    candidates: Sequence[str],
    description: str,
    exact_match: bool = False,
) -> GUIElement:
    """
    Find a required screen element or raise a clear error.
    """

    matches = find_text_candidates(
        pipeline=pipeline,
        result=result,
        texts=candidates,
        exact_match=exact_match,
    )

    best_match = select_best_element(matches)

    if best_match is None:
        visible_texts = result.get_texts()

        raise RuntimeError(
            f"未找到{description}。"
            f"候选文字={tuple(candidates)}。\n"
            f"当前屏幕识别出的文字包括：\n"
            f"{visible_texts}"
        )

    print_gui_element(
        title=f"定位结果：{description}",
        element=best_match,
    )

    return best_match


# ================================================================
# Action helpers
# ================================================================

def double_click_element(
    mouse: MouseController,
    element: GUIElement,
) -> MouseActionResult:
    """
    Double-click the centre of a GUIElement.
    """

    if element.center is None:
        raise ValueError(
            "目标元素不存在中心坐标，无法双击。"
        )

    return mouse.double_click(
        x=element.center[0],
        y=element.center[1],
        button="left",
        interval=0.12,
        duration=0.30,
    )


def right_click_element(
    mouse: MouseController,
    element: GUIElement,
) -> MouseActionResult:
    """
    Right-click the centre of a GUIElement.
    """

    if element.center is None:
        raise ValueError(
            "目标元素不存在中心坐标，无法右键单击。"
        )

    return mouse.right_click(
        x=element.center[0],
        y=element.center[1],
        duration=0.30,
    )


def left_click_element(
    mouse: MouseController,
    element: GUIElement,
) -> MouseActionResult:
    """
    Left-click the centre of a GUIElement.
    """

    if element.center is None:
        raise ValueError(
            "目标元素不存在中心坐标，无法单击。"
        )

    return mouse.left_click(
        x=element.center[0],
        y=element.center[1],
        duration=0.25,
    )


def calculate_scroll_position(
    mouse: MouseController,
) -> tuple[int, int]:
    """
    Select a point in the left-side file-list area.

    Adjust the ratios when the folder window layout differs.
    """

    screen_width, screen_height = (
        mouse.get_screen_size()
    )

    x = int(
        screen_width
        * SCROLL_AREA_X_RATIO
    )

    y = int(
        screen_height
        * SCROLL_AREA_Y_RATIO
    )

    mouse.validate_position(x, y)

    return x, y

def drag_element_to_element(
    mouse: MouseController,
    source_element: GUIElement,
    target_element: GUIElement,
    duration: float = 1.2,
) -> MouseActionResult:

    if source_element.center is None:
        raise ValueError(
            "源元素不存在中心坐标，无法执行拖拽。"
        )

    if target_element.center is None:
        raise ValueError(
            "目标元素不存在中心坐标，无法执行拖拽。"
        )

    # 首先移动到源元素中心
    move_result = mouse.move_to(
        x=source_element.center[0],
        y=source_element.center[1],
        duration=0.3,
    )

    if not move_result.success:
        raise RuntimeError(
            "鼠标无法移动到源元素位置。"
        )

    # 从源元素拖拽到目标元素
    return mouse.drag_to(
        x=target_element.center[0],
        y=target_element.center[1],
        duration=duration,
        button="left",
    )

# ================================================================
# Main test workflow
# ================================================================

def run_mouse_perception_test() -> None:
    """
    Execute the requested perception-and-mouse test.
    """

    print_separator(
        "GUI Agent 感知与鼠标执行器集成测试"
    )

    print(f"Dry-run模式    : {DRY_RUN}")
    print(f"目标文件夹1    : {SRC_FOLDER_TEXT!r}")
    print(f"目标文件夹2    : {EXECUTOR_FOLDER_TEXT!r}")
    print(f"识别区域       : {PERCEPTION_REGION}")
    print()
    print(
        "运行前请在文件资源管理器中打开"
        "GUI Agent项目根目录。"
    )
    print(
        "紧急停止方法：将鼠标快速移动到"
        "屏幕左上角。"
    )

    pipeline = PerceptionPipeline(
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
        include_unmatched_ocr=True,
    )

    mouse = MouseController(
        pause=0.10,
        fail_safe=True,
        default_duration=0.25,
        dry_run=DRY_RUN,
        raise_on_error=True,
    )

    all_mouse_results: list[MouseActionResult] = []

    # ------------------------------------------------------------
    # Step 1: Locate and double-click src
    # ------------------------------------------------------------

    root_result = capture_perception(
        pipeline=pipeline,
        title="阶段1：识别项目根目录",
    )

    src_element = require_text_element(
        pipeline=pipeline,
        result=root_result,
        candidates=(SRC_FOLDER_TEXT,),
        description="src文件夹",
        exact_match=True,
    )

    src_double_click_result = double_click_element(
        mouse=mouse,
        element=src_element,
    )

    all_mouse_results.append(
        src_double_click_result
    )

    print_mouse_result(
        step_number=1,
        title="双击打开src文件夹",
        result=src_double_click_result,
    )

    if DRY_RUN:
        print()
        print(
            "当前为Dry-run模式，src文件夹并未真正打开。"
            "后续依赖界面变化的步骤将不继续执行。"
        )

        print_final_summary(all_mouse_results)
        return

    time.sleep(AFTER_DOUBLE_CLICK_WAIT)

    # ------------------------------------------------------------
    # Step 2: Locate and right-click executor
    # ------------------------------------------------------------

    src_result = capture_perception(
        pipeline=pipeline,
        title="阶段2：识别src文件夹内容",
    )

    executor_element = require_text_element(
        pipeline=pipeline,
        result=src_result,
        candidates=(EXECUTOR_FOLDER_TEXT,),
        description="executor文件夹",
        exact_match=True,
    )

    executor_right_click_result = (
        right_click_element(
            mouse=mouse,
            element=executor_element,
        )
    )

    all_mouse_results.append(
        executor_right_click_result
    )

    print_mouse_result(
        step_number=2,
        title="右键单击executor文件夹",
        result=executor_right_click_result,
    )

    time.sleep(AFTER_RIGHT_CLICK_WAIT)

    # ------------------------------------------------------------
    # Step 3: Detect and click Properties
    # ------------------------------------------------------------

    context_menu_result = capture_perception(
        pipeline=pipeline,
        title="阶段3：识别右键菜单",
    )

    property_element = require_text_element(
        pipeline=pipeline,
        result=context_menu_result,
        candidates=PROPERTY_TEXT_CANDIDATES,
        description="属性菜单项",
        exact_match=True,
    )

    property_click_result = left_click_element(
        mouse=mouse,
        element=property_element,
    )

    all_mouse_results.append(
        property_click_result
    )

    print_mouse_result(
        step_number=3,
        title="单击属性菜单项",
        result=property_click_result,
    )

    time.sleep(AFTER_PROPERTY_CLICK_WAIT)

    # ------------------------------------------------------------
    # Step 4: Close property window
    # ------------------------------------------------------------

    property_dialog_result = capture_perception(
        pipeline=pipeline,
        title="阶段4：识别属性窗口",
    )

    close_element = require_text_element(
        pipeline=pipeline,
        result=property_dialog_result,
        candidates=CLOSE_TEXT_CANDIDATES,
        description="属性窗口关闭按钮",
        exact_match=True,
    )

    close_click_result = left_click_element(
        mouse=mouse,
        element=close_element,
    )

    all_mouse_results.append(
        close_click_result
    )

    print_mouse_result(
        step_number=4,
        title="关闭executor属性窗口",
        result=close_click_result,
    )

    time.sleep(AFTER_CLOSE_WAIT)

    # ------------------------------------------------------------
    # Step 5: Move to file list
    # ------------------------------------------------------------

    scroll_x, scroll_y = (
        calculate_scroll_position(mouse)
    )

    move_result = mouse.move_to(
        x=scroll_x,
        y=scroll_y,
        duration=0.30,
    )

    all_mouse_results.append(move_result)

    print_mouse_result(
        step_number=5,
        title="移动鼠标至文件列表区域",
        result=move_result,
    )

    # ------------------------------------------------------------
    # Step 6: Scroll upward 10
    # ------------------------------------------------------------

    scroll_up_result = mouse.scroll(
        amount=10,
        x=scroll_x,
        y=scroll_y,
    )

    all_mouse_results.append(
        scroll_up_result
    )

    print_mouse_result(
        step_number=6,
        title="文件列表向上滚动10格",
        result=scroll_up_result,
    )

    time.sleep(AFTER_SCROLL_WAIT)

    # ------------------------------------------------------------
    # Step 7: Scroll downward 10
    # ------------------------------------------------------------

    scroll_down_result = mouse.scroll(
        amount=-10,
        x=scroll_x,
        y=scroll_y,
    )

    all_mouse_results.append(
        scroll_down_result
    )

    print_mouse_result(
        step_number=7,
        title="文件列表向下滚动10格",
        result=scroll_down_result,
    )

    time.sleep(AFTER_SCROLL_WAIT)


    # ------------------------------------------------------------
    # Step 8: Move executor folder to Desktop
    # ------------------------------------------------------------

    if MOVE_EXECUTOR_TO_DESKTOP:
        print_separator(
            "阶段5：将executor文件夹移动到桌面"
        )

        # 滚动后界面可能发生变化，因此重新截图识别
        move_stage_result = capture_perception(
            pipeline=pipeline,
            title="阶段5：重新识别executor与桌面位置",
        )

        # 重新定位 executor 文件夹
        executor_for_move = require_text_element(
            pipeline=pipeline,
            result=move_stage_result,
            candidates=(EXECUTOR_FOLDER_TEXT,),
            description="待移动的executor文件夹",
            exact_match=True,
        )

        # 识别文件资源管理器左侧导航栏中的桌面
        desktop_element = require_text_element(
            pipeline=pipeline,
            result=move_stage_result,
            candidates=DESKTOP_TEXT_CANDIDATES,
            description="文件资源管理器左侧的桌面目录",
            exact_match=True,
        )

        print_gui_element(
            title="拖拽源：executor文件夹",
            element=executor_for_move,
        )

        print_gui_element(
            title="拖拽目标：桌面目录",
            element=desktop_element,
        )

        move_executor_result = drag_element_to_element(
            mouse=mouse,
            source_element=executor_for_move,
            target_element=desktop_element,
            duration=1.5,
        )

        all_mouse_results.append(
            move_executor_result
        )

        print_mouse_result(
            step_number=8,
            title="将executor文件夹拖拽到桌面",
            result=move_executor_result,
        )

        time.sleep(AFTER_DRAG_WAIT)

        # 重新感知，检查源目录中是否仍存在 executor
        verification_result = capture_perception(
            pipeline=pipeline,
            title="阶段6：检查移动结果",
        )

        remaining_executor = pipeline.find_text(
            result=verification_result,
            query=EXECUTOR_FOLDER_TEXT,
            exact_match=True,
            case_sensitive=False,
        )

        print_separator(
            "步骤8：移动结果验证",
            "-",
        )

        if remaining_executor:
            print(
                "验证结果       : executor仍在当前目录中"
            )
            print(
                "可能原因       : 拖拽未成功、目标识别错误，"
                "或系统要求额外确认"
            )
        else:
            print(
                "验证结果       : 当前目录中未识别到executor"
            )
            print(
                "初步判断       : executor可能已成功移动到桌面"
            )

    else:
        print_separator(
            "步骤8：移动executor到桌面",
            "-",
        )

        print("执行状态       : 已跳过")
        print(
            "跳过原因       : MOVE_EXECUTOR_TO_DESKTOP=False"
        )
        print(
            "安全提示       : 移动src/executor会破坏当前项目结构，"
            "建议使用executor_test进行测试"
        )

    print_final_summary(all_mouse_results)


# ================================================================
# Final summary
# ================================================================

def print_final_summary(
    results: Sequence[MouseActionResult],
) -> None:
    """
    Print the complete action summary.
    """

    print_separator(
        "鼠标执行器测试结果汇总"
    )

    successful_count = sum(
        1
        for result in results
        if result.success
    )

    failed_count = (
        len(results)
        - successful_count
    )

    total_elapsed_time = sum(
        result.elapsed_time
        for result in results
    )

    print(f"动作总数       : {len(results)}")
    print(f"成功动作数     : {successful_count}")
    print(f"失败动作数     : {failed_count}")
    print(
        f"累计执行耗时   : "
        f"{total_elapsed_time:.4f} 秒"
    )
    print()

    header = (
        f"{'序号':<6}"
        f"{'动作':<22}"
        f"{'状态':<10}"
        f"{'起始坐标':<18}"
        f"{'结束坐标':<18}"
        f"{'耗时/秒':<12}"
    )

    print(header)
    print("-" * 100)

    for index, result in enumerate(
        results,
        start=1,
    ):
        status = (
            "成功"
            if result.success
            else "失败"
        )

        print(
            f"{index:<6}"
            f"{result.action:<22}"
            f"{status:<10}"
            f"{str(result.start_position.as_tuple()):<18}"
            f"{str(result.end_position.as_tuple()):<18}"
            f"{result.elapsed_time:<12.4f}"
        )

    print()
    print(
        "测试结论："
        + (
            "全部鼠标动作执行成功。"
            if failed_count == 0
            else "部分动作执行失败，请检查上方日志。"
        )
    )

def test_mouse_perception_integration() -> None:
    run_mouse_perception_test()
    

if __name__ == "__main__":
    try:
        run_mouse_perception_test()

    except KeyboardInterrupt:
        print()
        print("测试被用户中断。")

    except Exception as error:
        logger.exception(
            "感知与鼠标执行器集成测试失败。"
        )

        print()
        print_separator(
            "测试执行失败"
        )

        print(f"错误类型：{type(error).__name__}")
        print(f"错误信息：{error}")

        raise