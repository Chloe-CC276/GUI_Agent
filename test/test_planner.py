"""
tests/test_vlm_executor_automation.py

真实端到端 GUI Agent 测试：

1. 截取当前屏幕；
2. 运行 PerceptionPipeline，获得 OCR/UI 元素；
3. 将“当前截图的 PNG 字节”直接传给 Qwen-VL；
4. Planner 根据截图与任务生成一个动作；
5. Executor 真实执行动作；
6. 重新截图并继续规划，直到模型判断任务完成。

目标任务：
    打开当前页面中名为 executor 的文件夹。

重要安全说明
------------
本测试会真实移动鼠标并点击当前桌面。运行前请：
1. 打开包含 executor 文件夹的测试窗口；
2. 确保该窗口处于前台且没有敏感操作；
3. 不要在运行过程中移动鼠标或切换窗口；
4. 首次调试时把 DRY_RUN 改为 True；
5. Windows 上可把鼠标快速移到屏幕左上角触发 PyAutoGUI failsafe。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2


# =====================================================================
# Project path
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Project imports
# =====================================================================

from src.agent.planner import (
    InvalidResponsePolicy,
    Planner,
    PlannerConfig,
)
from src.agent.result import (
    ErrorInfo,
    ResultStatus,
    ToolResult,
)
from src.agent.state import (
    AgentState,
    ObservationSource,
    ObservationState,
)
from src.executor.executor import Executor
from src.model.qwen_vlm import QwenVLM
from src.perception.perception_pipeline import (
    PerceptionPipeline,
    PerceptionResult,
)


# =====================================================================
# Test configuration
# =====================================================================

TASK = (
    "请观察当前屏幕，找到名称为 executor 的文件夹并打开它。"
    "通常应双击文件夹。每次只返回一个动作；"
    "打开成功后返回 finish。"
)

MAX_STEPS = 6

# 首次运行建议设为 True，只验证规划，不真实点击。
DRY_RUN = False

# 每个动作执行后等待 GUI 更新。
WAIT_AFTER_ACTION_SECONDS = 1.2

# 是否运行 OCR/UI 检测。截图本身始终会直接传给 VLM。
ENABLE_OCR = True
ENABLE_UI_DETECTION = True

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vlm_executor_automation"

MODEL_NAME = "qwen3-vl-plus"

QWEN_REGION = "beijing"



# =====================================================================
# VLM adapter
# =====================================================================

class PlannerVLMAdapter:
    """
    只向 Planner 暴露 generate()。

    原因：
    BaseVLM.generate_json() 返回：
        (parsed_json, VLMResponse)

    当前 Planner 优先寻找 generate_json()，但其默认解析器期待单个
    VLMResponse 或 Mapping。这个适配器让 Planner 使用 generate()，
    并由 Planner 自己解析 response.text 中的 JSON。
    """

    def __init__(self, vlm: QwenVLM) -> None:
        self._vlm = vlm

    def generate(
        self,
        prompt: str,
        *,
        images: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        response_format = kwargs.get("response_format")

        if isinstance(response_format, dict):
            response_format = response_format.get("type")

        return self._vlm.generate(
            prompt=prompt,
            images=images,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            timeout=kwargs.get("timeout"),
            response_format=response_format,
        )


# =====================================================================
# Conversion helpers
# =====================================================================

def gui_element_to_dict(element: Any) -> dict[str, Any]:
    """
    将 GUIElement 转成 Planner 可序列化的数据。

    保留 text、bbox、center、confidence 和 element_type，
    让 VLM 同时获得视觉截图和传统感知结果。
    """
    bbox = getattr(element, "bbox", None)
    center = getattr(element, "center", None)

    return {
        "text": str(getattr(element, "text", "") or ""),
        "element_type": str(
            getattr(element, "element_type", "unknown") or "unknown"
        ),
        "bbox": list(bbox) if bbox is not None else None,
        "center": list(center) if center is not None else None,
        "confidence": float(
            getattr(element, "confidence", 0.0) or 0.0
        ),
    }


def image_to_png_bytes(image: Any) -> bytes:
    """
    把 OpenCV BGR ndarray 编码为 PNG 字节。

    这里不依赖临时文件。PNG bytes 会通过 BaseVLM 的 ImageInput.from_bytes()
    转成 data URL，并直接发送给多模态模型。
    """
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError("OpenCV failed to encode screenshot as PNG.")

    return encoded.tobytes()


def perception_to_observation(
    result: PerceptionResult,
    *,
    screenshot_path: Path,
) -> ObservationState:
    height, width = result.original_image.shape[:2]

    elements = [
        gui_element_to_dict(element)
        for element in result.merged_elements
    ]

    ocr_items = [
        gui_element_to_dict(element)
        for element in result.ocr_elements
    ]

    ocr_text = "\n".join(
        text
        for text in result.get_texts()
        if text.strip()
    )

    screenshot_bytes = image_to_png_bytes(
        result.original_image
    )

    return ObservationState(
        # 关键点：当前截图的内存 PNG bytes 直接交给 Planner/VLM。
        screenshot=screenshot_bytes,
        screenshot_path=str(screenshot_path),
        screen_width=width,
        screen_height=height,
        ocr_text=ocr_text or None,
        ocr_items=ocr_items,
        gui_elements=elements,
        source=ObservationSource.PERCEPTION,
        raw_observation=result,
        metadata={
            **result.summary(),
            "image_transport": "in_memory_png_bytes",
        },
    )


def planner_action_factory(
    action_type: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """
    将 Planner 的标准动作转换为 Executor.execute() 可接收的字典。

    Executor 支持 Action、dict 或 JSON 字符串，并会在内部调用
    Action.from_dict()。其动作字段使用 type，而不是 action_type。
    """
    params = dict(parameters)

    # Planner 使用 duration 表示 wait 时间，而 Executor 的 WAIT Action
    # 使用 seconds。普通鼠标动作仍然保留 duration。
    if action_type == "wait" and "duration" in params:
        params["seconds"] = params.pop("duration")

    return {
        "type": action_type,
        **params,
        "description": (
            f"VLM planned action: {action_type}"
        ),
        "metadata": {
            "source": "qwen_vlm_planner",
        },
    }


def execution_to_tool_result(
    execution_result: Any,
) -> ToolResult:
    if execution_result.success:
        return ToolResult.success(
            tool_name="executor.execute",
            output=execution_result,
            message=execution_result.message,
            metadata=execution_result.summary(),
        )

    return ToolResult.failed(
        tool_name="executor.execute",
        error=ErrorInfo(
            error_type="ExecutorActionFailed",
            message=(
                execution_result.error
                or execution_result.message
                or "Executor action failed."
            ),
            retryable=False,
            details=execution_result.summary(),
        ),
        output=execution_result,
        message=execution_result.message,
        metadata=execution_result.summary(),
    )


# =====================================================================
# Prompt
# =====================================================================

SYSTEM_PROMPT = """
你是桌面 GUI Agent 的规划器。你会收到：
1. 用户任务；
2. 当前桌面截图；
3. OCR 和 GUI 元素；
4. 之前执行过的动作。

请基于当前截图选择下一步单一动作。

本次任务是在当前测试页面中找到名为 executor 的文件夹并打开它。
文件夹通常需要 double_click。不能仅根据 OCR 猜测坐标，应结合截图、
bbox 和 center。动作坐标必须是完整屏幕坐标。

只返回 JSON，不要输出 Markdown。格式示例：

{
  "decision": "act",
  "action": {
    "type": "double_click",
    "parameters": {
      "x": 500,
      "y": 300,
      "button": "left"
    }
  },
  "reason": "截图中 executor 文件夹中心位于该坐标",
  "confidence": 0.95
}

确认 executor 文件夹已经打开后返回：

{
  "decision": "finish",
  "finish_message": "executor 文件夹已经打开",
  "reason": "当前截图显示 executor 文件夹内容",
  "confidence": 0.95
}

截图不足或界面尚未刷新时可以返回 retry。不要一次返回多个动作。
""".strip()


# =====================================================================
# Main automation
# =====================================================================

def run_automation() -> AgentState:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
    logger = logging.getLogger("vlm_executor_test")

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "未设置 DASHSCOPE_API_KEY。请先在环境变量中配置 Qwen API Key。"
        )

    state = AgentState.create(
        task=TASK,
        max_steps=MAX_STEPS,
        max_retries=3,
        language="zh",
        constraints=[
            "每次只执行一个动作",
            "只能点击当前截图中可见并有证据支持的坐标",
            "不得关闭测试窗口或操作无关应用",
        ],
        success_criteria=[
            "executor 文件夹已经被打开",
        ],
    )
    state.begin()

    perception = PerceptionPipeline(
        enable_preprocessing=False,
        enable_ocr=ENABLE_OCR,
        enable_ui_detection=ENABLE_UI_DETECTION,
        merge_results=True,
        include_unmatched_ocr=True,
    )

    qwen = QwenVLM(
        model=MODEL_NAME,
        api_key=api_key,
        region="beijing",
        workspace_id=os.getenv(
            "DASHSCOPE_WORKSPACE_ID"
        ),
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL"
        ),
        keep_history=False,
        enable_thinking=False,
    )

    planner = Planner(
        vlm=PlannerVLMAdapter(qwen),
        config=PlannerConfig(
            system_prompt=SYSTEM_PROMPT,
            max_attempts=2,
            temperature=0.0,
            max_tokens=900,
            timeout_seconds=90.0,
            include_screenshot=True,
            history_limit=8,
            max_ocr_chars=3000,
            max_elements=120,
            require_reason=True,
            invalid_response_policy=(
                InvalidResponsePolicy.RETRY
            ),
            allowed_actions=(
                "click",
                "double_click",
                "wait",
                "finish",
                "retry",
                "fail",
            ),
            validate_coordinates=True,
            clamp_coordinates=False,
            include_raw_response=True,
            include_prompt_in_metadata=False,
        ),
        action_factory=planner_action_factory,
    )

    executor = Executor(
        dry_run=DRY_RUN,
        stop_on_failure=True,
        raise_on_error=False,
        default_wait_after_action=(
            WAIT_AFTER_ACTION_SECONDS
        ),
        keep_history=True,
    )

    try:
        for step_index in range(MAX_STEPS):
            logger.info(
                "========== Agent step %d/%d ==========",
                step_index + 1,
                MAX_STEPS,
            )

            # -----------------------------------------------------
            # 1. Capture + perception
            # -----------------------------------------------------
            perception_result = perception.capture_and_run()

            screenshot_path = (
                OUTPUT_DIR
                / f"step_{step_index:02d}_screen.png"
            )

            if not cv2.imwrite(
                str(screenshot_path),
                perception_result.original_image,
            ):
                raise RuntimeError(
                    f"Failed to save screenshot: {screenshot_path}"
                )

            observation = perception_to_observation(
                perception_result,
                screenshot_path=screenshot_path,
            )

            observation_tool_result = ToolResult.success(
                tool_name="perception.capture_and_run",
                output=perception_result,
                message=(
                    "Current screen captured and perception completed."
                ),
                metadata=perception_result.summary(),
            )

            state.update_observation(
                observation,
                tool_result=observation_tool_result,
            )

            logger.info(
                "Screenshot=%s, elements=%d, OCR texts=%s",
                screenshot_path,
                perception_result.element_count,
                perception_result.get_texts()[:12],
            )

            # -----------------------------------------------------
            # 2. VLM planning
            # -----------------------------------------------------
            planner_result = planner.plan(state)
            state.update_planner_result(planner_result)

            logger.info(
                "Planner decision=%s, reason=%s, action=%s",
                planner_result.decision.value,
                planner_result.reason,
                planner_result.action,
            )

            planner_log_path = (
                OUTPUT_DIR
                / f"step_{step_index:02d}_planner.json"
            )
            planner_log_path.write_text(
                json.dumps(
                    planner_result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # -----------------------------------------------------
            # 3. Terminal decisions
            # -----------------------------------------------------
            if planner_result.is_finished:
                state.finish(
                    planner_result.finish_message
                    or "executor 文件夹已经打开"
                )
                logger.info(
                    "Task finished: %s",
                    state.final_message,
                )
                break

            if planner_result.should_retry:
                logger.warning(
                    "Planner requested retry: %s",
                    planner_result.reason,
                )

                # 当前 State 中 retry 后仍处于 PLANNING。
                # 下一轮需要重新截图，因此切回 OBSERVING。
                state.set_phase("observing")
                time.sleep(0.8)
                continue

            if not planner_result.should_execute:
                error = (
                    planner_result.error
                    or ErrorInfo(
                        error_type="PlannerNoAction",
                        message="Planner returned no executable action.",
                    )
                )
                state.fail(error=error)
                break

            # -----------------------------------------------------
            # 4. Real executor action
            # -----------------------------------------------------
            execution_result = executor.execute(
                planner_result.action
            )
            execution_tool_result = execution_to_tool_result(
                execution_result
            )
            state.update_execution_result(
                execution_tool_result
            )

            logger.info(
                "Executor success=%s, action=%s, message=%s",
                execution_result.success,
                execution_result.action_type.value,
                execution_result.message,
            )

            # -----------------------------------------------------
            # 5. Commit this observe-plan-execute step
            # -----------------------------------------------------
            step_result = state.build_current_step()
            state.commit_step(step_result)

            if not execution_result.success:
                logger.error(
                    "Execution failed: %s",
                    execution_result.error,
                )
                break

        if not state.is_terminal:
            state.fail(
                error=ErrorInfo(
                    error_type="MaximumStepsExceeded",
                    message=(
                        f"Task did not finish within {MAX_STEPS} steps."
                    ),
                ),
                reason="max_steps",
            )

    except KeyboardInterrupt:
        state.cancel("用户通过键盘中断测试。")
        raise

    except Exception as error:
        if not state.is_terminal:
            state.fail(
                error=ErrorInfo.from_exception(error),
                message="自动化测试异常终止。",
            )
        raise

    finally:
        run_result = state.to_run_result()

        run_result.save_json(
            OUTPUT_DIR / "agent_run_result.json",
            include_raw_output=True,
            include_tool_output=False,
        )

        state.save_json(
            OUTPUT_DIR / "agent_state.json",
            include_observation_raw=False,
            include_screenshot=False,
            include_result_raw_output=True,
            include_tool_output=False,
        )

        qwen.close()

    return state


def test_open_executor_folder_with_vlm() -> None:
    """
    Pytest 真实自动化入口。

    默认会真实点击。仅在已准备好测试窗口时运行：

        pytest tests/test_vlm_executor_automation.py -v -s
    """
    state = run_automation()

    assert state.is_finished, (
        "GUI Agent 未完成任务。"
        f"phase={state.phase.value}, "
        f"reason={state.termination_reason}, "
        f"error={state.error}"
    )


if __name__ == "__main__":
    final_state = run_automation()

    print(
        json.dumps(
            final_state.to_run_result().summary(),
            ensure_ascii=False,
            indent=2,
        )
    )