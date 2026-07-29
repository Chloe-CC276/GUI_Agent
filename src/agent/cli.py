"""Command-line interface for the LangChain GUI Agent.

Run from the project root:

    python -m agent.cli
    python -m agent.cli --dry-run
    python -m agent.cli --task "在 Name 输入框输入 GUIAgent 并点击 Submit"

The CLI owns user interaction only.  Perception, planning, execution,
verification, reflection and memory are delegated to ``AgentChain``.
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import importlib
import json
import logging
import os
import shlex
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


class CLIError(RuntimeError):
    """Raised for configuration or command-line integration failures."""


@dataclass(slots=True)
class CLIConfig:
    model: str = "qwen3-vl-8b-instruct"
    region: str = "frankfurt"
    max_steps: int = 12
    max_retries: int = 2
    max_reflections: int = 3
    post_action_wait: float = 0.5
    dry_run: bool = False
    language: str = "zh"
    verbose: bool = True
    task_retry_count: int = 1
    log_file: str = "logs"

    def update(self, name: str, raw_value: str) -> None:
        aliases = {
            "wait": "post_action_wait",
            "steps": "max_steps",
            "retries": "max_retries",
            "reflections": "max_reflections",
        }
        name = aliases.get(name, name)
        if not hasattr(self, name):
            raise CLIError(f"未知配置项：{name}")
        current = getattr(self, name)
        if isinstance(current, bool):
            value = _parse_bool(raw_value)
        elif isinstance(current, int):
            value = int(raw_value)
        elif isinstance(current, float):
            value = float(raw_value)
        else:
            value = raw_value
        if name in {"max_steps", "max_retries", "max_reflections"} and value < 0:
            raise CLIError(f"{name} 不能小于 0")
        if name == "max_steps" and value == 0:
            raise CLIError("max_steps 必须大于 0")
        if name == "post_action_wait" and not 0 <= value <= 60:
            raise CLIError("post_action_wait 必须在 0～60 秒之间")
        setattr(self, name, value)


@dataclass(slots=True)
class AgentRuntime:
    """Objects required to submit one or more tasks in the same CLI session."""

    chain: Any
    state_cls: type
    components: Mapping[str, Any]

    def new_state(self, task: str, config: CLIConfig) -> Any:
        return self.state_cls.create(
            task=task,
            max_steps=config.max_steps,
            max_retries=config.max_retries,
            language=config.language,
            constraints=[
                "每次只规划并执行一个动作",
                "只能操作当前任务相关的可见界面",
                "没有视觉证据时应重新感知，不得猜测坐标",
                (
                    "点击类动作必须提供检测结果中的 target_text 或 "
                    "element_id，禁止只提供 x/y"
                ),
                (
                    "打开桌面快捷方式、文件或文件夹使用 double_click；"
                    "按钮、菜单、标签页和任务栏图标使用 click"
                ),
            ],
            success_criteria=["通过动作后界面证据确认用户指令已经完成"],
            metadata={"entrypoint": "src.agent.cli", "dry_run": config.dry_run},
        )


def build_default_runtime(config: CLIConfig) -> AgentRuntime:
    """Map the CLI to the existing GUI Agent modules.

    Imports are resolved lazily so ``python -m agent.cli --help`` works even
    before optional OCR, OpenCV, Qwen or LangChain dependencies are installed.
    """

    try:
        from .agent_chain import AgentChainConfig, create_agent_chain
        from .memory import AgentMemory
        from .planner import Planner, PlannerConfig
        from .state import AgentState
        from .tools import AgentTools
        from ..executor.executor import Executor
        from ..model.qwen_vlm import QwenVLM
        from ..perception.perception_pipeline import PerceptionPipeline
    except ImportError as error:
        raise CLIError(
            "无法导入 src.agent 运行模块。请确认从项目根目录执行 "
            "`python -m src.agent.cli`，并检查 src、src/agent、"
            "src/agent/prompts 均包含 __init__.py。"
            f"\n原始错误：{error}"
        ) from error

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise CLIError(
            "未设置 DASHSCOPE_API_KEY，无法连接 Qwen VLM。"
            "请先设置环境变量，或使用 --factory 注入自定义/测试 Agent。"
        )

    perception = PerceptionPipeline(
        enable_preprocessing=True,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
        include_unmatched_ocr=True,
        preprocess_options={
            "resize_width": 1280,
            "resize_height": None,
            "use_gray": False,
        },
    )
    executor = Executor(
        dry_run=config.dry_run,
        stop_on_failure=True,
        raise_on_error=False,
        default_wait_after_action=config.post_action_wait,
        keep_history=True,
    )
    vlm = QwenVLM(
        model=config.model,
        api_key=api_key,
        region=config.region,
        workspace_id=os.getenv("DASHSCOPE_WORKSPACE_ID"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        keep_history=False,
        enable_thinking=False,
    )
    planner = Planner(
        vlm=_PlannerVLMAdapter(vlm),
        config=PlannerConfig(
            max_attempts=2,
            temperature=0.0,
            max_tokens=400,
            include_screenshot=True,
            include_previous_observation=True,
            history_limit=10,
            validate_coordinates=True,
        ),
        action_factory=_planner_action_factory,
    )
    tools = AgentTools(perception=perception, executor=executor)
    memory = AgentMemory()
    chain_config = AgentChainConfig(
        post_action_wait_seconds=config.post_action_wait,
        max_reflections=config.max_reflections,
        max_model_repairs=1,
        max_chain_iterations=max(50, config.max_steps * 12),
    )
    chain = create_agent_chain(
        planner=planner,
        tools=tools,
        vlm=vlm,
        memory=memory,
        config=chain_config,
    )
    return AgentRuntime(
        chain=chain,
        state_cls=AgentState,
        components={
            "perception": perception,
            "vlm": vlm,
            "planner": planner,
            "executor": executor,
            "tools": tools,
            "memory": memory,
        },
    )


class AgentCLI:
    """Interactive command shell for submitting tasks to ``AgentChain``."""

    def __init__(
        self,
        config: CLIConfig,
        runtime_factory: Callable[[CLIConfig], AgentRuntime],
    ) -> None:
        self.config = config
        self.runtime_factory = runtime_factory
        self.runtime: AgentRuntime | None = None
        self.history: list[dict[str, Any]] = []
        self.last_result: Any = None
        self.logger, self.log_path = _build_file_logger(config.log_file)
        self.config.log_file = str(self.log_path)
        atexit.register(self._close_log)
        self.logger.info("CLI session started; log=%s", self.log_path)
        _flush_logs()

    def run(self, initial_task: str | None = None) -> int:
        self._print_banner()
        if initial_task:
            return 0 if self.execute_task(initial_task) else 1
        while True:
            try:
                line = input("\nagent> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                return 0
            if not line:
                continue
            try:
                if line.startswith("/"):
                    if not self.handle_command(line):
                        return 0
                else:
                    self.execute_task(line)
            except (CLIError, ValueError) as error:
                print(f"[配置错误] {error}")
            except KeyboardInterrupt:
                print("\n当前任务已由用户中断。")
            except Exception as error:
                print(f"[运行错误] {type(error).__name__}: {error}")
                if self.config.verbose:
                    import traceback

                    traceback.print_exc()

    def handle_command(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as error:
            raise CLIError(f"命令格式错误：{error}") from error
        command = parts[0].lower()
        args = parts[1:]
        if command in {"/exit", "/quit", "/q"}:
            print("已退出。")
            return False
        if command in {"/help", "/h", "/?"}:
            self._print_help()
        elif command == "/run":
            if not args:
                raise CLIError("用法：/run <任务指令>")
            self.execute_task(" ".join(args))
        elif command == "/config":
            print(json.dumps(asdict(self.config), ensure_ascii=False, indent=2))
        elif command == "/set":
            if len(args) != 2:
                raise CLIError("用法：/set <配置项> <值>")
            self.config.update(args[0], args[1])
            self.runtime = None
            print(f"已设置 {args[0]}={getattr(self.config, args[0], args[1])}")
        elif command == "/history":
            self._print_history()
        elif command == "/last":
            print(json.dumps(_json_safe(self.last_result), ensure_ascii=False, indent=2))
        elif command == "/reset":
            self.runtime = None
            self.last_result = None
            print("Agent 运行实例已重置，历史任务记录仍保留。")
        elif command == "/check":
            self._ensure_runtime()
            names = ", ".join(self.runtime.components)
            print(f"模块连接成功：{names}, chain")
        else:
            raise CLIError(f"未知命令：{command}。输入 /help 查看帮助。")
        return True

    def execute_task(self, task: str) -> bool:
        task = task.strip()
        if not task:
            raise CLIError("任务指令不能为空")

        started = time.perf_counter()
        self._emit("TASK", f"任务={task}")
        self._emit("STATUS", "Agent 已启动")

        attempts = 1 + max(0, int(self.config.task_retry_count))
        result: Any = None
        final_context: Mapping[str, Any] = {}
        captured_exception: BaseException | None = None
        captured_traceback = ""
        interrupted = False
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        try:
            runtime = self._ensure_runtime()
            for attempt in range(1, attempts + 1):
                state = runtime.new_state(task, self.config)
                captured_exception = None
                captured_traceback = ""
                final_context = {}
                self.logger.info("task=%r attempt=%d/%d started", task, attempt, attempts)
                previous_stage_at = time.perf_counter()

                if hasattr(runtime.chain, "stream_steps"):
                    final_state = state
                    for context in runtime.chain.stream_steps(state):
                        now = time.perf_counter()
                        stage_elapsed = now - previous_stage_at
                        previous_stage_at = now
                        final_context = context
                        final_state = context.get("agent_state", final_state)
                        next_stage = str(context.get("stage", "unknown"))
                        stage = str(context.get("completed_stage", next_stage))
                        stage_elapsed = float(
                            context.get("stage_elapsed_seconds", stage_elapsed)
                        )
                        step = getattr(final_state, "step_index", "?")
                        usage = _stage_usage(context, stage, next_stage)
                        for key in total_usage:
                            total_usage[key] += usage[key]
                        detail = _stage_detail(context, stage)
                        self._emit(
                            "STEP",
                            (
                                f"尝试={attempt}/{attempts} | 步骤={step} | "
                                f"完成={stage} → 当前={next_stage} | "
                                f"耗时={stage_elapsed:.3f}s | "
                                f"Token={usage['input_tokens']}/"
                                f"{usage['output_tokens']}/{usage['total_tokens']}"
                                + (f" | {detail}" if detail else "")
                            ),
                        )
                        self.logger.info(
                            "task=%r attempt=%d stage=%s step=%s "
                            "elapsed=%.3fs usage=%s detail=%s context=%s",
                            task,
                            attempt,
                            stage,
                            step,
                            stage_elapsed,
                            usage,
                            detail,
                            json.dumps(_json_value(context), ensure_ascii=False),
                        )
                        _flush_logs()
                    result = final_state.to_run_result()
                else:
                    result = runtime.chain.invoke(state)

                data = _json_safe(result)
                status = str(data.get("status", "unknown")).lower()
                succeeded = status in {
                    "success", "succeeded", "finished", "completed"
                }
                if succeeded:
                    break

                if attempt < attempts:
                    reason = _failure_reason(
                        result, final_context, captured_exception
                    )
                    self._emit("RETRY", f"本次执行未成功：{reason}")
                    self._emit("RETRY", "使用全新状态重新执行")
        except KeyboardInterrupt as error:
            interrupted = True
            captured_exception = error
            captured_traceback = traceback.format_exc()
            status = "interrupted"
            self._emit("INTERRUPT", "用户中断运行，正在保存当前日志")
            self.logger.warning(
                "task=%r interrupted by user; context=%s",
                task,
                json.dumps(_json_value(final_context), ensure_ascii=False),
            )
        except BaseException as error:
            captured_exception = error
            captured_traceback = traceback.format_exc()
            status = "error"
            self.logger.exception("task=%r terminated by %s", task, type(error).__name__)
        finally:
            _flush_logs()

        elapsed = time.perf_counter() - started
        self.last_result = result

        data = _json_safe(result) if result is not None else {}
        status = (
            "interrupted"
            if interrupted
            else str(data.get("status", "error" if captured_exception else "unknown"))
        )
        message = data.get("final_message") or data.get("message") or ""
        succeeded = status.lower() in {
            "success", "succeeded", "finished", "completed"
        }

        if not succeeded:
            reason = _failure_reason(result, final_context, captured_exception)
            self._emit("FAIL", f"阶段={final_context.get('stage', 'unknown')}")
            self._emit("FAIL", f"原因={reason}")
            error_data = (
                data.get("error")
                or final_context.get("error_details")
                or final_context.get("error")
            )
            if error_data:
                self._emit(
                    "ERROR",
                    json.dumps(_json_value(error_data), ensure_ascii=False),
                )
            chain_error = final_context.get("chain_error")
            if chain_error and str(chain_error) != reason:
                self._emit("ERROR", f"链路错误={chain_error}")
            if captured_traceback:
                self.logger.error("traceback:\n%s", captured_traceback.rstrip())
            self._emit("LOG", str(self.log_path))
            self.logger.error(
                "task=%r failed after %d attempts; reason=%s; result=%s; context=%s",
                task,
                attempts,
                reason,
                json.dumps(_json_value(data), ensure_ascii=False),
                json.dumps(_json_value(final_context), ensure_ascii=False),
            )

        self.history.append(
            {
                "task": task,
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
                "message": message,
            }
        )

        self._emit("RESULT", status)
        if message:
            self._emit("MESSAGE", str(message))
        self._emit(
            "SUMMARY",
            (
                f"总耗时={elapsed:.2f}s | Token="
                f"{total_usage['input_tokens']}/{total_usage['output_tokens']}/"
                f"{total_usage['total_tokens']} (输入/输出/总计)"
            ),
        )
        _flush_logs()

        return succeeded
    
    def _ensure_runtime(self) -> AgentRuntime:
        if self.runtime is None:
            self._emit("INIT", "连接 Perception/VLM/Planner/Executor/AgentChain")
            self.runtime = self.runtime_factory(self.config)
        return self.runtime

    def _emit(self, category: str, message: str) -> None:
        line = f"[{category:<9}] {message}"
        print(line, flush=True)
        self.logger.info(line)
        _flush_logs()

    def _close_log(self) -> None:
        if getattr(self, "logger", None) is None:
            return
        self.logger.info("CLI session closing")
        _flush_logs()

    def _print_stage(self, context: Mapping[str, Any]) -> None:
        stage = str(context.get("stage", "unknown"))
        state = context.get("agent_state")
        step = getattr(state, "step_index", "?")
        details = ""
        if stage == "verify":
            details = "，正在比较动作前后界面"
        elif stage == "reflect":
            details = "，正在分析失败原因并重新规划"
        elif stage == "execute":
            details = "，准备执行已校验动作"
        print(f"[步骤 {step}] {stage}{details}")

    def _print_history(self) -> None:
        if not self.history:
            print("当前会话还没有任务记录。")
            return
        for index, item in enumerate(self.history, 1):
            print(
                f"{index:>2}. [{item['status']}] {item['task']} "
                f"({item['elapsed_seconds']}s)"
            )

    @staticmethod
    def _print_banner() -> None:
        print(
            "\nGUI Agent 命令行\n"
            "直接输入自然语言任务即可执行；输入 /help 查看命令。"
        )

    @staticmethod
    def _print_help() -> None:
        print(
            "\n命令：\n"
            "  <自然语言任务>          直接提交给 Agent\n"
            "  /run <任务>             提交任务\n"
            "  /check                  检查模块连接\n"
            "  /config                 查看当前配置\n"
            "  /set <名称> <值>        修改配置并重建 Agent\n"
            "  /history                查看本次会话任务\n"
            "  /last                   查看上一次完整结果\n"
            "  /reset                  重建 Agent 运行实例\n"
            "  /exit                   退出\n\n"
            "示例：\n"
            "  /set dry_run true\n"
            "  /set max_steps 10\n"
            "  在 Name 输入框输入 GUIAgent，然后点击 Submit"
        )


class _PlannerVLMAdapter:
    """Keep Planner on ``generate`` while AgentChain may use structured calls."""

    def __init__(self, vlm: Any) -> None:
        self._vlm = vlm

    def generate(
        self,
        prompt: str,
        *,
        images: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:

        kwargs.pop("messages", None)
        kwargs.pop("image", None)
        return self._vlm.generate(prompt=prompt, images=images or [], **kwargs)


def _planner_action_factory(
    action_type: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_type = (
        str(action_type).strip().lower().replace("-", "_").replace(" ", "_")
    )
    normalized_type = {
        "doubleclick": "double_click",
        "dblclick": "double_click",
        "dbl_click": "double_click",
    }.get(normalized_type, normalized_type)
    params = dict(parameters)
    if normalized_type == "wait" and "duration" in params:
        params["seconds"] = params.pop("duration")
    planner_metadata = params.pop("metadata", {})
    metadata = (
        dict(planner_metadata)
        if isinstance(planner_metadata, Mapping)
        else {}
    )
    metadata["source"] = "agent_cli"
    return {
        "type": normalized_type,
        **params,
        "description": f"VLM planned action: {normalized_type}",
        "metadata": metadata,
    }


def _load_external_factory(spec: str) -> Callable[[CLIConfig], AgentRuntime]:
    if ":" not in spec:
        raise CLIError("--factory 格式必须为 module:function")
    module_name, function_name = spec.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise CLIError(f"{spec} 不是可调用的 Agent 工厂")
    return function


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise CLIError(f"无法解析布尔值：{value}")


def _json_safe(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    for method_name in ("to_dict", "summary", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return {str(k): _json_value(v) for k, v in result.items()}
    return {"value": _json_value(value)}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value"):
        return _json_value(value.value)
    return str(value)


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                data = method()
            except Exception:
                continue
            if isinstance(data, Mapping):
                return dict(data)
    result: dict[str, Any] = {}
    for name in (
        "type", "action_type", "x", "y", "target_text", "element_id",
        "metadata", "input_tokens", "output_tokens", "total_tokens",
    ):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = item
    return result


def _usage_dict(value: Any) -> dict[str, int]:
    data = _as_mapping(value)
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    result: dict[str, int] = {}
    for target, names in aliases.items():
        raw = next((data.get(name) for name in names if data.get(name) is not None), 0)
        try:
            result[target] = max(0, int(raw))
        except (TypeError, ValueError):
            result[target] = 0
    if not result["total_tokens"]:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def _stage_usage(
    context: Mapping[str, Any],
    completed_stage: str,
    next_stage: str,
) -> dict[str, int]:
    empty = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if completed_stage not in {"plan", "verify", "reflect", "memory"}:
        return empty
    state = context.get("agent_state")
    candidates = [
        context.get("usage"),
        _value(context.get("last_raw_response"), "usage"),
    ]
    if completed_stage == "plan":
        planner_result = _value(state, "last_planner_result")
        candidates.extend(
            [
                _value(planner_result, "usage"),
                _value(_value(planner_result, "metadata", {}), "usage"),
            ]
        )
    for candidate in candidates:
        usage = _usage_dict(candidate)
        if usage["total_tokens"]:
            return usage
    return empty


def _action_detail(action: Any) -> str:
    data = _as_mapping(action)
    metadata = _as_mapping(data.get("metadata"))
    validation = _as_mapping(metadata.get("target_validation"))
    action_type = data.get("type") or data.get("action_type") or "unknown"
    if hasattr(action_type, "value"):
        action_type = action_type.value
    x = data.get("x", validation.get("center_x"))
    y = data.get("y", validation.get("center_y"))
    target = (
        data.get("target_text")
        or validation.get("target_text")
        or validation.get("matched_text")
    )
    element_id = data.get("element_id", validation.get("element_id"))
    bbox = validation.get("matched_bbox") or metadata.get("matched_bbox")
    fields = [f"动作={action_type}"]
    if target:
        fields.append(f"目标={target!r}")
    if element_id is not None:
        fields.append(f"element_id={element_id}")
    if x is not None or y is not None:
        fields.append(f"坐标=({x},{y})")
    if bbox:
        fields.append(f"bbox={bbox}")
    return " | ".join(fields)


def _stage_detail(context: Mapping[str, Any], stage: str) -> str:
    state = context.get("agent_state")
    observation = _value(state, "observation")
    if stage in {"observe", "observe_after"}:
        width = _value(observation, "screen_width")
        height = _value(observation, "screen_height")
        elements = (
            _value(observation, "elements")
            or _value(observation, "merged_elements")
            or _value(_value(observation, "metadata", {}), "target_candidates")
            or []
        )
        elapsed = _value(observation, "elapsed_time")
        parts = []
        if width and height:
            parts.append(f"屏幕={width}x{height}")
        try:
            parts.append(f"元素={len(elements)}")
        except TypeError:
            pass
        if elapsed is not None:
            try:
                parts.append(f"感知耗时={float(elapsed):.3f}s")
            except (TypeError, ValueError):
                pass
        return " | ".join(parts)
    if stage in {"plan", "execute"}:
        action = _value(state, "latest_action")
        if action is None:
            action = _value(_value(state, "last_planner_result"), "action")
        return _action_detail(action) if action is not None else "未生成动作"
    if stage == "verify":
        data = _as_mapping(context.get("verify_data"))
        return (
            f"验证成功={data.get('success', data.get('completed', 'unknown'))}"
            f" | 置信度={data.get('confidence', 'unknown')}"
        )
    if stage == "reflect":
        data = _as_mapping(context.get("reflection_data"))
        return f"反思={data.get('reason') or data.get('analysis') or '重新规划'}"
    return ""


def _failure_reason(
    result: Any,
    context: Mapping[str, Any],
    error: BaseException | None,
) -> str:
    if error is not None:
        return f"{type(error).__name__}: {error}"
    chain_error = context.get("chain_error")
    if chain_error:
        return str(chain_error)
    data = _json_safe(result)
    error_data = data.get("error")
    if isinstance(error_data, Mapping):
        message = error_data.get("message")
        if message:
            return str(message)
    return str(
        data.get("final_message")
        or data.get("message")
        or data.get("termination_reason")
        or "任务未成功，但运行结果没有提供具体原因。"
    )


def _timestamped_log_path(log_file: str) -> Path:
    requested = Path(log_file)
    directory = requested if not requested.suffix else requested.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return directory / f"log_{stamp}.log"


def _build_file_logger(log_file: str) -> tuple[logging.Logger, Path]:
    path = _timestamped_log_path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=False)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logger = logging.getLogger("src.agent.cli")
    logger.setLevel(logging.INFO)
    return logger, path.resolve()


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUI Agent command-line interface")
    parser.add_argument("--task", help="执行一次任务后退出")
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "qwen3-vl-8b-instruct"))
    parser.add_argument("--region", default=os.getenv("QWEN_REGION", "frankfurt"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-reflections", type=int, default=3)
    parser.add_argument(
        "--task-retries",
        type=int,
        default=1,
        choices=(0, 1),
        help="整条任务失败后是否使用全新状态重试一次",
    )
    parser.add_argument(
        "--log-file",
        default="logs",
        help="日志目录（或任意旧式日志路径）；实际文件名自动生成为 log_当前时间.log",
    )
    parser.add_argument("--post-action-wait", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="禁止真实鼠标键盘操作")
    parser.add_argument(
        "--factory",
        help="自定义运行时工厂，格式为 module:function",
    )
    parser.add_argument("--quiet", action="store_true", help="关闭详细异常堆栈")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CLIConfig(
        model=args.model,
        region=args.region,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        max_reflections=args.max_reflections,
        post_action_wait=args.post_action_wait,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        task_retry_count=args.task_retries,
        log_file=args.log_file,
    )
    factory = _load_external_factory(args.factory) if args.factory else build_default_runtime
    return AgentCLI(config, factory).run(args.task)


if __name__ == "__main__":
    raise SystemExit(main())