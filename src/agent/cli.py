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
    log_file: str = "logs/agent_debug.log"

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
        self.logger = _build_file_logger(config.log_file)

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

        runtime = self._ensure_runtime()
        started = time.perf_counter()
        print(f"\n[任务] {task}")
        print("[状态] Agent 已启动")

        attempts = 1 + max(0, int(self.config.task_retry_count))
        result: Any = None
        final_context: Mapping[str, Any] = {}
        captured_exception: BaseException | None = None
        captured_traceback = ""

        for attempt in range(1, attempts + 1):
            state = runtime.new_state(task, self.config)
            captured_exception = None
            captured_traceback = ""
            final_context = {}
            self.logger.info("task=%r attempt=%d/%d started", task, attempt, attempts)

            try:
                if hasattr(runtime.chain, "stream_steps"):
                    final_state = state
                    for context in runtime.chain.stream_steps(state):
                        final_context = context
                        final_state = context.get("agent_state", final_state)
                        stage = str(context.get("stage", "unknown"))
                        step = getattr(final_state, "step_index", "?")
                        print(f"[尝试 {attempt}/{attempts}][步骤 {step}] {stage}")
                        self.logger.info(
                            "task=%r attempt=%d context=%s",
                            task,
                            attempt,
                            json.dumps(_json_value(context), ensure_ascii=False),
                        )
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
            except Exception as error:
                captured_exception = error
                captured_traceback = traceback.format_exc()
                status = "error"
                self.logger.exception(
                    "task=%r attempt=%d/%d raised an exception",
                    task,
                    attempt,
                    attempts,
                )

            if attempt < attempts:
                reason = _failure_reason(
                    result, final_context, captured_exception
                )
                print(f"[重试] 第一次执行未成功：{reason}")
                print("[重试] 正在使用全新状态重试一次...")

        elapsed = time.perf_counter() - started
        self.last_result = result

        data = _json_safe(result) if result is not None else {}
        status = str(data.get("status", "error" if captured_exception else "unknown"))
        message = data.get("final_message") or data.get("message") or ""
        succeeded = status.lower() in {
            "success", "succeeded", "finished", "completed"
        }

        if not succeeded:
            reason = _failure_reason(result, final_context, captured_exception)
            print("\n[失败诊断]")
            print(f"[失败阶段] {final_context.get('stage', 'unknown')}")
            print(f"[失败原因] {reason}")
            error_data = (
                data.get("error")
                or final_context.get("error_details")
                or final_context.get("error")
            )
            if error_data:
                print("[错误详情]")
                print(json.dumps(_json_value(error_data), ensure_ascii=False, indent=2))
            chain_error = final_context.get("chain_error")
            if chain_error and str(chain_error) != reason:
                print(f"[链路错误] {chain_error}")
            if captured_traceback:
                print("[完整异常堆栈]")
                print(captured_traceback.rstrip())
            print(f"[完整日志] {Path(self.config.log_file)}")
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

        print(f"\n[结果] {status}")
        if message:
            print(f"[说明] {message}")
        print(f"[耗时] {elapsed:.2f}s")

        return succeeded
    
    def _ensure_runtime(self) -> AgentRuntime:
        if self.runtime is None:
            print("[初始化] 正在连接感知、VLM、Planner、Executor 与 AgentChain...")
            self.runtime = self.runtime_factory(self.config)
        return self.runtime

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


def _build_file_logger(log_file: str) -> logging.Logger:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("src.agent.cli")
    logger.setLevel(logging.INFO)
    resolved = str(path.resolve())
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == resolved
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(handler)
    return logger


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
    parser.add_argument("--log-file", default="logs/agent_debug.log")
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