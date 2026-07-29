"""
executor
1.接受经过验证的Action对象
2.将鼠标操作给MouseController
3.将键盘操作给KeyboardController
4.执行WAIT,FINISH,FAIL控制操作
5.执行ActionSequence
6.记录结构化执行结果
7.更新Action状态
8.支持dry_run
9.支持失败时停止
10.提供执行统计和历史记录
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .action import (
    Action,
    ActionSequence,
    ActionStatus,
    ActionType,
)
from .keyboard import (
    KeyboardActionResult,
    KeyboardController,
)
from .mouse import (
    MouseActionResult,
    MouseController,
)


logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:

    action: Action
    success: bool
    status: ActionStatus
    elapsed_time: float

    message: str = ""

    mouse_result: Optional[MouseActionResult] = None
    keyboard_result: Optional[KeyboardActionResult] = None

    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_id(self) -> str:
        """
        Return the ID of the executed action.
        """

        return self.action.action_id

    @property
    def action_type(self) -> ActionType:
        """
        Return the executed action type.
        """

        return self.action.type

    def summary(self) -> dict[str, Any]:
        """
        Return a JSON-compatible result summary.
        """

        return {
            "action_id": self.action.action_id,
            "action_type": self.action.type.value,
            "description": self.action.description,
            "success": self.success,
            "status": self.status.value,
            "elapsed_time_seconds": self.elapsed_time,
            "message": self.message,
            "error": self.error,
            "metadata": self.metadata,
        }
    

@dataclass
class SequenceExecutionResult:

    sequence: ActionSequence
    results: list[ExecutionResult] = field(default_factory=list)

    success: bool = True
    stopped_early: bool = False
    elapsed_time: float = 0.0

    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_actions(self) -> int:
        return len(self.sequence.actions)

    @property
    def executed_actions(self) -> int:
        return len(self.results)

    @property
    def successful_actions(self) -> int:
        return sum(
            1
            for result in self.results
            if result.success
        )

    @property
    def failed_actions(self) -> int:
        return sum(
            1
            for result in self.results
            if not result.success
        )

    def summary(self) -> dict[str, Any]:
        """
        Return a compact sequence execution summary.
        """

        return {
            "sequence_id": self.sequence.sequence_id,
            "description": self.sequence.description,
            "total_actions": self.total_actions,
            "executed_actions": self.executed_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "success": self.success,
            "stopped_early": self.stopped_early,
            "elapsed_time_seconds": self.elapsed_time,
            "error": self.error,
            "metadata": self.metadata,
        }
    

class Executor:

    def __init__(
        self,
        mouse: Optional[MouseController] = None,
        keyboard: Optional[KeyboardController] = None,
        dry_run: bool = False,
        stop_on_failure: bool = True,
        raise_on_error: bool = False,
        default_wait_after_action: float = 0.0,
        keep_history: bool = True,
    ) -> None:
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a bool.")

        if not isinstance(stop_on_failure, bool):
            raise TypeError(
                "stop_on_failure must be a bool."
            )

        if not isinstance(raise_on_error, bool):
            raise TypeError(
                "raise_on_error must be a bool."
            )

        if not isinstance(keep_history, bool):
            raise TypeError(
                "keep_history must be a bool."
            )

        self._validate_non_negative_number(
            default_wait_after_action,
            "default_wait_after_action",
        )

        self.dry_run = dry_run
        self.stop_on_failure = stop_on_failure
        self.raise_on_error = raise_on_error
        self.default_wait_after_action = float(
            default_wait_after_action
        )
        self.keep_history = keep_history

        self.mouse = mouse or MouseController(
            dry_run=dry_run,
            raise_on_error=raise_on_error,
        )

        self.keyboard = keyboard or KeyboardController(
            dry_run=dry_run,
            raise_on_error=raise_on_error,
        )

        # Ensure injected controllers follow Executor dry-run configuration.
        self.mouse.set_dry_run(dry_run)
        self.keyboard.set_dry_run(dry_run)

        self._history: list[ExecutionResult] = []
        self._lock = threading.RLock()
        self._stop_requested = False

        self._dispatch_table: dict[
            ActionType,
            Callable[[Action], ExecutionResult],
        ] = {
            # Mouse
            ActionType.MOVE_TO: self._execute_move_to,
            ActionType.MOVE_BY: self._execute_move_by,
            ActionType.CLICK: self._execute_click,
            ActionType.DOUBLE_CLICK: self._execute_double_click,
            ActionType.RIGHT_CLICK: self._execute_right_click,
            ActionType.MIDDLE_CLICK: self._execute_middle_click,
            ActionType.MOUSE_DOWN: self._execute_mouse_down,
            ActionType.MOUSE_UP: self._execute_mouse_up,
            ActionType.DRAG_TO: self._execute_drag_to,
            ActionType.DRAG_BY: self._execute_drag_by,
            ActionType.SCROLL: self._execute_scroll,
            ActionType.HORIZONTAL_SCROLL:
                self._execute_horizontal_scroll,

            # Keyboard
            ActionType.PRESS: self._execute_press,
            ActionType.HOTKEY: self._execute_hotkey,
            ActionType.TYPE_TEXT: self._execute_type_text,
            ActionType.PASTE_TEXT: self._execute_paste_text,
            ActionType.KEY_DOWN: self._execute_key_down,
            ActionType.KEY_UP: self._execute_key_up,

            ActionType.ENTER: self._execute_enter,
            ActionType.ESCAPE: self._execute_escape,
            ActionType.TAB: self._execute_tab,
            ActionType.BACKSPACE: self._execute_backspace,
            ActionType.DELETE: self._execute_delete,
            ActionType.COPY: self._execute_copy,
            ActionType.PASTE: self._execute_paste,
            ActionType.CUT: self._execute_cut,
            ActionType.SELECT_ALL: self._execute_select_all,
            ActionType.SAVE: self._execute_save,
            ActionType.UNDO: self._execute_undo,
            ActionType.REDO: self._execute_redo,
            ActionType.FIND: self._execute_find,
            ActionType.CLOSE_WINDOW: self._execute_close_window,
            ActionType.SWITCH_WINDOW: self._execute_switch_window,

            # Control
            ActionType.WAIT: self._execute_wait,
            ActionType.FINISH: self._execute_finish,
            ActionType.FAIL: self._execute_fail,
        }

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def execute(
        self,
        action: Action | dict[str, Any] | str,
    ) -> ExecutionResult:
        """
        Execute one action.

        Parameters
        ----------
        action:
            Action object, action dictionary or JSON string.
        """

        resolved_action = self._resolve_action(action)

        with self._lock:
            if self._stop_requested:
                resolved_action.mark_skipped(
                    "Executor stop was requested."
                )

                result = ExecutionResult(
                    action=resolved_action,
                    success=False,
                    status=ActionStatus.SKIPPED,
                    elapsed_time=0.0,
                    message="Action skipped because stop was requested.",
                    error="Executor stopped.",
                )

                self._record_result(result)
                return result

            logger.info(
                "Executing action: type=%s, id=%s, x=%s, y=%s, "
                "normalised=(%s,%s), screen=%sx%s, description=%s",
                resolved_action.type.value,
                resolved_action.action_id,
                resolved_action.x,
                resolved_action.y,
                resolved_action.normalised_x,
                resolved_action.normalised_y,
                getattr(self.mouse, "screen_width", None),
                getattr(self.mouse, "screen_height", None),
                resolved_action.description,
            )

            handler = self._dispatch_table.get(
                resolved_action.type
            )

            if handler is None:
                return self._handle_execution_error(
                    action=resolved_action,
                    error=NotImplementedError(
                        "No executor handler registered for "
                        f"{resolved_action.type.value!r}."
                    ),
                    start_time=time.perf_counter(),
                )

            start_time = time.perf_counter()
            resolved_action.mark_running()

            try:
                result = handler(resolved_action)

                if (
                    result.success
                    and self.default_wait_after_action > 0
                    and resolved_action.type
                    not in {
                        ActionType.WAIT,
                        ActionType.FINISH,
                        ActionType.FAIL,
                    }
                ):
                    time.sleep(
                        self.default_wait_after_action
                    )

                self._record_result(result)
                logger.info(
                    "Action completed: type=%s success=%s elapsed=%.3fs "
                    "requested_xy=(%s,%s)",
                    resolved_action.type.value,
                    result.success,
                    result.elapsed_time,
                    resolved_action.x,
                    resolved_action.y,
                )
                return result

            except Exception as error:
                result = self._handle_execution_error(
                    action=resolved_action,
                    error=error,
                    start_time=start_time,
                )

                self._record_result(result)

                if self.raise_on_error:
                    raise

                return result

    def execute_sequence(
        self,
        sequence: ActionSequence | Sequence[Action],
        stop_on_failure: Optional[bool] = None,
    ) -> SequenceExecutionResult:
        """
        Execute a sequence of actions in order.
        """

        resolved_sequence = self._resolve_sequence(
            sequence
        )

        should_stop_on_failure = (
            self.stop_on_failure
            if stop_on_failure is None
            else stop_on_failure
        )

        if not isinstance(
            should_stop_on_failure,
            bool,
        ):
            raise TypeError(
                "stop_on_failure must be a bool."
            )

        self._stop_requested = False
        start_time = time.perf_counter()

        sequence_result = SequenceExecutionResult(
            sequence=resolved_sequence
        )

        for index, action in enumerate(
            resolved_sequence.actions
        ):
            if self._stop_requested:
                sequence_result.stopped_early = True
                sequence_result.success = False
                sequence_result.error = (
                    "Execution stopped by request."
                )
                break

            logger.info(
                "Executing sequence action %d/%d: %s",
                index + 1,
                len(resolved_sequence.actions),
                action.type.value,
            )

            result = self.execute(action)
            sequence_result.results.append(result)

            if action.type == ActionType.FINISH:
                sequence_result.stopped_early = (
                    index + 1
                    < len(resolved_sequence.actions)
                )
                break

            if action.type == ActionType.FAIL:
                sequence_result.success = False
                sequence_result.stopped_early = True
                sequence_result.error = (
                    action.description
                    or "FAIL action executed."
                )
                break

            if not result.success:
                sequence_result.success = False
                sequence_result.error = result.error

                if should_stop_on_failure:
                    sequence_result.stopped_early = True
                    break

        sequence_result.elapsed_time = (
            time.perf_counter()
            - start_time
        )

        if sequence_result.failed_actions > 0:
            sequence_result.success = False

        return sequence_result

    def execute_json(
        self,
        json_text: str,
    ) -> ExecutionResult | SequenceExecutionResult:
        """
        Execute an Action or ActionSequence represented as JSON.

        Accepted formats
        ----------------
        Single action:
            {"type": "click", "x": 100, "y": 200}

        Sequence:
            {
                "description": "...",
                "actions": [...]
            }
        """

        if not isinstance(json_text, str):
            raise TypeError(
                "json_text must be a string."
            )

        import json

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid execution JSON: {error}"
            ) from error

        if not isinstance(parsed, dict):
            raise ValueError(
                "Execution JSON must represent an object."
            )

        if "actions" in parsed:
            return self.execute_sequence(
                ActionSequence.from_dict(parsed)
            )

        return self.execute(
            Action.from_dict(parsed)
        )

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def _execute_move_to(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_absolute_position(action)

        mouse_result = self.mouse.move_to(
            x=x,
            y=y,
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_move_by(
        self,
        action: Action,
    ) -> ExecutionResult:
        mouse_result = self.mouse.move_by(
            offset_x=self._require_value(
                action.offset_x,
                "offset_x",
            ),
            offset_y=self._require_value(
                action.offset_y,
                "offset_y",
            ),
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_click(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.click(
            x=x,
            y=y,
            button=action.button.value,
            clicks=action.clicks,
            interval=action.interval,
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_double_click(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.double_click(
            x=x,
            y=y,
            button=action.button.value,
            interval=action.interval,
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_right_click(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.right_click(
            x=x,
            y=y,
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_middle_click(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.middle_click(
            x=x,
            y=y,
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_mouse_down(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.mouse_down(
            x=x,
            y=y,
            button=action.button.value,
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_mouse_up(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.mouse_up(
            x=x,
            y=y,
            button=action.button.value,
            duration=action.duration,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_drag_to(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_absolute_position(action)

        mouse_result = self.mouse.drag_to(
            x=x,
            y=y,
            duration=action.duration,
            button=action.button.value,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_drag_by(
        self,
        action: Action,
    ) -> ExecutionResult:
        mouse_result = self.mouse.drag_by(
            offset_x=self._require_value(
                action.offset_x,
                "offset_x",
            ),
            offset_y=self._require_value(
                action.offset_y,
                "offset_y",
            ),
            duration=action.duration,
            button=action.button.value,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_scroll(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.scroll(
            amount=self._require_value(
                action.amount,
                "amount",
            ),
            x=x,
            y=y,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    def _execute_horizontal_scroll(
        self,
        action: Action,
    ) -> ExecutionResult:
        x, y = self._resolve_optional_position(action)

        mouse_result = self.mouse.horizontal_scroll(
            amount=self._require_value(
                action.amount,
                "amount",
            ),
            x=x,
            y=y,
        )

        return self._from_mouse_result(
            action,
            mouse_result,
        )

    # ------------------------------------------------------------------
    # Keyboard handlers
    # ------------------------------------------------------------------

    def _execute_press(
        self,
        action: Action,
    ) -> ExecutionResult:
        keyboard_result = self.keyboard.press(
            key=self._require_value(
                action.key,
                "key",
            ),
            presses=action.presses,
            interval=action.interval,
        )

        return self._from_keyboard_result(
            action,
            keyboard_result,
        )

    def _execute_hotkey(
        self,
        action: Action,
    ) -> ExecutionResult:
        keyboard_result = self.keyboard.hotkey(
            *action.keys,
            interval=action.interval,
        )

        return self._from_keyboard_result(
            action,
            keyboard_result,
        )

    def _execute_type_text(
        self,
        action: Action,
    ) -> ExecutionResult:
        keyboard_result = self.keyboard.write(
            text=self._require_value(
                action.text,
                "text",
            ),
            interval=action.interval,
            use_clipboard_for_unicode=(
                action.use_clipboard_for_unicode
            ),
        )

        return self._from_keyboard_result(
            action,
            keyboard_result,
        )

    def _execute_paste_text(
        self,
        action: Action,
    ) -> ExecutionResult:
        keyboard_result = self.keyboard.paste_text(
            text=self._require_value(
                action.text,
                "text",
            ),
            restore_clipboard=(
                action.restore_clipboard
            ),
        )

        return self._from_keyboard_result(
            action,
            keyboard_result,
        )

    def _execute_key_down(
        self,
        action: Action,
    ) -> ExecutionResult:
        keyboard_result = self.keyboard.key_down(
            self._require_value(
                action.key,
                "key",
            )
        )

        return self._from_keyboard_result(
            action,
            keyboard_result,
        )

    def _execute_key_up(
        self,
        action: Action,
    ) -> ExecutionResult:
        keyboard_result = self.keyboard.key_up(
            self._require_value(
                action.key,
                "key",
            )
        )

        return self._from_keyboard_result(
            action,
            keyboard_result,
        )

    def _execute_enter(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.enter(),
        )

    def _execute_escape(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.escape(),
        )

    def _execute_tab(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.tab(action.presses),
        )

    def _execute_backspace(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.backspace(
                action.presses
            ),
        )

    def _execute_delete(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.delete(
                action.presses
            ),
        )

    def _execute_copy(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.copy(),
        )

    def _execute_paste(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.paste(),
        )

    def _execute_cut(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.cut(),
        )

    def _execute_select_all(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.select_all(),
        )

    def _execute_save(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.save(),
        )

    def _execute_undo(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.undo(),
        )

    def _execute_redo(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.redo(),
        )

    def _execute_find(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.find(),
        )

    def _execute_close_window(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.close_window(),
        )

    def _execute_switch_window(
        self,
        action: Action,
    ) -> ExecutionResult:
        return self._from_keyboard_result(
            action,
            self.keyboard.switch_window(),
        )

    # ------------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------------

    def _execute_wait(
        self,
        action: Action,
    ) -> ExecutionResult:
        seconds = float(
            self._require_value(
                action.seconds,
                "seconds",
            )
        )

        start_time = time.perf_counter()

        if not self.dry_run:
            time.sleep(seconds)

        elapsed = (
            time.perf_counter()
            - start_time
        )

        action.mark_success()

        return ExecutionResult(
            action=action,
            success=True,
            status=ActionStatus.SUCCESS,
            elapsed_time=elapsed,
            message=f"Waited for {seconds:.3f} seconds.",
            metadata={
                "seconds": seconds,
                "dry_run": self.dry_run,
            },
        )

    def _execute_finish(
        self,
        action: Action,
    ) -> ExecutionResult:
        action.mark_success()

        return ExecutionResult(
            action=action,
            success=True,
            status=ActionStatus.SUCCESS,
            elapsed_time=0.0,
            message=(
                action.description
                or "Task completed."
            ),
        )

    def _execute_fail(
        self,
        action: Action,
    ) -> ExecutionResult:
        message = (
            action.description
            or "Task failed."
        )

        action.mark_failed(message)

        return ExecutionResult(
            action=action,
            success=False,
            status=ActionStatus.FAILED,
            elapsed_time=0.0,
            message=message,
            error=message,
        )

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _from_mouse_result(
        action: Action,
        result: MouseActionResult,
    ) -> ExecutionResult:
        if result.success:
            action.mark_success()
            status = ActionStatus.SUCCESS
        else:
            action.mark_failed(result.error)
            status = ActionStatus.FAILED

        return ExecutionResult(
            action=action,
            success=result.success,
            status=status,
            elapsed_time=result.elapsed_time,
            message=result.message,
            mouse_result=result,
            error=result.error,
            metadata=result.metadata or {},
        )

    @staticmethod
    def _from_keyboard_result(
        action: Action,
        result: KeyboardActionResult,
    ) -> ExecutionResult:
        if result.success:
            action.mark_success()
            status = ActionStatus.SUCCESS
        else:
            action.mark_failed(result.error)
            status = ActionStatus.FAILED

        return ExecutionResult(
            action=action,
            success=result.success,
            status=status,
            elapsed_time=result.elapsed_time,
            message=result.message,
            keyboard_result=result,
            error=result.error,
            metadata=result.metadata or {},
        )

    # ------------------------------------------------------------------
    # Position resolution
    # ------------------------------------------------------------------

    def _resolve_absolute_position(
        self,
        action: Action,
    ) -> tuple[int, int]:
        if (
            action.normalised_x is not None
            and action.normalised_y is not None
        ):
            return self.mouse.normalised_to_screen(
                action.normalised_x,
                action.normalised_y,
            )

        return (
            self._require_value(action.x, "x"),
            self._require_value(action.y, "y"),
        )

    def _resolve_optional_position(
        self,
        action: Action,
    ) -> tuple[Optional[int], Optional[int]]:
        if (
            action.normalised_x is not None
            and action.normalised_y is not None
        ):
            return self.mouse.normalised_to_screen(
                action.normalised_x,
                action.normalised_y,
            )

        return action.x, action.y

    # ------------------------------------------------------------------
    # History and control
    # ------------------------------------------------------------------

    @property
    def history(self) -> tuple[ExecutionResult, ...]:
        """
        Return immutable execution history.
        """

        return tuple(self._history)

    def clear_history(self) -> None:
        """
        Clear execution history.
        """

        self._history.clear()

    def request_stop(self) -> None:
        """
        Request sequence execution to stop.
        """

        self._stop_requested = True

    def reset_stop(self) -> None:
        """
        Clear the stop request.
        """

        self._stop_requested = False

    def set_dry_run(
        self,
        enabled: bool,
    ) -> None:
        """
        Update dry-run mode for Executor and controllers.
        """

        if not isinstance(enabled, bool):
            raise TypeError(
                "enabled must be a bool."
            )

        self.dry_run = enabled
        self.mouse.set_dry_run(enabled)
        self.keyboard.set_dry_run(enabled)

    def history_summary(self) -> dict[str, Any]:
        """
        Return execution-history statistics.
        """

        successful = sum(
            1
            for result in self._history
            if result.success
        )

        failed = len(self._history) - successful

        return {
            "total_actions": len(self._history),
            "successful_actions": successful,
            "failed_actions": failed,
            "total_elapsed_time_seconds": sum(
                result.elapsed_time
                for result in self._history
            ),
            "dry_run": self.dry_run,
        }

    def _record_result(
        self,
        result: ExecutionResult,
    ) -> None:
        if self.keep_history:
            self._history.append(result)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _handle_execution_error(
        self,
        action: Action,
        error: Exception,
        start_time: float,
    ) -> ExecutionResult:
        elapsed = (
            time.perf_counter()
            - start_time
        )

        error_message = str(error)
        action.mark_failed(error_message)

        logger.exception(
            "Action execution failed: type=%s, id=%s",
            action.type.value,
            action.action_id,
        )

        return ExecutionResult(
            action=action,
            success=False,
            status=ActionStatus.FAILED,
            elapsed_time=elapsed,
            message=(
                action.description
                or f"Failed to execute {action.type.value}."
            ),
            error=error_message,
            metadata={
                "exception_type":
                    type(error).__name__,
            },
        )

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_action(
        action: Action | dict[str, Any] | str,
    ) -> Action:
        if isinstance(action, Action):
            return action

        if isinstance(action, dict):
            return Action.from_dict(action)

        if isinstance(action, str):
            return Action.from_json(action)

        raise TypeError(
            "action must be Action, dictionary or JSON string."
        )

    @staticmethod
    def _resolve_sequence(
        sequence: ActionSequence | Sequence[Action],
    ) -> ActionSequence:
        if isinstance(sequence, ActionSequence):
            return sequence

        if isinstance(sequence, Sequence):
            return ActionSequence(
                actions=list(sequence)
            )

        raise TypeError(
            "sequence must be ActionSequence "
            "or a sequence of Action objects."
        )

    # ------------------------------------------------------------------
    # Primitive helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_value(
        value: Any,
        name: str,
    ) -> Any:
        if value is None:
            raise ValueError(
                f"{name} is required for this action."
            )

        return value

    @staticmethod
    def _validate_non_negative_number(
        value: Any,
        name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be an int or float."
            )

        if value < 0:
            raise ValueError(
                f"{name} must be non-negative."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dry_run={self.dry_run}, "
            f"stop_on_failure={self.stop_on_failure}, "
            f"raise_on_error={self.raise_on_error}, "
            f"default_wait_after_action="
            f"{self.default_wait_after_action}, "
            f"keep_history={self.keep_history}"
            f")"
        )