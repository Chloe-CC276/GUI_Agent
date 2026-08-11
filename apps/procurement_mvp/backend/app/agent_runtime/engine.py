"""Observe→Decide→Act→Verify task engine backed by AgentTask rows."""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..errors import AppError
from ..models import AgentTask, OAApplication, utcnow
from ..oa_services import normalize_procurement_status
from .excel_reader import (
    list_excel_files,
    parse_oa_excel,
    validate_application_payload,
)
from .intents import QUICK_CHIPS, route_message
from .tasks import build_import_purchase_steps, build_submit_approved_steps

MAX_STEP_RETRIES = 2
GUI_OPERATIONS = {"import_purchase_to_oa", "submit_approved_purchase"}


def _new_task_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _task_view(task: AgentTask) -> dict[str, Any]:
    context = dict(task.context_json or {})
    result = dict(task.result or {})
    return {
        "task_id": task.task_id,
        "business_key": task.business_key,
        "operation": task.operation,
        "status": task.status,
        "result": result,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "current_route": task.current_route,
        "context_json": context,
        "is_paused": bool(task.is_paused),
        "steps": result.get("steps") or context.get("steps") or [],
        "messages": result.get("messages") or [],
        "waiting": context.get("waiting"),
        "application": context.get("application"),
        "chips": QUICK_CHIPS,
    }


def get_agent_task_view(session: Session, task_id: str) -> dict[str, Any]:
    task = session.scalar(
        select(AgentTask).where(AgentTask.task_id == task_id).order_by(AgentTask.id.desc())
    )
    if task is None:
        raise AppError(404, "TASK_NOT_FOUND", "Agent task not found")
    return _task_view(task)


def _append_message(task: AgentTask, role: str, content: str) -> None:
    result = dict(task.result or {})
    messages = list(result.get("messages") or [])
    messages.append({"role": role, "content": content, "at": utcnow().isoformat()})
    result["messages"] = messages[-50:]
    task.result = result


def _set_waiting(task: AgentTask, waiting: dict[str, Any], message: str) -> None:
    context = dict(task.context_json or {})
    context["waiting"] = waiting
    task.context_json = context
    task.status = "wait_user"
    task.is_paused = False
    _append_message(task, "assistant", message)


def _clear_waiting(task: AgentTask) -> None:
    context = dict(task.context_json or {})
    context.pop("waiting", None)
    task.context_json = context


def _latest_gui_task(session: Session) -> AgentTask | None:
    return session.scalar(
        select(AgentTask)
        .where(AgentTask.operation.in_(sorted(GUI_OPERATIONS)))
        .order_by(AgentTask.id.desc())
    )


def _create_task(
    session: Session,
    *,
    operation: str,
    business_key: str,
    context: dict[str, Any],
    message: str,
) -> AgentTask:
    task = AgentTask(
        task_id=_new_task_id(operation.replace("_", "-")),
        business_key=business_key,
        operation=operation,
        status="pending",
        is_paused=False,
        current_route=context.get("route"),
        context_json=context,
        result={"messages": [], "steps": []},
    )
    session.add(task)
    session.flush()
    _append_message(task, "user", message)
    return task


def _activate_import_steps(task: AgentTask, application: dict[str, Any]) -> None:
    steps = build_import_purchase_steps(application)
    context = dict(task.context_json or {})
    context["application"] = {
        **application,
        "total_budget": str(application["total_budget"]),
        "line_total": str(application["line_total"]),
        "lines": [
            {
                **line,
                "quantity": str(line["quantity"]),
                "estimated_unit_price": str(line["estimated_unit_price"]),
                "line_amount": str(line["line_amount"]),
            }
            for line in application["lines"]
        ],
    }
    context["step_index"] = 0
    task.context_json = context
    result = dict(task.result or {})
    result["steps"] = steps
    task.result = result
    task.status = "running"
    _clear_waiting(task)
    _append_message(task, "assistant", "校验通过，开始在 OA 表单逐项填写并保存草稿。")


def _prepare_import_from_bytes(
    task: AgentTask, content: bytes, *, department: str | None
) -> None:
    application = parse_oa_excel(content, department_filter=department)
    errors = validate_application_payload(application)
    if errors:
        _set_waiting(
            task,
            {
                "type": "validation_errors",
                "errors": errors,
                "allow_upload": True,
            },
            "Excel 校验失败，请修正后重新上传：" + "；".join(errors),
        )
        return
    _activate_import_steps(task, application)


def _prepare_import_from_path(
    task: AgentTask, path: str, *, department: str | None
) -> None:
    application = parse_oa_excel(path, department_filter=department)
    errors = validate_application_payload(application)
    if errors:
        _set_waiting(
            task,
            {"type": "validation_errors", "errors": errors, "allow_upload": True},
            "Excel 校验失败，请修正后重新选择：" + "；".join(errors),
        )
        return
    context = dict(task.context_json or {})
    context["excel_path"] = path
    task.context_json = context
    _activate_import_steps(task, application)


def _start_import_purchase_to_oa(
    session: Session,
    *,
    message: str,
    params: dict[str, Any],
    page_context: dict[str, Any],
) -> AgentTask:
    department = params.get("department") or "生产部"
    task = _create_task(
        session,
        operation="import_purchase_to_oa",
        business_key=f"import-{department}",
        context={
            "department": department,
            "route": page_context.get("route"),
            "folder_path": params.get("folder_path"),
            "excel_path": params.get("excel_path"),
        },
        message=message,
    )
    folder_path = params.get("folder_path")
    excel_path = params.get("excel_path")
    if excel_path:
        _prepare_import_from_path(task, excel_path, department=department)
    elif folder_path:
        files = list_excel_files(folder_path)
        if not files:
            _set_waiting(
                task,
                {"type": "select_folder", "reason": "empty"},
                f"目录中未找到 Excel，请重新选择文件夹。",
            )
        elif len(files) == 1:
            _prepare_import_from_path(task, files[0]["path"], department=department)
        else:
            _set_waiting(
                task,
                {"type": "select_excel", "files": files, "folder_path": folder_path},
                f"发现 {len(files)} 个 Excel，请确认要导入的文件。",
            )
    else:
        _set_waiting(
            task,
            {"type": "select_folder", "department": department},
            "未提供文件路径，请选择包含采购申请 Excel 的文件夹（Agent 不会猜测路径）。",
        )
    session.commit()
    return task


def _candidate_approved_apps(session: Session) -> list[OAApplication]:
    apps = session.scalars(
        select(OAApplication)
        .options(selectinload(OAApplication.lines))
        .where(OAApplication.status == "APPROVED")
        .order_by(OAApplication.id.asc())
    ).all()
    result = []
    for app in apps:
        if normalize_procurement_status(app.procurement_status) != "NOT_STARTED":
            continue
        if app.linked_pr_no:
            continue
        result.append(app)
    return result


def _activate_submit_steps(session: Session, task: AgentTask, app: OAApplication) -> None:
    line_total = sum(
        (Decimal(str(line.quantity)) * Decimal(str(line.estimated_unit_price)) for line in app.lines),
        Decimal("0"),
    ).quantize(Decimal("0.01"))
    expected_total = Decimal(str(app.total_budget or line_total)).quantize(Decimal("0.01"))
    steps = build_submit_approved_steps(
        oa_id=app.id,
        application_no=app.application_no,
        expected_header={
            "title": app.title,
            "department": app.department,
            "applicant": app.applicant,
            "application_no": app.application_no,
        },
        expected_lines=[
            {
                "item_name": line.item_name,
                "quantity": str(line.quantity),
                "estimated_unit_price": str(line.estimated_unit_price),
                "line_amount": str(
                    (Decimal(str(line.quantity)) * Decimal(str(line.estimated_unit_price))).quantize(
                        Decimal("0.01")
                    )
                ),
            }
            for line in app.lines
        ],
        expected_total=str(expected_total),
    )
    context = dict(task.context_json or {})
    context.update(
        {
            "oa_id": app.id,
            "application_no": app.application_no,
            "step_index": 0,
            "expected_total": str(expected_total),
        }
    )
    task.context_json = context
    result = dict(task.result or {})
    result["steps"] = steps
    task.result = result
    task.business_key = app.application_no
    task.status = "running"
    _clear_waiting(task)
    _append_message(
        task,
        "assistant",
        f"已选定 {app.application_no}，开始核对并提交采购。",
    )


def _start_submit_approved_purchase(
    session: Session,
    *,
    message: str,
    page_context: dict[str, Any],
    selected_oa_id: int | None = None,
) -> AgentTask:
    task = _create_task(
        session,
        operation="submit_approved_purchase",
        business_key="submit-approved",
        context={"route": page_context.get("route")},
        message=message,
    )
    candidates = _candidate_approved_apps(session)
    if selected_oa_id is not None:
        hit = next((item for item in candidates if item.id == selected_oa_id), None)
        if hit is None:
            raise AppError(404, "OA_NOT_FOUND", "Selected OA is not eligible")
        _activate_submit_steps(session, task, hit)
        session.commit()
        return task
    if not candidates:
        task.status = "failed"
        _append_message(task, "assistant", "没有找到 APPROVED 且未开始采购的申请。")
        session.commit()
        return task
    if len(candidates) > 1:
        _set_waiting(
            task,
            {
                "type": "select_oa",
                "options": [
                    {
                        "id": item.id,
                        "application_no": item.application_no,
                        "title": item.title,
                        "department": item.department,
                        "total_budget": str(item.total_budget),
                    }
                    for item in candidates
                ],
            },
            f"命中 {len(candidates)} 条已通过且未开始采购的申请，请选择目标（不会自动随机选择）。",
        )
        session.commit()
        return task
    _activate_submit_steps(session, task, candidates[0])
    session.commit()
    return task


def create_task_from_message(
    session: Session,
    *,
    message: str,
    page_context: dict[str, Any] | None = None,
    folder_path: str | None = None,
    excel_path: str | None = None,
) -> dict[str, Any]:
    page_context = page_context or {}
    routed = route_message(message, context=page_context)
    intent = routed.get("intent")
    params = dict(routed.get("params") or {})
    if folder_path:
        params["folder_path"] = folder_path
    if excel_path:
        params["excel_path"] = excel_path

    if intent == "unknown":
        return {
            "intent": intent,
            "reply": routed.get("reply"),
            "chips": QUICK_CHIPS,
            "task": None,
        }

    if intent == "view_current_task":
        latest = _latest_gui_task(session)
        if latest is None:
            return {"intent": intent, "reply": "当前没有 GUI Agent 任务。", "chips": QUICK_CHIPS, "task": None}
        return {"intent": intent, "reply": f"当前任务 {latest.task_id} 状态 {latest.status}", "chips": QUICK_CHIPS, "task": _task_view(latest)}

    if intent == "resume_last_task":
        latest = _latest_gui_task(session)
        if latest is None:
            return {"intent": intent, "reply": "没有可继续的任务。", "chips": QUICK_CHIPS, "task": None}
        if latest.status == "paused":
            latest.is_paused = False
            latest.status = "running" if (latest.result or {}).get("steps") else latest.status
            session.commit()
        elif latest.status == "wait_user":
            pass
        elif latest.status == "stopped":
            return {"intent": intent, "reply": "上次任务已停止，请重新发起。", "chips": QUICK_CHIPS, "task": _task_view(latest)}
        return {"intent": intent, "reply": f"已定位任务 {latest.task_id}", "chips": QUICK_CHIPS, "task": _task_view(latest)}

    if intent == "import_purchase_to_oa":
        task = _start_import_purchase_to_oa(
            session, message=message, params=params, page_context=page_context
        )
        return {
            "intent": intent,
            "reply": "已创建导入任务。",
            "chips": QUICK_CHIPS,
            "task": _task_view(task),
        }

    if intent == "submit_approved_purchase":
        task = _start_submit_approved_purchase(
            session, message=message, page_context=page_context
        )
        return {
            "intent": intent,
            "reply": "已创建提交采购任务。",
            "chips": QUICK_CHIPS,
            "task": _task_view(task),
        }

    raise AppError(400, "UNKNOWN_INTENT", f"Unsupported intent: {intent}")


def continue_agent_task(
    session: Session,
    task_id: str,
    *,
    payload: dict[str, Any],
    upload: bytes | None = None,
    upload_name: str | None = None,
) -> dict[str, Any]:
    task = session.scalar(
        select(AgentTask).where(AgentTask.task_id == task_id).order_by(AgentTask.id.desc())
    )
    if task is None:
        raise AppError(404, "TASK_NOT_FOUND", "Agent task not found")
    if task.status == "stopped":
        raise AppError(409, "TASK_STOPPED", "Task already stopped")

    waiting = (task.context_json or {}).get("waiting") or {}
    wait_type = waiting.get("type")
    department = (task.context_json or {}).get("department")

    if upload is not None:
        _append_message(task, "user", f"上传文件 {upload_name or 'upload.xlsx'}")
        _prepare_import_from_bytes(task, upload, department=department)
        session.commit()
        return _task_view(task)

    if wait_type == "select_folder":
        folder_path = payload.get("folder_path")
        if not folder_path:
            raise AppError(400, "FOLDER_REQUIRED", "folder_path is required")
        files = list_excel_files(folder_path)
        if not files:
            _set_waiting(
                task,
                {"type": "select_folder", "reason": "empty"},
                "目录中未找到 Excel，请重新选择。",
            )
        elif len(files) == 1:
            _prepare_import_from_path(task, files[0]["path"], department=department)
        else:
            _set_waiting(
                task,
                {"type": "select_excel", "files": files, "folder_path": folder_path},
                f"发现 {len(files)} 个 Excel，请确认文件。",
            )
        session.commit()
        return _task_view(task)

    if wait_type == "select_excel":
        excel_path = payload.get("excel_path")
        if not excel_path:
            raise AppError(400, "EXCEL_REQUIRED", "excel_path is required")
        # Ensure selected file belongs to offered list when available.
        offered = {item["path"] for item in waiting.get("files") or []}
        if offered and str(Path(excel_path).resolve()) not in {
            str(Path(item).resolve()) for item in offered
        } and excel_path not in offered:
            # allow exact string match from UI
            if excel_path not in offered:
                raise AppError(400, "EXCEL_INVALID", "excel_path is not in candidate list")
        _prepare_import_from_path(task, excel_path, department=department)
        session.commit()
        return _task_view(task)

    if wait_type == "select_oa":
        oa_id = payload.get("oa_id")
        if oa_id is None:
            raise AppError(400, "OA_REQUIRED", "oa_id is required")
        candidates = _candidate_approved_apps(session)
        hit = next((item for item in candidates if item.id == int(oa_id)), None)
        if hit is None:
            raise AppError(404, "OA_NOT_FOUND", "Selected OA is not eligible")
        _activate_submit_steps(session, task, hit)
        session.commit()
        return _task_view(task)

    if wait_type == "validation_errors":
        raise AppError(400, "UPLOAD_REQUIRED", "Please upload a corrected Excel file")

    if wait_type == "step_failed":
        # User acknowledged and asked to retry current step from scratch count reset optional
        result = dict(task.result or {})
        steps = list(result.get("steps") or [])
        index = int((task.context_json or {}).get("step_index") or 0)
        if 0 <= index < len(steps):
            steps[index]["retry_count"] = 0
            steps[index]["status"] = "pending"
            steps[index]["actual"] = None
        result["steps"] = steps
        task.result = result
        task.status = "running"
        _clear_waiting(task)
        _append_message(task, "assistant", "已根据你的确认继续重试当前步骤。")
        session.commit()
        return _task_view(task)

    raise AppError(409, "NOT_WAITING", "Task is not waiting for user input", {"status": task.status})


def report_step_result(
    session: Session,
    task_id: str,
    *,
    step_id: str,
    status: str,
    actual: Any = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = session.scalar(
        select(AgentTask).where(AgentTask.task_id == task_id).order_by(AgentTask.id.desc())
    )
    if task is None:
        raise AppError(404, "TASK_NOT_FOUND", "Agent task not found")
    if task.status == "stopped":
        raise AppError(409, "TASK_STOPPED", "Task already stopped")
    if task.is_paused or task.status == "paused":
        raise AppError(409, "TASK_PAUSED", "Task is paused")

    result = dict(task.result or {})
    steps = list(result.get("steps") or [])
    index = next((i for i, step in enumerate(steps) if step.get("step_id") == step_id), None)
    if index is None:
        raise AppError(404, "STEP_NOT_FOUND", f"Step not found: {step_id}")

    step = dict(steps[index])
    step["actual"] = actual if actual is not None else (detail or {})
    if status == "passed":
        step["status"] = "passed"
        steps[index] = step
        result["steps"] = steps
        task.result = result
        context = dict(task.context_json or {})
        context["step_index"] = index + 1
        task.context_json = context
        if index + 1 >= len(steps):
            task.status = "completed"
            if task.operation == "import_purchase_to_oa":
                app_no = (detail or {}).get("application_no") or (actual or {}).get("application_no")
                if app_no:
                    context["application_no"] = app_no
                    task.context_json = context
                    task.business_key = str(app_no)
                _append_message(
                    task,
                    "assistant",
                    f"草稿已保存。approval_status=DRAFT，procurement_status=NOT_STARTED，OA号={app_no or '-'}。",
                )
            else:
                pr_no = (detail or {}).get("linked_pr_no") or (actual or {}).get("linked_pr_no")
                _append_message(
                    task,
                    "assistant",
                    f"提交采购完成。approval_status=APPROVED，procurement_status=PREPARING"
                    + (f"，linked_pr_no={pr_no}" if pr_no else "")
                    + "。",
                )
        else:
            task.status = "running"
        session.commit()
        return _task_view(task)

    # failed verify → retry or wait_user
    retry_count = int(step.get("retry_count") or 0) + 1
    step["retry_count"] = retry_count
    step["status"] = "failed"
    steps[index] = step
    result["steps"] = steps
    task.result = result
    if retry_count > MAX_STEP_RETRIES:
        _set_waiting(
            task,
            {
                "type": "step_failed",
                "step_id": step_id,
                "expected": step.get("expected"),
                "actual": step.get("actual"),
                "retry_count": retry_count,
            },
            f"步骤 {step_id} 验证失败且已重试 {MAX_STEP_RETRIES} 次，等待你处理（不会自动跳过）。",
        )
    else:
        step["status"] = "pending"
        steps[index] = step
        result["steps"] = steps
        task.result = result
        task.status = "running"
        _append_message(
            task,
            "assistant",
            f"步骤 {step_id} 验证失败，准备第 {retry_count} 次重试。",
        )
    session.commit()
    return _task_view(task)
