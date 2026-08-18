"""ERP PO creation via GUI Agent (Scheme A) — task orchestration & dashboards.

PRD v1.1 Phase 1: draft → pre-save verify → save → readback → upstream writeback.
VLM is reserved via vlm_adapter but not invoked.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from .errors import AppError
from .excel_tracking import write_batch_excel
from .models import (
    AgentBatch,
    AgentSafetyLog,
    AgentStepLog,
    AgentTask,
    BusinessLineage,
    ERPPurchaseOrder,
    ERPPurchaseOrderLine,
    IntegrationTransfer,
    ProcurementRequest,
    utcnow,
)
from .oa_services import (
    PROCUREMENT_STATUS_AWARDED,
    normalize_procurement_status,
    set_procurement_status,
)
from .services import _unique_number, _upsert_lineage, money
from .vlm_adapter import vlm_adapter

PO_OPERATION = "create_erp_po"

# v1.1 state machine (primary)
TASK_WAITING = "WAITING"
TASK_PREPARING = "PREPARING"
TASK_DRAFT_EDITING = "DRAFT_EDITING"
TASK_PRE_SAVE_VERIFY = "PRE_SAVE_VERIFY"
TASK_SAVING = "SAVING"
TASK_READBACK = "READBACK"
TASK_SUCCESS = "SUCCESS"
TASK_WAIT_USER = "WAIT_USER"
TASK_FAILED = "FAILED"
TASK_DUPLICATE_BLOCKED = "DUPLICATE_BLOCKED"
TASK_STOPPED = "STOPPED"
TASK_PAUSED = "PAUSED"
TASK_PENDING = "pending"  # batch created, not started

# Legacy aliases kept for compatibility with earlier MVP code/UI
TASK_WAITING_PO = TASK_WAITING
TASK_QUEUED = TASK_PENDING
TASK_RUNNING = TASK_DRAFT_EDITING
TASK_VERIFYING = TASK_PRE_SAVE_VERIFY
TASK_PO_CREATED = TASK_SUCCESS

AMOUNT_TOLERANCE = Decimal("0.01")

STEP_NAMES = (
    "READ_PR_DATA",
    "CREATE_TASK_SNAPSHOT",
    "OPEN_ERP_FORM",
    "CREATE_DRAFT",
    "FILL_HEADER",
    "FILL_LINES",
    "PRE_SAVE_VERIFY",
    "SAVE_PO",
    "READ_BACK_PO_NO",
    "UPSTREAM_WRITEBACK",
)

DEFAULT_ERP_CONFIG = {
    "purchasing_org": "1000",
    "purchasing_group": "P01",
    "currency_code": "CNY",
    "payment_terms": "NET30",
    "buyer_id": "BUYER-01",
    "tax_rate": Decimal("0.13"),
}


def _snapshot_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_task_status(status: str | None) -> str:
    value = (status or "").strip()
    mapping = {
        "WAITING_PO": TASK_WAITING,
        "QUEUED": TASK_PENDING,
        "pending": TASK_PENDING,
        "RUNNING": TASK_DRAFT_EDITING,
        "running": TASK_DRAFT_EDITING,
        "VERIFYING": TASK_PRE_SAVE_VERIFY,
        "PO_CREATED": TASK_SUCCESS,
        "success": TASK_SUCCESS,
        "failed": TASK_FAILED,
        "wait_user": TASK_WAIT_USER,
        "stopped": TASK_STOPPED,
        "paused": TASK_PAUSED,
        "duplicate_blocked": TASK_DUPLICATE_BLOCKED,
    }
    return mapping.get(value, value.upper() if value.isupper() or value in mapping else value.upper() if value else TASK_WAITING)


def _dec(value: Any, default: str = "0") -> Decimal:
    try:
        return money(Decimal(str(value if value is not None else default)))
    except Exception:  # noqa: BLE001
        return money(Decimal(default))


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _parse_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _candidate_status(request: ProcurementRequest, task: AgentTask | None) -> str:
    if request.po_no:
        return TASK_SUCCESS
    if task is not None and task.status:
        return _normalize_task_status(task.status)
    sync = (request.erp_sync_status or "").upper()
    if sync in {TASK_WAITING, "WAITING_PO", TASK_WAITING_PO}:
        return TASK_WAITING
    if sync == "FAILED":
        return TASK_FAILED
    return TASK_WAITING


def _serialize_candidate(request: ProcurementRequest, task: AgentTask | None) -> dict[str, Any]:
    status = _candidate_status(request, task)
    # UI may still filter WAITING_PO
    display_status = "WAITING_PO" if status == TASK_WAITING else status
    return {
        "pr_no": request.request_no,
        "oa_apply_no": request.oa_apply_no,
        "title": request.oa_title,
        "department": request.oa_department,
        "supplier_code": request.supplier_code,
        "supplier_name": request.supplier_name,
        "line_count": len(request.lines or []),
        "total_amount": request.final_total_amount_tax or request.total_amount,
        "purchase_method": request.purchase_method_confirmed or request.purchase_type,
        "award_confirmed_at": request.award_confirmed_at,
        "status": display_status,
        "candidate_status": display_status,
        "task_id": task.task_id if task else None,
        "batch_id": task.batch_id if task else None,
        "po_no": request.po_no or (task.po_no if task else None),
        "error_code": task.error_code if task else None,
        "erp_sync_status": request.erp_sync_status,
        "source_snapshot_hash": task.source_snapshot_hash if task else None,
    }


def _latest_po_task(session: Session, pr_no: str) -> AgentTask | None:
    return session.scalar(
        select(AgentTask)
        .where(
            AgentTask.operation == PO_OPERATION,
            AgentTask.business_key == pr_no,
        )
        .order_by(AgentTask.id.desc())
    )


def list_po_candidates(
    session: Session,
    *,
    status: str | None = None,
    q: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[dict[str, Any]], int]:
    statement = (
        select(ProcurementRequest)
        .options(selectinload(ProcurementRequest.lines))
        .where(
            or_(
                ProcurementRequest.erp_sync_status == "WAITING_PO",
                ProcurementRequest.po_no.is_not(None),
                ProcurementRequest.erp_sync_status.in_(
                    ["FAILED", "SUCCESS", "SENDING", "QUEUED", "RUNNING"]
                ),
            )
        )
        .order_by(ProcurementRequest.updated_at.desc(), ProcurementRequest.id.desc())
    )
    rows = list(session.scalars(statement).all())
    items: list[dict[str, Any]] = []
    for request in rows:
        oa = request.oa_application
        awarded = (
            normalize_procurement_status(oa.procurement_status if oa else None)
            == PROCUREMENT_STATUS_AWARDED
            or (request.erp_sync_status or "") == "WAITING_PO"
            or bool(request.po_no)
        )
        if not awarded:
            continue
        task = _latest_po_task(session, request.request_no)
        item = _serialize_candidate(request, task)
        if status:
            want = status.upper()
            got = str(item["status"]).upper()
            waiting_alias = {"WAITING", "WAITING_PO"}
            success_alias = {"SUCCESS", "PO_CREATED"}
            if want in waiting_alias:
                if got not in waiting_alias:
                    continue
            elif want in success_alias:
                if got not in success_alias:
                    continue
            elif got != want:
                continue
        if q:
            term = q.lower()
            blob = " ".join(
                str(item.get(key) or "")
                for key in ("pr_no", "oa_apply_no", "title", "department", "supplier_name", "po_no")
            ).lower()
            if term not in blob:
                continue
        items.append(item)
    total = len(items)
    start = max(page - 1, 0) * page_size
    return items[start : start + page_size], total


def _build_create_steps(pr_no: str, form: dict[str, Any]) -> list[dict[str, Any]]:
    header = form.get("header") or {}
    lines = form.get("lines") or []
    fields = [
        {"testid": "erp-po-supplier", "value": header.get("supplier_name") or ""},
        {"testid": "erp-po-purchasing-org", "value": header.get("purchasing_org") or ""},
        {"testid": "erp-po-purchasing-group", "value": header.get("purchasing_group") or ""},
        {"testid": "erp-po-currency", "value": header.get("currency_code") or "CNY"},
        {"testid": "erp-po-payment-terms", "value": header.get("payment_terms") or ""},
        {"testid": "erp-po-request-dept", "value": header.get("request_dept") or ""},
        {"testid": "erp-po-buyer", "value": header.get("buyer_id") or ""},
    ]
    line_fields: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        line_fields.extend(
            [
                {
                    "testid": f"erp-po-line-material-{index}",
                    "value": line.get("material_code") or "",
                },
                {
                    "testid": f"erp-po-line-qty-{index}",
                    "value": str(line.get("quantity") or ""),
                    "input_type": "number",
                },
                {
                    "testid": f"erp-po-line-price-{index}",
                    "value": str(line.get("unit_price_tax") or line.get("unit_price") or ""),
                    "input_type": "number",
                },
            ]
        )
    return [
        {
            "step_id": "READ_PR_DATA",
            "title": "读取采购云定标数据",
            "action": {"type": "noop", "pr_no": pr_no},
            "verify": {"type": "always"},
            "expected": f"PR {pr_no} awarded without po_no",
            "status": "pending",
            "retry_count": 0,
        },
        {
            "step_id": "OPEN_ERP_FORM",
            "title": "打开 ERP 建单页",
            "action": {"type": "navigate", "path": None},
            "verify": {"type": "testid_visible", "testid": "erp-po-create-form"},
            "expected": "ERP create form visible",
            "status": "pending",
            "retry_count": 0,
        },
        {
            "step_id": "FILL_HEADER",
            "title": "填写 PO Header",
            "action": {"type": "fill_fields", "fields": fields},
            "verify": {
                "type": "fields_equals",
                "fields": [
                    {"testid": "erp-po-supplier", "value": header.get("supplier_name") or ""},
                    {
                        "testid": "erp-po-purchasing-org",
                        "value": header.get("purchasing_org") or "",
                    },
                ],
            },
            "expected": "header fields match",
            "status": "pending",
            "retry_count": 0,
        },
        {
            "step_id": "FILL_LINES",
            "title": "填写物资行",
            "action": {"type": "fill_fields", "fields": line_fields},
            "verify": {"type": "line_count_at_least", "count": len(lines), "testid": "erp-po-line-row"},
            "expected": f"line count = {len(lines)}",
            "status": "pending",
            "retry_count": 0,
        },
        {
            "step_id": "SAVE_PO",
            "title": "创建 PO",
            "action": {"type": "click", "testid": "erp-po-create-button"},
            "verify": {"type": "testid_visible", "testid": "erp-po-created-po-no"},
            "expected": "PO number visible after save",
            "status": "pending",
            "retry_count": 0,
        },
        {
            "step_id": "READ_BACK_PO_NO",
            "title": "回读 PO 号",
            "action": {"type": "read_text", "testid": "erp-po-created-po-no"},
            "verify": {"type": "po_no_present", "testid": "erp-po-created-po-no"},
            "expected": "po_no non-empty",
            "status": "pending",
            "retry_count": 0,
        },
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def build_create_form_payload(request: ProcurementRequest) -> dict[str, Any]:
    tax_rate = DEFAULT_ERP_CONFIG["tax_rate"]
    lines = []
    for index, line in enumerate(request.lines or [], 1):
        unit_price_tax = money(line.unit_price)
        qty = money(line.quantity)
        line_amount_tax = money(qty * unit_price_tax)
        lines.append(
            {
                "po_item_no": index,
                "material_code": line.material_code,
                "material_name": line.material_name,
                "specification": line.specification or "",
                "quantity": qty,
                "uom": line.unit,
                "unit": line.unit,
                "unit_price": unit_price_tax,
                "unit_price_tax": unit_price_tax,
                "tax_rate": tax_rate,
                "line_amount": line_amount_tax,
                "line_amount_tax": line_amount_tax,
                "delivery_date": request.expected_delivery_date,
            }
        )
    header = {
        "supplier_code": request.supplier_code,
        "supplier_name": request.supplier_name,
        "request_dept": request.oa_department,
        "purchasing_org": DEFAULT_ERP_CONFIG["purchasing_org"],
        "purchasing_group": DEFAULT_ERP_CONFIG["purchasing_group"],
        "currency_code": DEFAULT_ERP_CONFIG["currency_code"],
        "payment_terms": DEFAULT_ERP_CONFIG["payment_terms"],
        "buyer_id": DEFAULT_ERP_CONFIG["buyer_id"],
        "total_amount_tax": money(
            sum((Decimal(str(item["line_amount_tax"])) for item in lines), Decimal("0"))
        ),
    }
    return _jsonable(
        {
            "pr_no": request.request_no,
            "oa_apply_no": request.oa_apply_no,
            "award_confirmed_at": request.award_confirmed_at,
            "purchase_method": request.purchase_method_confirmed or request.purchase_type,
            "header": header,
            "lines": lines,
        }
    )


def create_po_batch(
    session: Session,
    pr_nos: list[str],
    *,
    operator: str | None = None,
) -> dict[str, Any]:
    """Create batch + tasks only (no PO). Enters pending/PREPARING with Excel snapshot."""
    if not pr_nos:
        raise AppError(422, "EMPTY_SELECTION", "At least one PR is required")
    batch_id = _new_id("batch")
    batch = AgentBatch(
        batch_id=batch_id,
        status="open",
        total_count=0,
        operator=operator,
    )
    session.add(batch)
    tasks: list[dict[str, Any]] = []
    excel_tasks: list[dict[str, Any]] = []
    excel_lines: list[dict[str, Any]] = []
    for pr_no in pr_nos:
        request = session.scalar(
            select(ProcurementRequest)
            .where(ProcurementRequest.request_no == pr_no)
            .options(selectinload(ProcurementRequest.lines))
        )
        if request is None:
            raise AppError(404, "PR_NOT_FOUND", f"PR not found: {pr_no}")
        existing_order = session.scalar(
            select(ERPPurchaseOrder).where(ERPPurchaseOrder.pr_no == pr_no)
        )
        if request.po_no or existing_order is not None:
            po_no = request.po_no or (existing_order.po_no if existing_order else None)
            record_safety_event(
                session,
                event_type="DUPLICATE_BLOCKED",
                severity="WARN",
                pr_no=pr_no,
                po_no=po_no,
                batch_id=batch_id,
                stage="CREATE_BATCH",
                expected="no existing po_no",
                actual=f"po_no={po_no}",
                action_taken="skip",
            )
            tasks.append(
                {
                    "task_id": None,
                    "pr_no": pr_no,
                    "status": TASK_DUPLICATE_BLOCKED,
                    "po_no": po_no,
                    "error_code": "DUPLICATE_BLOCKED",
                    "route": None,
                }
            )
            continue
        form = build_create_form_payload(request)
        snap_hash = _snapshot_hash(form)
        task_id = _new_id("po-task")
        route = f"/erp/po-create/{task_id}"
        steps = _build_create_steps(pr_no, form)
        for step in steps:
            if step["step_id"] == "OPEN_ERP_FORM":
                step["action"]["path"] = route
        vlm = vlm_adapter.maybe_call(
            scenario="page_understand",
            payload={"route": route},
            rpa_ok=False,
        )
        task = AgentTask(
            task_id=task_id,
            business_key=pr_no,
            operation=PO_OPERATION,
            status=TASK_PENDING,
            batch_id=batch_id,
            retry_count=0,
            executor_type="rpa",
            executor_mode="rpa+vlm" if vlm.get("called") else "rpa",
            vlm_called=bool(vlm.get("called")),
            takeover_flag=False,
            source_snapshot_hash=snap_hash,
            draft_json=form,
            current_route=route,
            context_json={
                "pr_no": pr_no,
                "batch_id": batch_id,
                "route": route,
                "form": form,
                "source": form,
                "operator": operator,
                "steps": steps,
                "source_snapshot_hash": snap_hash,
                "vlm": vlm,
            },
            result={"messages": [], "steps": steps},
        )
        session.add(task)
        session.flush()
        for step in steps:
            session.add(
                AgentStepLog(
                    step_id=f"{task.task_id}-{step['step_id']}",
                    task_id=task.task_id,
                    batch_id=batch_id,
                    step_name=step["step_id"],
                    expected_json={"expected": step.get("expected")},
                    status="pending",
                    retry_count=0,
                    executor_type="rpa",
                )
            )
        request.erp_sync_status = "QUEUED"
        task_view = {
            "task_id": task.task_id,
            "pr_no": pr_no,
            "status": TASK_PENDING,
            "po_no": None,
            "error_code": None,
            "route": route,
            "source_snapshot_hash": snap_hash,
        }
        tasks.append(task_view)
        excel_tasks.append(
            {
                **task_view,
                "oa_apply_no": request.oa_apply_no,
                "supplier_code": request.supplier_code,
                "supplier_name": request.supplier_name,
                "total_amount": str(request.final_total_amount_tax or request.total_amount),
                "retry_count": 0,
                "created_at": utcnow().isoformat(),
            }
        )
        for line in form.get("lines") or []:
            excel_lines.append({"pr_no": pr_no, **line})

    excel_path = write_batch_excel(
        batch_id, operator=operator, tasks=excel_tasks, lines=excel_lines
    )
    batch.total_count = len([t for t in tasks if t.get("task_id")])
    batch.failed_count = len([t for t in tasks if t.get("status") == TASK_DUPLICATE_BLOCKED])
    batch.excel_snapshot_path = str(excel_path)
    session.commit()
    return {
        "batch_id": batch_id,
        "tasks": tasks,
        "excel_snapshot_path": str(excel_path),
        "status": "open",
    }


def get_po_task(session: Session, task_id: str) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    context = dict(task.context_json or {})
    result = dict(task.result or {})
    steps = result.get("steps") or context.get("steps") or []
    return {
        "task_id": task.task_id,
        "batch_id": task.batch_id,
        "pr_no": task.business_key,
        "operation": task.operation,
        "status": task.status,
        "po_no": task.po_no,
        "error_code": task.error_code,
        "retry_count": task.retry_count,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "ended_at": task.finished_at,
        "executor_type": task.executor_type,
        "executor_mode": getattr(task, "executor_mode", None) or "rpa",
        "vlm_called": bool(getattr(task, "vlm_called", False)),
        "takeover_flag": task.takeover_flag,
        "current_route": task.current_route,
        "source_snapshot_hash": getattr(task, "source_snapshot_hash", None),
        "writeback_status": getattr(task, "writeback_status", None),
        "form": context.get("form") or getattr(task, "draft_json", None),
        "draft": getattr(task, "draft_json", None) or context.get("form"),
        "steps": steps,
        "context_json": context,
        "result": result,
    }


def run_po_task(session: Session, task_id: str) -> dict[str, Any]:
    """Alias of start: enter PREPARING → DRAFT_EDITING (does not create PO)."""
    return start_po_task(session, task_id)


def start_po_task(session: Session, task_id: str) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    if _normalize_task_status(task.status) == TASK_SUCCESS and task.po_no:
        task.status = TASK_DUPLICATE_BLOCKED
        task.error_code = "DUPLICATE_BLOCKED"
        record_safety_event(
            session,
            event_type="DUPLICATE_BLOCKED",
            severity="WARN",
            task_id=task.task_id,
            batch_id=task.batch_id,
            pr_no=task.business_key,
            po_no=task.po_no,
            stage="START",
            expected="no po_no",
            actual=f"po_no={task.po_no}",
            action_taken="skip",
        )
        session.commit()
        payload = get_po_task(session, task_id)
        payload["error_code"] = "DUPLICATE_BLOCKED"
        return payload
    existing = session.scalar(
        select(ERPPurchaseOrder).where(ERPPurchaseOrder.pr_no == task.business_key)
    )
    if existing is not None:
        task.status = TASK_DUPLICATE_BLOCKED
        task.po_no = existing.po_no
        task.error_code = "DUPLICATE_BLOCKED"
        task.finished_at = utcnow()
        record_safety_event(
            session,
            event_type="DUPLICATE_BLOCKED",
            severity="WARN",
            task_id=task.task_id,
            batch_id=task.batch_id,
            pr_no=task.business_key,
            po_no=existing.po_no,
            stage="START",
            expected="no existing PO",
            actual=f"po_no={existing.po_no}",
            action_taken="skip",
        )
        session.commit()
        payload = get_po_task(session, task_id)
        payload["error_code"] = "DUPLICATE_BLOCKED"
        return payload
    if _normalize_task_status(task.status) == TASK_STOPPED:
        raise AppError(409, "TASK_STOPPED", "Task already stopped")
    task.status = TASK_PREPARING
    task.started_at = task.started_at or utcnow()
    task.error_code = None
    record_step_progress(
        session,
        task.task_id,
        step_name="CREATE_DRAFT",
        status="success",
        expected={"action": "enter_draft"},
        actual={"route": task.current_route},
    )
    task.status = TASK_DRAFT_EDITING
    request = session.scalar(
        select(ProcurementRequest).where(ProcurementRequest.request_no == task.business_key)
    )
    if request is not None:
        request.erp_sync_status = "DRAFT_EDITING"
    route = task.current_route or f"/erp/po-create/{task.task_id}"
    context = dict(task.context_json or {})
    context["route"] = route
    if not task.draft_json:
        task.draft_json = context.get("form")
    task.context_json = context
    task.current_route = route
    session.commit()
    return get_po_task(session, task_id)


def record_step_progress(
    session: Session,
    task_id: str,
    *,
    step_name: str,
    status: str,
    expected: dict[str, Any] | None = None,
    actual: dict[str, Any] | None = None,
    error_code: str | None = None,
    duration_ms: int | None = None,
    retry_count: int = 0,
) -> AgentStepLog:
    step_id = f"{task_id}-{step_name}-{uuid.uuid4().hex[:8]}"
    log = AgentStepLog(
        step_id=step_id,
        task_id=task_id,
        step_name=step_name,
        expected_json=expected,
        actual_json=actual,
        status=status,
        retry_count=retry_count,
        duration_ms=duration_ms,
        error_code=error_code,
    )
    session.add(log)
    return log


def record_safety_event(
    session: Session,
    *,
    event_type: str,
    severity: str = "INFO",
    task_id: str | None = None,
    batch_id: str | None = None,
    pr_no: str | None = None,
    po_no: str | None = None,
    stage: str | None = None,
    expected: str | None = None,
    actual: str | None = None,
    action_taken: str | None = None,
    retry_count: int = 0,
) -> AgentSafetyLog:
    event = AgentSafetyLog(
        event_id=_new_id("safe"),
        task_id=task_id,
        batch_id=batch_id,
        pr_no=pr_no,
        po_no=po_no,
        stage=stage,
        event_type=event_type,
        severity=severity,
        expected=expected,
        actual=actual,
        action_taken=action_taken,
        retry_count=retry_count,
    )
    session.add(event)
    return event


def create_po_from_erp_form(
    session: Session,
    task_id: str,
    *,
    header: dict[str, Any],
    lines: list[dict[str, Any]],
    simulate_readback_fail: bool = False,
) -> dict[str, Any]:
    """ERP 本系统建单保存（非采购云→ERP 业务 API）。"""
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    pr_no = task.business_key
    request = session.scalar(
        select(ProcurementRequest)
        .where(ProcurementRequest.request_no == pr_no)
        .options(selectinload(ProcurementRequest.lines), selectinload(ProcurementRequest.oa_application))
    )
    if request is None:
        raise AppError(404, "PR_NOT_FOUND", "PR not found")

    existing = session.scalar(select(ERPPurchaseOrder).where(ERPPurchaseOrder.pr_no == pr_no))
    if existing is not None or request.po_no:
        po_no = request.po_no or existing.po_no
        task.status = TASK_DUPLICATE_BLOCKED
        task.po_no = po_no
        task.error_code = "DUPLICATE_BLOCKED"
        task.finished_at = utcnow()
        record_safety_event(
            session,
            event_type="DUPLICATE_BLOCKED",
            severity="WARN",
            task_id=task.task_id,
            batch_id=task.batch_id,
            pr_no=pr_no,
            po_no=po_no,
            stage="SAVE_PO",
            expected="create once",
            actual=f"existing {po_no}",
            action_taken="block",
        )
        session.commit()
        raise AppError(409, "DUPLICATE_BLOCKED", f"PO already exists: {po_no}", {"po_no": po_no})

    if not lines:
        raise AppError(422, "EMPTY_LINES", "At least one PO line is required")

    # Enforce pre-save verify before irreversible create
    verify = pre_save_verify(session, task_id, header=header, lines=lines, commit=False)
    if not verify.get("passed"):
        task.status = TASK_FAILED
        task.error_code = "VALIDATION_FAILED"
        record_safety_event(
            session,
            event_type="VALIDATION_FAILED",
            severity="BLOCKER",
            task_id=task.task_id,
            batch_id=task.batch_id,
            pr_no=pr_no,
            stage="PRE_SAVE_VERIFY",
            expected="all checks pass",
            actual=json.dumps(verify.get("errors") or [], ensure_ascii=False),
            action_taken="block_save",
        )
        session.commit()
        raise AppError(422, "VALIDATION_FAILED", "Pre-save validation failed", verify)

    task.status = TASK_SAVING
    task.started_at = task.started_at or utcnow()
    record_step_progress(
        session,
        task.task_id,
        step_name="SAVE_PO",
        status="running",
        expected={"action": "create"},
    )

    computed_lines: list[dict[str, Any]] = []
    total = Decimal("0")
    for index, raw in enumerate(lines, 1):
        qty = _dec(raw.get("quantity"))
        unit_price_tax = _dec(raw.get("unit_price_tax", raw.get("unit_price")))
        tax_rate = _dec(raw.get("tax_rate"), str(DEFAULT_ERP_CONFIG["tax_rate"]))
        line_amount_tax = money(qty * unit_price_tax)
        total += line_amount_tax
        computed_lines.append(
            {
                "line_no": index,
                "po_item_no": int(raw.get("po_item_no") or index),
                "material_code": str(raw.get("material_code") or ""),
                "material_name": str(raw.get("material_name") or ""),
                "specification": str(raw.get("specification") or ""),
                "unit": str(raw.get("uom") or raw.get("unit") or "EA"),
                "uom": str(raw.get("uom") or raw.get("unit") or "EA"),
                "quantity": qty,
                "unit_price": unit_price_tax,
                "unit_price_tax": unit_price_tax,
                "tax_rate": tax_rate,
                "line_amount": line_amount_tax,
                "line_amount_tax": line_amount_tax,
                "delivery_date": _parse_date(raw.get("delivery_date")),
            }
        )
        if not computed_lines[-1]["material_code"]:
            raise AppError(422, "INVALID_LINE", f"Line {index} missing material_code")

    if simulate_readback_fail:
        task.status = TASK_FAILED
        task.error_code = "PO_READBACK_FAIL"
        task.finished_at = utcnow()
        request.erp_sync_status = "FAILED"
        record_step_progress(
            session,
            task.task_id,
            step_name="READ_BACK_PO_NO",
            status="failed",
            error_code="PO_READBACK_FAIL",
            actual={"po_no": None},
        )
        record_safety_event(
            session,
            event_type="PO_READBACK_FAIL",
            severity="BLOCKER",
            task_id=task.task_id,
            batch_id=task.batch_id,
            pr_no=pr_no,
            stage="READ_BACK_PO_NO",
            expected="po_no non-empty",
            actual="empty",
            action_taken="fail",
        )
        session.commit()
        raise AppError(500, "PO_READBACK_FAIL", "PO saved but po_no readback failed")

    po_no = _unique_number("PO-2026-")
    # Phase2: optional OCR/readback assist (does not replace structured po_no).
    vlm_read = vlm_adapter.maybe_call(
        scenario="po_readback_ocr",
        payload={"po_no": po_no, "task_id": task.task_id},
        rpa_ok=True,
    )
    if vlm_read.get("called"):
        task.vlm_called = True

    order = ERPPurchaseOrder(
        po_no=po_no,
        pr_no=pr_no,
        oa_apply_no=request.oa_apply_no,
        submission_version=request.submission_version or 1,
        task_id=task.task_id,
        status="created",
        total_amount=money(total),
        supplier_code=header.get("supplier_code") or request.supplier_code,
        supplier_name=header.get("supplier_name") or request.supplier_name,
        request_dept=header.get("request_dept") or request.oa_department,
        purchasing_org=header.get("purchasing_org") or DEFAULT_ERP_CONFIG["purchasing_org"],
        purchasing_group=header.get("purchasing_group")
        or DEFAULT_ERP_CONFIG["purchasing_group"],
        currency_code=header.get("currency_code") or "CNY",
        payment_terms=header.get("payment_terms") or DEFAULT_ERP_CONFIG["payment_terms"],
        buyer_id=header.get("buyer_id") or DEFAULT_ERP_CONFIG["buyer_id"],
        total_amount_tax=money(total),
        created_by_agent_task_id=task.task_id,
        batch_id=task.batch_id,
        purchase_method=(
            (task.context_json or {}).get("form", {}).get("purchase_method")
            or request.purchase_method_confirmed
            or request.purchase_type
        ),
    )
    for item in computed_lines:
        order.lines.append(
            ERPPurchaseOrderLine(
                line_no=item["line_no"],
                po_item_no=item["po_item_no"],
                material_code=item["material_code"],
                material_name=item["material_name"],
                specification=item["specification"],
                unit=item["unit"],
                uom=item["uom"],
                quantity=item["quantity"],
                unit_price=item["unit_price"],
                unit_price_tax=item["unit_price_tax"],
                tax_rate=item["tax_rate"],
                line_amount=item["line_amount"],
                line_amount_tax=item["line_amount_tax"],
                delivery_date=item["delivery_date"],
            )
        )
    session.add(order)
    session.flush()

    if not order.po_no:
        task.status = TASK_FAILED
        task.error_code = "PO_READBACK_FAIL"
        task.finished_at = utcnow()
        request.erp_sync_status = "FAILED"
        record_safety_event(
            session,
            event_type="PO_READBACK_FAIL",
            severity="BLOCKER",
            task_id=task.task_id,
            batch_id=task.batch_id,
            pr_no=pr_no,
            stage="READ_BACK_PO_NO",
            expected="po_no non-empty",
            actual="empty",
            action_taken="fail",
        )
        session.commit()
        raise AppError(500, "PO_READBACK_FAIL", "PO created without readable po_no")

    task.status = TASK_READBACK
    record_step_progress(
        session,
        task.task_id,
        step_name="READ_BACK_PO_NO",
        status="running",
        expected={"po_no": "non-empty"},
    )

    request.po_no = order.po_no
    request.erp_status = "success"
    request.erp_sync_status = "SUCCESS"
    request.status = "submitted"
    request.submitted_at = utcnow()
    oa = request.oa_application
    if oa is not None:
        oa.linked_po_no = order.po_no
        oa.erp_status = "success"
        set_procurement_status(oa, PROCUREMENT_STATUS_AWARDED)

    task.po_no = order.po_no
    task.error_code = None
    result = dict(task.result or {})
    steps = list(result.get("steps") or [])
    gui_driven = bool((task.context_json or {}).get("gui_driven"))
    # GUI Agent still needs READ_BACK_PO_NO; keep loop alive until DOM readback reports.
    done_ids = {
        "SAVE_PO",
        "FILL_HEADER",
        "FILL_LINES",
        "OPEN_ERP_FORM",
        "READ_PR_DATA",
        "PRE_SAVE_VERIFY",
        "CREATE_DRAFT",
        "UPSTREAM_WRITEBACK",
    }
    for step in steps:
        sid = step.get("step_id")
        if sid == "READ_BACK_PO_NO" and gui_driven:
            step["status"] = "pending"
            step["actual"] = {"po_no": order.po_no}
            continue
        if sid in done_ids or (sid == "READ_BACK_PO_NO" and not gui_driven):
            step["status"] = "passed" if gui_driven else "success"
            if sid == "READ_BACK_PO_NO":
                step["actual"] = {"po_no": order.po_no}
            elif sid == "SAVE_PO":
                step["actual"] = {"po_no": order.po_no}
    result["steps"] = steps
    result["po_no"] = order.po_no
    task.result = result
    if gui_driven:
        task.status = "running"
        task.finished_at = None
    else:
        task.status = TASK_SUCCESS
        task.finished_at = utcnow()

    record_step_progress(
        session,
        task.task_id,
        step_name="SAVE_PO",
        status="success",
        actual={"po_no": order.po_no},
    )
    record_step_progress(
        session,
        task.task_id,
        step_name="READ_BACK_PO_NO",
        status="success",
        actual={"po_no": order.po_no},
    )
    _upsert_lineage(
        session,
        request.oa_apply_no or "",
        pr_no=pr_no,
        po_no=order.po_no,
        task_id=task.task_id,
        status="po_created",
    )
    # Upstream writeback (procurement cloud API). Failure must not recreate PO.
    writeback = link_po_to_procurement(
        session,
        pr_no,
        po_no=order.po_no,
        task_id=task.task_id,
        commit=False,
        simulate_failure=False,
    )
    task.writeback_status = writeback.get("status")
    _refresh_batch_excel(session, task.batch_id)
    session.commit()
    return {
        "task_id": task.task_id,
        "batch_id": task.batch_id,
        "pr_no": pr_no,
        "po_no": order.po_no,
        "status": TASK_SUCCESS,
        "writeback_status": task.writeback_status,
        "order": order,
    }


def mark_po_created(
    session: Session,
    task_id: str,
    *,
    po_no: str | None,
    success: bool = True,
    error_code: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    request = session.scalar(
        select(ProcurementRequest).where(ProcurementRequest.request_no == task.business_key)
    )
    if success:
        if not po_no:
            task.status = TASK_FAILED
            task.error_code = "PO_READBACK_FAIL"
            task.finished_at = utcnow()
            if request is not None:
                request.erp_sync_status = "FAILED"
            record_safety_event(
                session,
                event_type="PO_READBACK_FAIL",
                severity="BLOCKER",
                task_id=task.task_id,
                batch_id=task.batch_id,
                pr_no=task.business_key,
                stage="MARK_CREATED",
                expected="po_no non-empty",
                actual="empty",
                action_taken="fail",
            )
            session.commit()
            raise AppError(422, "PO_READBACK_FAIL", "Cannot mark success without po_no")
        order = session.scalar(select(ERPPurchaseOrder).where(ERPPurchaseOrder.po_no == po_no))
        if order is None:
            raise AppError(404, "PO_NOT_FOUND", f"PO not found: {po_no}")
        task.status = TASK_PO_CREATED
        task.po_no = po_no
        task.error_code = None
        task.finished_at = utcnow()
        if request is not None:
            request.po_no = po_no
            request.erp_sync_status = "SUCCESS"
            request.erp_status = "success"
        session.commit()
        return get_po_task(session, task_id)

    task.status = TASK_FAILED if error_code != "WAIT_USER" else TASK_WAIT_USER
    task.error_code = error_code or "FAILED"
    task.takeover_flag = task.status == TASK_WAIT_USER
    task.finished_at = utcnow() if task.status == TASK_FAILED else task.finished_at
    if request is not None and task.status == TASK_FAILED:
        request.erp_sync_status = "FAILED"
    record_safety_event(
        session,
        event_type=error_code or "FAILED",
        severity="WARN" if task.status == TASK_WAIT_USER else "BLOCKER",
        task_id=task.task_id,
        batch_id=task.batch_id,
        pr_no=task.business_key,
        stage="MARK_CREATED",
        expected="PO_CREATED",
        actual=message or error_code,
        action_taken="wait_user" if task.takeover_flag else "fail",
    )
    session.commit()
    return get_po_task(session, task_id)


def stop_batch(session: Session, batch_id: str) -> dict[str, Any]:
    tasks = list(
        session.scalars(
            select(AgentTask).where(
                AgentTask.batch_id == batch_id,
                AgentTask.operation == PO_OPERATION,
            )
        ).all()
    )
    stopped = 0
    for task in tasks:
        if task.status in {TASK_QUEUED, TASK_RUNNING, TASK_WAIT_USER}:
            task.status = TASK_STOPPED
            task.finished_at = utcnow()
            stopped += 1
            record_safety_event(
                session,
                event_type="EMERGENCY_STOP",
                severity="WARN",
                task_id=task.task_id,
                batch_id=batch_id,
                pr_no=task.business_key,
                stage="STOP",
                action_taken="stop",
            )
    session.commit()
    return {"batch_id": batch_id, "stopped": stopped, "total": len(tasks)}


def get_po_detail(session: Session, po_no: str) -> dict[str, Any]:
    order = session.scalar(
        select(ERPPurchaseOrder)
        .where(ERPPurchaseOrder.po_no == po_no)
        .options(selectinload(ERPPurchaseOrder.lines))
    )
    if order is None:
        raise AppError(404, "PO_NOT_FOUND", "Purchase order not found")
    task = None
    if order.created_by_agent_task_id:
        task = session.scalar(
            select(AgentTask).where(AgentTask.task_id == order.created_by_agent_task_id)
        )
    if task is None:
        task = session.scalar(
            select(AgentTask)
            .where(AgentTask.po_no == po_no, AgentTask.operation == PO_OPERATION)
            .order_by(AgentTask.id.desc())
        )
    duration_ms = None
    if task and task.started_at and task.finished_at:
        duration_ms = int((task.finished_at - task.started_at).total_seconds() * 1000)
    return {
        "po_no": order.po_no,
        "pr_no": order.pr_no,
        "oa_apply_no": order.oa_apply_no,
        "status": order.status,
        "supplier_code": order.supplier_code,
        "supplier_name": order.supplier_name,
        "request_dept": order.request_dept,
        "purchasing_org": order.purchasing_org,
        "purchasing_group": order.purchasing_group,
        "currency_code": order.currency_code,
        "payment_terms": order.payment_terms,
        "buyer_id": order.buyer_id,
        "total_amount": order.total_amount,
        "total_amount_tax": order.total_amount_tax or order.total_amount,
        "batch_id": order.batch_id,
        "task_id": order.created_by_agent_task_id or order.task_id,
        "created_at": order.created_at,
        "updated_at": order.updated_at,
        "lines": [
            {
                "id": line.id,
                "line_no": line.line_no,
                "po_item_no": line.po_item_no or line.line_no,
                "material_code": line.material_code,
                "material_name": line.material_name,
                "specification": line.specification,
                "quantity": line.quantity,
                "unit": line.unit,
                "uom": line.uom or line.unit,
                "unit_price": line.unit_price,
                "unit_price_tax": line.unit_price_tax or line.unit_price,
                "tax_rate": line.tax_rate,
                "line_amount": line.line_amount,
                "line_amount_tax": line.line_amount_tax or line.line_amount,
                "delivery_date": line.delivery_date,
            }
            for line in order.lines
        ],
        "agent_summary": {
            "status": task.status if task else None,
            "retry_count": task.retry_count if task else 0,
            "takeover_flag": bool(task.takeover_flag) if task else False,
            "duration_ms": duration_ms,
            "error_code": task.error_code if task else None,
            "executor_type": task.executor_type if task else None,
        },
    }


def get_po_lineage(session: Session, po_no: str) -> dict[str, Any]:
    order = session.scalar(select(ERPPurchaseOrder).where(ERPPurchaseOrder.po_no == po_no))
    if order is None:
        raise AppError(404, "PO_NOT_FOUND", "Purchase order not found")
    lineage = None
    if order.oa_apply_no:
        lineage = session.scalar(
            select(BusinessLineage).where(BusinessLineage.oa_apply_no == order.oa_apply_no)
        )
    transfers = []
    statement = select(IntegrationTransfer).order_by(IntegrationTransfer.id.desc())
    for transfer in session.scalars(statement).all():
        keys = {transfer.source_key, transfer.target_key, transfer.task_id}
        if order.oa_apply_no in keys or order.pr_no in keys or order.po_no in keys:
            transfers.append(transfer)
        elif order.created_by_agent_task_id and transfer.task_id == order.created_by_agent_task_id:
            transfers.append(transfer)
    task_ids = list(
        dict.fromkeys(
            [order.created_by_agent_task_id or order.task_id]
            + [item.task_id for item in transfers]
            + ([lineage.task_id] if lineage and lineage.task_id else [])
        )
    )
    task_ids = [item for item in task_ids if item]
    return {
        "oa_apply_no": order.oa_apply_no,
        "pr_no": order.pr_no,
        "po_no": order.po_no,
        "batch_id": order.batch_id,
        "task_id": order.created_by_agent_task_id or order.task_id,
        "task_ids": task_ids,
        "latest_status": lineage.latest_status if lineage else "po_created",
        "transfers": transfers,
    }


def _filter_tasks(
    session: Session,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    department: str | None = None,
    batch_id: str | None = None,
    status: str | None = None,
) -> list[AgentTask]:
    statement = select(AgentTask).where(AgentTask.operation == PO_OPERATION)
    if batch_id:
        statement = statement.where(AgentTask.batch_id == batch_id)
    if status:
        statement = statement.where(AgentTask.status == status)
    tasks = list(session.scalars(statement.order_by(AgentTask.id.desc())).all())
    filtered: list[AgentTask] = []
    for task in tasks:
        created = task.created_at
        if date_from and created and created < date_from:
            continue
        if date_to and created and created > date_to:
            continue
        if department:
            request = session.scalar(
                select(ProcurementRequest).where(ProcurementRequest.request_no == task.business_key)
            )
            if not request or (request.oa_department or "") != department:
                continue
        filtered.append(task)
    return filtered


def agent_dashboard_summary(
    session: Session,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    department: str | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    waiting_items, _ = list_po_candidates(session, status="WAITING_PO", page_size=10000)
    waiting_count = len(waiting_items)

    tasks = _filter_tasks(
        session,
        date_from=date_from,
        date_to=date_to,
        department=department,
        batch_id=batch_id,
    )
    created = [
        t
        for t in tasks
        if _normalize_task_status(t.status) == TASK_SUCCESS and t.po_no
    ]
    finished = [
        t
        for t in tasks
        if _normalize_task_status(t.status)
        in {TASK_SUCCESS, TASK_FAILED, TASK_STOPPED, TASK_DUPLICATE_BLOCKED}
    ]
    success_rate = (len(created) / len(finished)) if finished else 0.0
    first_pass = [t for t in created if (t.retry_count or 0) == 0]
    first_pass_rate = (len(first_pass) / len(created)) if created else 0.0
    durations = []
    for task in created:
        if task.started_at and task.finished_at:
            durations.append((task.finished_at - task.started_at).total_seconds())
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0
    step_retries = session.scalar(
        select(func.coalesce(func.sum(AgentStepLog.retry_count), 0)).where(
            AgentStepLog.task_id.in_([t.task_id for t in tasks] or ["__none__"])
        )
    )
    takeover = len(
        [
            t
            for t in tasks
            if _normalize_task_status(t.status) == TASK_WAIT_USER or t.takeover_flag
        ]
    )
    duplicate = session.scalar(
        select(func.count())
        .select_from(AgentSafetyLog)
        .where(AgentSafetyLog.event_type == "DUPLICATE_BLOCKED")
    )
    emergency = session.scalar(
        select(func.count())
        .select_from(AgentSafetyLog)
        .where(
            or_(
                AgentSafetyLog.event_type == "EMERGENCY_STOP",
                AgentSafetyLog.event_type == "STOP",
                AgentSafetyLog.event_type == "MANUAL_STOP",
            )
        )
    )
    readback_fail = session.scalar(
        select(func.count())
        .select_from(AgentSafetyLog)
        .where(
            or_(
                AgentSafetyLog.event_type == "PO_READBACK_FAIL",
                AgentSafetyLog.event_type == "PO_READBACK_FAILED",
            )
        )
    )
    vlm_called_count = len([t for t in tasks if getattr(t, "vlm_called", False)])
    executed = [t for t in tasks if _normalize_task_status(t.status) != TASK_PENDING]
    return {
        "waiting_pr_count": waiting_count,
        "task_count": len(tasks),
        "agent_created_po_count": len(created),
        "success_rate": round(success_rate, 4),
        "first_pass_rate": round(first_pass_rate, 4),
        "avg_duration_seconds": round(avg_duration, 2),
        "retry_count_total": int(step_retries or 0),
        "takeover_count": takeover,
        "duplicate_blocked_count": int(duplicate or 0),
        "emergency_stop_count": int(emergency or 0),
        "po_readback_fail_count": int(readback_fail or 0),
        "vlm_called_count": vlm_called_count,
        "vlm_call_rate": round((vlm_called_count / len(executed)), 4) if executed else 0.0,
    }


def agent_dashboard_funnel(
    session: Session,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    tasks = _filter_tasks(session, date_from=date_from, date_to=date_to, batch_id=batch_id)
    task_ids = [t.task_id for t in tasks]
    queued_like = [
        t
        for t in tasks
        if t.status
        in {TASK_QUEUED, TASK_RUNNING, TASK_PO_CREATED, TASK_FAILED, TASK_STOPPED, TASK_VERIFYING}
    ]

    def _step_success(name: str) -> int:
        if not task_ids:
            return 0
        return int(
            session.scalar(
                select(func.count(func.distinct(AgentStepLog.task_id))).where(
                    AgentStepLog.task_id.in_(task_ids),
                    AgentStepLog.step_name == name,
                    AgentStepLog.status == "success",
                )
            )
            or 0
        )

    return {
        "queued": len(queued_like),
        "read_pr_data": _step_success("READ_PR_DATA"),
        "task_snapshot": _step_success("CREATE_TASK_SNAPSHOT") or len(queued_like),
        "open_erp_form": _step_success("OPEN_ERP_FORM"),
        "draft_entered": _step_success("CREATE_DRAFT")
        or len([t for t in tasks if _normalize_task_status(t.status) in {
            TASK_DRAFT_EDITING, TASK_PRE_SAVE_VERIFY, TASK_SAVING, TASK_READBACK, TASK_SUCCESS
        }]),
        "fill_header": _step_success("FILL_HEADER"),
        "fill_lines": _step_success("FILL_LINES"),
        "pre_save_verify": _step_success("PRE_SAVE_VERIFY"),
        "save_po": _step_success("SAVE_PO"),
        "read_back_po_no": len(
            [t for t in tasks if t.po_no and _normalize_task_status(t.status) == TASK_SUCCESS]
        ),
        "upstream_writeback": _step_success("UPSTREAM_WRITEBACK"),
    }


def agent_dashboard_events(
    session: Session,
    *,
    event_type: str | None = None,
    severity: str | None = None,
    stage: str | None = None,
    task_id: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[AgentSafetyLog], int]:
    statement = select(AgentSafetyLog)
    if event_type:
        statement = statement.where(AgentSafetyLog.event_type == event_type)
    if severity:
        statement = statement.where(AgentSafetyLog.severity == severity)
    if stage:
        statement = statement.where(AgentSafetyLog.stage == stage)
    if task_id:
        statement = statement.where(AgentSafetyLog.task_id == task_id)
    rows = list(session.scalars(statement.order_by(AgentSafetyLog.id.desc())).all())
    total = len(rows)
    start = max(page - 1, 0) * page_size
    return rows[start : start + page_size], total


def agent_dashboard_tasks(
    session: Session,
    *,
    status: str | None = None,
    batch_id: str | None = None,
    pr_no: str | None = None,
    po_no: str | None = None,
) -> list[dict[str, Any]]:
    statement = select(AgentTask).where(AgentTask.operation == PO_OPERATION)
    if status:
        statement = statement.where(AgentTask.status == status)
    if batch_id:
        statement = statement.where(AgentTask.batch_id == batch_id)
    if pr_no:
        statement = statement.where(AgentTask.business_key == pr_no)
    if po_no:
        statement = statement.where(AgentTask.po_no == po_no)
    tasks = list(session.scalars(statement.order_by(AgentTask.id.desc())).all())
    return [get_po_task(session, task.task_id) for task in tasks]


def agent_task_steps(session: Session, task_id: str) -> list[dict[str, Any]]:
    logs = list(
        session.scalars(
            select(AgentStepLog)
            .where(AgentStepLog.task_id == task_id)
            .order_by(AgentStepLog.id.asc())
        ).all()
    )
    if logs:
        return [
            {
                "step_id": log.step_id,
                "task_id": log.task_id,
                "step_name": log.step_name,
                "expected_json": log.expected_json,
                "actual_json": log.actual_json,
                "status": log.status,
                "retry_count": log.retry_count,
                "duration_ms": log.duration_ms,
                "error_code": log.error_code,
                "created_at": log.created_at,
            }
            for log in logs
        ]
    task = get_po_task(session, task_id)
    return task.get("steps") or []


def retry_po_task(session: Session, task_id: str) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    if task.status == TASK_PO_CREATED and task.po_no:
        raise AppError(409, "DUPLICATE_BLOCKED", "PO already created")
    task.status = TASK_QUEUED
    task.error_code = None
    task.retry_count = (task.retry_count or 0) + 1
    task.finished_at = None
    task.takeover_flag = False
    session.commit()
    return run_po_task(session, task_id)


def _po_amount(order: ERPPurchaseOrder) -> Decimal:
    if order.lines:
        return money(
            sum(
                (
                    (line.line_amount_tax or line.line_amount or Decimal("0"))
                    for line in order.lines
                ),
                Decimal("0"),
            )
        )
    return money(order.total_amount_tax or order.total_amount or Decimal("0"))


def _filter_orders(
    session: Session,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    department: str | None = None,
    supplier: str | None = None,
) -> list[ERPPurchaseOrder]:
    statement = select(ERPPurchaseOrder).options(selectinload(ERPPurchaseOrder.lines))
    orders = list(session.scalars(statement.order_by(ERPPurchaseOrder.id.desc())).all())
    filtered: list[ERPPurchaseOrder] = []
    for order in orders:
        if date_from and order.created_at and order.created_at < date_from:
            continue
        if date_to and order.created_at and order.created_at > date_to:
            continue
        if department and (order.request_dept or "") != department:
            continue
        if supplier and supplier not in {
            order.supplier_code or "",
            order.supplier_name or "",
        }:
            continue
        filtered.append(order)
    return filtered


def po_dashboard_summary(
    session: Session,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    department: str | None = None,
    supplier: str | None = None,
) -> dict[str, Any]:
    orders = _filter_orders(
        session,
        date_from=date_from,
        date_to=date_to,
        department=department,
        supplier=supplier,
    )
    total_amount = money(sum((_po_amount(o) for o in orders), Decimal("0")))
    line_count = sum(len(o.lines or []) for o in orders)
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    def _aware(dt: datetime) -> datetime:
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    month_new = len(
        [o for o in orders if o.created_at and _aware(o.created_at) >= month_start]
    )
    depts = {o.request_dept for o in orders if o.request_dept}
    complete = len([o for o in orders if o.oa_apply_no and o.pr_no and o.po_no])
    return {
        "po_count": len(orders),
        "po_total_amount": total_amount,
        "month_new_po_count": month_new,
        "line_count": line_count,
        "avg_po_amount": money(total_amount / len(orders)) if orders else Decimal("0"),
        "department_count": len(depts),
        "avg_lines_per_po": round(line_count / len(orders), 2) if orders else 0,
        "lineage_complete_rate": round(complete / len(orders), 4) if orders else 0,
    }


def po_dashboard_trend(
    session: Session,
    *,
    grain: str = "day",
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict[str, Any]]:
    orders = _filter_orders(session, date_from=date_from, date_to=date_to)
    buckets: dict[str, dict[str, Any]] = {}
    for order in orders:
        if not order.created_at:
            continue
        dt = order.created_at
        if grain == "month":
            key = f"{dt.year:04d}-{dt.month:02d}"
        elif grain == "week":
            key = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
        else:
            key = dt.date().isoformat()
        bucket = buckets.setdefault(key, {"period": key, "po_count": 0, "amount": Decimal("0")})
        bucket["po_count"] += 1
        bucket["amount"] = money(bucket["amount"] + _po_amount(order))
    return sorted(buckets.values(), key=lambda item: item["period"])


def po_dashboard_by_department(session: Session, metric: str = "amount") -> list[dict[str, Any]]:
    orders = _filter_orders(session)
    groups: dict[str, dict[str, Any]] = {}
    for order in orders:
        key = order.request_dept or "未分配"
        item = groups.setdefault(key, {"department": key, "po_count": 0, "amount": Decimal("0")})
        item["po_count"] += 1
        item["amount"] = money(item["amount"] + _po_amount(order))
    rows = list(groups.values())
    rows.sort(key=lambda item: item["amount" if metric == "amount" else "po_count"], reverse=True)
    return rows


def po_dashboard_by_supplier(
    session: Session, *, limit: int = 10, metric: str = "amount"
) -> list[dict[str, Any]]:
    orders = _filter_orders(session)
    groups: dict[str, dict[str, Any]] = {}
    for order in orders:
        key = order.supplier_name or order.supplier_code or "未知供应商"
        item = groups.setdefault(
            key,
            {
                "supplier": key,
                "supplier_code": order.supplier_code,
                "po_count": 0,
                "amount": Decimal("0"),
            },
        )
        item["po_count"] += 1
        item["amount"] = money(item["amount"] + _po_amount(order))
    rows = list(groups.values())
    rows.sort(key=lambda item: item["amount" if metric == "amount" else "po_count"], reverse=True)
    return rows[:limit]


def po_dashboard_by_material(
    session: Session, *, limit: int = 10, metric: str = "amount"
) -> list[dict[str, Any]]:
    orders = _filter_orders(session)
    groups: dict[str, dict[str, Any]] = {}
    for order in orders:
        for line in order.lines or []:
            key = line.material_code
            item = groups.setdefault(
                key,
                {
                    "material_code": line.material_code,
                    "material_name": line.material_name,
                    "quantity": Decimal("0"),
                    "amount": Decimal("0"),
                },
            )
            item["quantity"] = money(item["quantity"] + (line.quantity or Decimal("0")))
            item["amount"] = money(
                item["amount"] + (line.line_amount_tax or line.line_amount or Decimal("0"))
            )
    rows = list(groups.values())
    key = "amount" if metric == "amount" else "quantity"
    rows.sort(key=lambda item: item[key], reverse=True)
    return rows[:limit]


def recent_pos(
    session: Session, *, page: int = 1, page_size: int = 20, status: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    statement = select(ERPPurchaseOrder).options(selectinload(ERPPurchaseOrder.lines))
    if status:
        statement = statement.where(ERPPurchaseOrder.status == status)
    orders = list(session.scalars(statement.order_by(ERPPurchaseOrder.id.desc())).all())
    total = len(orders)
    start = max(page - 1, 0) * page_size
    slice_rows = orders[start : start + page_size]
    return [
        {
            "po_no": order.po_no,
            "pr_no": order.pr_no,
            "oa_apply_no": order.oa_apply_no,
            "supplier_name": order.supplier_name,
            "request_dept": order.request_dept,
            "total_amount": _po_amount(order),
            "status": order.status,
            "created_at": order.created_at,
        }
        for order in slice_rows
    ], total


def get_create_context(session: Session, reference: str) -> dict[str, Any]:
    """Load create page by task_id or pr_no."""
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == reference))
    if task is None:
        task = _latest_po_task(session, reference)
    if task is not None and task.operation == PO_OPERATION:
        payload = get_po_task(session, task.task_id)
        request = session.scalar(
            select(ProcurementRequest)
            .where(ProcurementRequest.request_no == task.business_key)
            .options(selectinload(ProcurementRequest.lines))
        )
        if request is None:
            raise AppError(404, "PR_NOT_FOUND", "PR not found")
        form = payload.get("form") or build_create_form_payload(request)
        return {**payload, "form": form, "source": build_create_form_payload(request)}

    request = session.scalar(
        select(ProcurementRequest)
        .where(ProcurementRequest.request_no == reference)
        .options(selectinload(ProcurementRequest.lines))
    )
    if request is None:
        raise AppError(404, "NOT_FOUND", "task_id or pr_no not found")
    form = build_create_form_payload(request)
    return {
        "task_id": None,
        "pr_no": request.request_no,
        "status": _candidate_status(request, None),
        "form": form,
        "source": form,
        "steps": [],
    }


def save_po_draft(
    session: Session,
    task_id: str,
    *,
    header: dict[str, Any],
    lines: list[dict[str, Any]],
) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    if _normalize_task_status(task.status) in {TASK_SUCCESS, TASK_DUPLICATE_BLOCKED}:
        raise AppError(409, "INVALID_STATE", f"Task already finished: {task.status}")
    context = dict(task.context_json or {})
    source = context.get("source") or context.get("form") or {}
    draft = {
        "pr_no": task.business_key,
        "oa_apply_no": source.get("oa_apply_no"),
        "purchase_method": source.get("purchase_method"),
        "award_confirmed_at": source.get("award_confirmed_at"),
        "header": header,
        "lines": lines,
    }
    task.draft_json = _jsonable(draft)
    context["form"] = task.draft_json
    task.context_json = context
    task.status = TASK_DRAFT_EDITING
    session.commit()
    return get_po_task(session, task_id)


def pre_save_verify(
    session: Session,
    task_id: str,
    *,
    header: dict[str, Any] | None = None,
    lines: list[dict[str, Any]] | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    context = dict(task.context_json or {})
    source = context.get("source") or context.get("form") or {}
    draft = task.draft_json or context.get("form") or {}
    header = header if header is not None else (draft.get("header") or {})
    lines = lines if lines is not None else list(draft.get("lines") or [])
    source_lines = list((source.get("lines") or []))
    errors: list[dict[str, Any]] = []

    if task.business_key != (draft.get("pr_no") or task.business_key):
        errors.append({"code": "SOURCE_MISMATCH", "message": "Draft pr_no mismatches task"})
    if not (header.get("supplier_name") or header.get("supplier_code")):
        errors.append({"code": "SUPPLIER_REQUIRED", "message": "Supplier is required"})
    if not lines:
        errors.append({"code": "EMPTY_LINES", "message": "At least one line is required"})
    if source_lines and len(lines) != len(source_lines):
        errors.append(
            {
                "code": "LINE_COUNT_MISMATCH",
                "message": f"line count {len(lines)} != source {len(source_lines)}",
            }
        )
    for index, line in enumerate(lines, 1):
        qty = _dec(line.get("quantity"))
        price = _dec(line.get("unit_price_tax", line.get("unit_price")))
        if qty <= 0:
            errors.append({"code": "INVALID_QTY", "message": f"line {index} quantity must be > 0"})
        if price < 0:
            errors.append({"code": "INVALID_PRICE", "message": f"line {index} unit_price must be >= 0"})
        if not (line.get("material_code") or line.get("material_name")):
            errors.append({"code": "MATERIAL_REQUIRED", "message": f"line {index} material required"})
    draft_total = money(
        sum(
            (_dec(line.get("quantity")) * _dec(line.get("unit_price_tax", line.get("unit_price"))) for line in lines),
            Decimal("0"),
        )
    )
    source_total = _dec((source.get("header") or {}).get("total_amount_tax") or source.get("total_amount") or 0)
    if source_total > 0 and abs(draft_total - source_total) > AMOUNT_TOLERANCE:
        errors.append(
            {
                "code": "AMOUNT_MISMATCH",
                "message": f"total {draft_total} vs source {source_total}",
            }
        )
    existing = session.scalar(
        select(ERPPurchaseOrder).where(ERPPurchaseOrder.pr_no == task.business_key)
    )
    if existing is not None or task.po_no:
        errors.append({"code": "DUPLICATE_BLOCKED", "message": "PO already exists for PR"})

    # Phase2: visual check via VLM when enabled; low confidence gates save.
    vlm = vlm_adapter.maybe_call(
        scenario="pre_save_visual",
        payload={"task_id": task_id, "rule_passed": not errors, "route": task.current_route},
        rpa_ok=False,
    )
    if vlm.get("called"):
        task.vlm_called = True
        if vlm.get("block_save") or vlm.get("low_confidence"):
            errors.append(
                {
                    "code": "VLM_LOW_CONFIDENCE",
                    "message": vlm.get("suggestion") or "VLM low confidence before save",
                }
            )
            task.takeover_flag = True
            record_safety_event(
                session,
                event_type="VLM_LOW_CONFIDENCE",
                severity="WARN",
                task_id=task.task_id,
                batch_id=task.batch_id,
                pr_no=task.business_key,
                stage="PRE_SAVE_VERIFY",
                expected="confidence>=0.55",
                actual=str(vlm.get("confidence")),
                action_taken="wait_user",
            )

    passed = not errors
    task.status = TASK_PRE_SAVE_VERIFY if passed else (
        TASK_WAIT_USER if any(e.get("code") == "VLM_LOW_CONFIDENCE" for e in errors) else TASK_DRAFT_EDITING
    )
    record_step_progress(
        session,
        task.task_id,
        step_name="PRE_SAVE_VERIFY",
        status="success" if passed else "failed",
        expected={"passed": True},
        actual={"passed": passed, "errors": errors, "draft_total": str(draft_total)},
        error_code=None if passed else "VALIDATION_FAILED",
    )
    if commit:
        session.commit()
    return {
        "task_id": task_id,
        "passed": passed,
        "errors": errors,
        "draft_total": draft_total,
        "source_total": source_total,
        "status": task.status,
    }


def link_po_to_procurement(
    session: Session,
    pr_no: str,
    *,
    po_no: str,
    task_id: str | None = None,
    commit: bool = True,
    simulate_failure: bool = False,
) -> dict[str, Any]:
    """采购云回写 po_no（采购云 API）。失败只记状态，禁止重建 ERP PO。"""
    request = session.scalar(
        select(ProcurementRequest)
        .where(ProcurementRequest.request_no == pr_no)
        .options(selectinload(ProcurementRequest.oa_application))
    )
    if request is None:
        raise AppError(404, "PR_NOT_FOUND", "PR not found")
    task = None
    if task_id:
        task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))

    if simulate_failure:
        if task is not None:
            task.writeback_status = "FAILED"
        record_safety_event(
            session,
            event_type="UPSTREAM_WRITEBACK_FAILED",
            severity="WARN",
            task_id=task_id,
            batch_id=task.batch_id if task else None,
            pr_no=pr_no,
            po_no=po_no,
            stage="UPSTREAM_WRITEBACK",
            expected="writeback success",
            actual="simulated failure",
            action_taken="retry_writeback_only",
        )
        record_step_progress(
            session,
            task_id or "unknown",
            step_name="UPSTREAM_WRITEBACK",
            status="failed",
            error_code="UPSTREAM_WRITEBACK_FAILED",
        )
        if commit:
            session.commit()
        return {"status": "FAILED", "pr_no": pr_no, "po_no": po_no, "retryable": True}

    request.po_no = po_no
    request.erp_sync_status = "SUCCESS"
    request.erp_status = "success"
    oa = request.oa_application
    if oa is not None:
        oa.linked_po_no = po_no
        oa.erp_status = "success"
        set_procurement_status(oa, PROCUREMENT_STATUS_AWARDED)
    if task is not None:
        task.writeback_status = "SUCCESS"
        task.po_no = po_no
    record_step_progress(
        session,
        task_id or "unknown",
        step_name="UPSTREAM_WRITEBACK",
        status="success",
        actual={"po_no": po_no},
    )
    if commit:
        session.commit()
    return {"status": "SUCCESS", "pr_no": pr_no, "po_no": po_no, "retryable": False}


def retry_upstream_writeback(
    session: Session, task_id: str, *, simulate_failure: bool = False
) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    if not task.po_no:
        raise AppError(409, "NO_PO", "Cannot retry writeback without local po_no")
    return link_po_to_procurement(
        session,
        task.business_key,
        po_no=task.po_no,
        task_id=task.task_id,
        simulate_failure=simulate_failure,
    )


def pause_po_task(session: Session, task_id: str) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    if _normalize_task_status(task.status) in {TASK_SUCCESS, TASK_DUPLICATE_BLOCKED, TASK_STOPPED}:
        raise AppError(409, "INVALID_STATE", f"Cannot pause task in {task.status}")
    task.status = TASK_PAUSED
    task.is_paused = True
    session.commit()
    return get_po_task(session, task_id)


def resume_po_task(session: Session, task_id: str) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    task.is_paused = False
    task.status = TASK_DRAFT_EDITING
    session.commit()
    return get_po_task(session, task_id)


def stop_po_task(session: Session, task_id: str) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    if _normalize_task_status(task.status) == TASK_SUCCESS and task.po_no:
        # stop must not delete existing PO
        record_safety_event(
            session,
            event_type="MANUAL_STOP",
            severity="INFO",
            task_id=task.task_id,
            batch_id=task.batch_id,
            pr_no=task.business_key,
            po_no=task.po_no,
            stage="STOP",
            action_taken="keep_po",
        )
        session.commit()
        return get_po_task(session, task_id)
    task.status = TASK_STOPPED
    task.finished_at = utcnow()
    record_safety_event(
        session,
        event_type="MANUAL_STOP",
        severity="WARN",
        task_id=task.task_id,
        batch_id=task.batch_id,
        pr_no=task.business_key,
        stage="STOP",
        action_taken="stop",
    )
    session.commit()
    return get_po_task(session, task_id)


def user_response_po_task(
    session: Session, task_id: str, *, action: str, note: str | None = None
) -> dict[str, Any]:
    task = session.scalar(select(AgentTask).where(AgentTask.task_id == task_id))
    if task is None or task.operation != PO_OPERATION:
        raise AppError(404, "TASK_NOT_FOUND", "PO create task not found")
    action = (action or "").lower()
    if action in {"continue", "resume"}:
        task.status = TASK_DRAFT_EDITING
        task.takeover_flag = False
        task.is_paused = False
    elif action in {"stop", "cancel"}:
        return stop_po_task(session, task_id)
    elif action == "retry_writeback":
        return retry_upstream_writeback(session, task_id)
    else:
        raise AppError(422, "INVALID_ACTION", f"Unsupported user action: {action}")
    context = dict(task.context_json or {})
    context["last_user_action"] = {"action": action, "note": note, "at": utcnow().isoformat()}
    task.context_json = context
    session.commit()
    return get_po_task(session, task_id)


def get_batch_excel_path(session: Session, batch_id: str) -> str:
    batch = session.scalar(select(AgentBatch).where(AgentBatch.batch_id == batch_id))
    if batch is None or not batch.excel_snapshot_path:
        raise AppError(404, "EXCEL_NOT_FOUND", "Batch excel tracking not found")
    return batch.excel_snapshot_path


def _refresh_batch_excel(session: Session, batch_id: str | None) -> None:
    if not batch_id:
        return
    batch = session.scalar(select(AgentBatch).where(AgentBatch.batch_id == batch_id))
    if batch is None:
        return
    tasks = list(
        session.scalars(
            select(AgentTask).where(
                AgentTask.batch_id == batch_id, AgentTask.operation == PO_OPERATION
            )
        ).all()
    )
    excel_tasks = []
    excel_lines = []
    success = failed = wait_user = 0
    for task in tasks:
        status = _normalize_task_status(task.status)
        if status == TASK_SUCCESS:
            success += 1
        elif status in {TASK_FAILED, TASK_DUPLICATE_BLOCKED}:
            failed += 1
        elif status == TASK_WAIT_USER:
            wait_user += 1
        form = task.draft_json or (task.context_json or {}).get("form") or {}
        excel_tasks.append(
            {
                "task_id": task.task_id,
                "pr_no": task.business_key,
                "oa_apply_no": form.get("oa_apply_no"),
                "supplier_name": (form.get("header") or {}).get("supplier_name"),
                "total_amount": (form.get("header") or {}).get("total_amount_tax"),
                "status": task.status,
                "retry_count": task.retry_count,
                "po_no": task.po_no,
                "error_code": task.error_code,
                "source_snapshot_hash": task.source_snapshot_hash,
                "created_at": task.created_at.isoformat() if task.created_at else "",
            }
        )
        for line in form.get("lines") or []:
            excel_lines.append({"pr_no": task.business_key, **line})
    path = write_batch_excel(
        batch_id, operator=batch.operator, tasks=excel_tasks, lines=excel_lines
    )
    batch.excel_snapshot_path = str(path)
    batch.success_count = success
    batch.failed_count = failed
    batch.wait_user_count = wait_user
    batch.total_count = len(tasks)


def agent_dashboard_errors(
    session: Session, *, limit: int = 20
) -> list[dict[str, Any]]:
    rows = list(
        session.scalars(
            select(AgentSafetyLog).order_by(AgentSafetyLog.id.desc()).limit(200)
        ).all()
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.event_type] = counts.get(row.event_type, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [{"event_type": key, "count": value} for key, value in ordered]


def po_dashboard_by_method(session: Session) -> list[dict[str, Any]]:
    orders = _filter_orders(session)
    groups: dict[str, dict[str, Any]] = {}
    for order in orders:
        key = order.purchase_method or "unknown"
        item = groups.setdefault(key, {"purchase_method": key, "po_count": 0, "amount": Decimal("0")})
        item["po_count"] += 1
        item["amount"] = money(item["amount"] + _po_amount(order))
    rows = list(groups.values())
    rows.sort(key=lambda item: item["amount"], reverse=True)
    return rows

