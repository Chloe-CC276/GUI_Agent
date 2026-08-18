"""ERP PO Phase-1 routes (PRD v1.1) with v1.0 compatibility aliases."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .api import get_session, page_payload, response
from .errors import AppError
from .erp_po_services import (
    agent_dashboard_errors,
    agent_dashboard_events,
    agent_dashboard_funnel,
    agent_dashboard_summary,
    agent_dashboard_tasks,
    agent_task_steps,
    create_po_batch,
    create_po_from_erp_form,
    get_batch_excel_path,
    get_create_context,
    get_po_detail,
    get_po_lineage,
    get_po_task,
    link_po_to_procurement,
    list_po_candidates,
    mark_po_created,
    pause_po_task,
    po_dashboard_by_department,
    po_dashboard_by_material,
    po_dashboard_by_method,
    po_dashboard_by_supplier,
    po_dashboard_summary,
    po_dashboard_trend,
    pre_save_verify,
    recent_pos,
    resume_po_task,
    retry_po_task,
    retry_upstream_writeback,
    run_po_task,
    save_po_draft,
    start_po_task,
    stop_batch,
    stop_po_task,
    user_response_po_task,
)
from .schemas import (
    ApiResponse,
    POBatchCreateInput,
    POCreateFormInput,
    POLinkInput,
    POMarkCreatedInput,
    POUserResponseInput,
    PurchaseOrderOut,
    TransferOut,
)

router = APIRouter()


def _parse_optional_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise AppError(422, "INVALID_DATE", f"Invalid datetime: {value}") from exc


@router.get("/procurement/erp-po-candidates", response_model=ApiResponse)
def get_procurement_erp_po_candidates(
    status: str | None = None,
    q: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> ApiResponse:
    items, total = list_po_candidates(
        session, status=status, q=q, page=page, page_size=page_size
    )
    return response(page_payload(items, total, page, page_size))


@router.post("/erp/po-batches", response_model=ApiResponse)
def post_erp_po_batches(
    payload: POBatchCreateInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(create_po_batch(session, payload.pr_nos, operator=payload.operator))


@router.post("/erp/po-tasks/{task_id}/start", response_model=ApiResponse)
def post_erp_po_task_start(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(start_po_task(session, task_id))


@router.post("/erp/po-tasks/{task_id}/pause", response_model=ApiResponse)
def post_erp_po_task_pause(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(pause_po_task(session, task_id))


@router.post("/erp/po-tasks/{task_id}/resume", response_model=ApiResponse)
def post_erp_po_task_resume(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(resume_po_task(session, task_id))


@router.post("/erp/po-tasks/{task_id}/stop", response_model=ApiResponse)
def post_erp_po_task_stop(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(stop_po_task(session, task_id))


@router.post("/erp/po-tasks/{task_id}/user-response", response_model=ApiResponse)
def post_erp_po_task_user_response(
    task_id: str,
    payload: POUserResponseInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        user_response_po_task(session, task_id, action=payload.action, note=payload.note)
    )


@router.post("/erp/po-tasks/{task_id}/save-draft", response_model=ApiResponse)
def post_erp_po_task_save_draft(
    task_id: str,
    payload: POCreateFormInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        save_po_draft(session, task_id, header=payload.header, lines=payload.lines)
    )


@router.post("/erp/po-tasks/{task_id}/pre-save-verify", response_model=ApiResponse)
def post_erp_po_task_pre_save_verify(
    task_id: str,
    payload: POCreateFormInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        pre_save_verify(
            session, task_id, header=payload.header, lines=payload.lines, commit=True
        )
    )


@router.post("/erp/po-tasks/{task_id}/save-and-create", response_model=ApiResponse)
def post_erp_po_task_save_and_create(
    task_id: str,
    payload: POCreateFormInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    result = create_po_from_erp_form(
        session,
        task_id,
        header=payload.header,
        lines=payload.lines,
        simulate_readback_fail=payload.simulate_readback_fail,
    )
    order = result.pop("order", None)
    data = {
        **result,
        "order": PurchaseOrderOut.model_validate(order) if order is not None else None,
    }
    return response(data, task_id, result.get("pr_no"))


@router.post("/procurement/requests/{pr_no}/po-link", response_model=ApiResponse)
def post_procurement_po_link(
    pr_no: str,
    payload: POLinkInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        link_po_to_procurement(
            session,
            pr_no,
            po_no=payload.po_no,
            task_id=payload.task_id,
            simulate_failure=payload.simulate_failure,
        )
    )


@router.get("/erp/po-batches/{batch_id}/excel-tracking")
def get_erp_po_batch_excel(
    batch_id: str, session: Session = Depends(get_session)
) -> FileResponse:
    path = Path(get_batch_excel_path(session, batch_id))
    if not path.exists():
        raise AppError(404, "EXCEL_NOT_FOUND", "Excel tracking file missing on disk")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@router.get("/erp/po-tasks/{task_id}", response_model=ApiResponse)
def get_erp_po_task(task_id: str, session: Session = Depends(get_session)) -> ApiResponse:
    return response(get_po_task(session, task_id))


@router.get("/erp/po-create-context/{reference}", response_model=ApiResponse)
def get_po_create_context(
    reference: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(get_create_context(session, reference))


@router.get("/erp/dashboard/agent/summary", response_model=ApiResponse)
def get_dashboard_agent_summary(
    date_from: str | None = None,
    date_to: str | None = None,
    department: str | None = None,
    batch_id: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        agent_dashboard_summary(
            session,
            date_from=_parse_optional_dt(date_from),
            date_to=_parse_optional_dt(date_to),
            department=department,
            batch_id=batch_id,
        )
    )


@router.get("/erp/dashboard/agent/funnel", response_model=ApiResponse)
def get_dashboard_agent_funnel(
    date_from: str | None = None,
    date_to: str | None = None,
    batch_id: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        agent_dashboard_funnel(
            session,
            date_from=_parse_optional_dt(date_from),
            date_to=_parse_optional_dt(date_to),
            batch_id=batch_id,
        )
    )


@router.get("/erp/dashboard/agent/errors", response_model=ApiResponse)
def get_dashboard_agent_errors(
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response({"items": agent_dashboard_errors(session, limit=limit)})


@router.get("/erp/dashboard/agent/tasks", response_model=ApiResponse)
def get_dashboard_agent_tasks(
    status: str | None = None,
    batch_id: str | None = None,
    pr_no: str | None = None,
    po_no: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        {
            "items": agent_dashboard_tasks(
                session, status=status, batch_id=batch_id, pr_no=pr_no, po_no=po_no
            )
        }
    )


@router.get("/erp/agent/tasks/{task_id}/steps", response_model=ApiResponse)
def get_erp_agent_task_steps(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response({"items": agent_task_steps(session, task_id)})


@router.get("/erp/dashboard/agent/security-events", response_model=ApiResponse)
def get_dashboard_agent_security_events(
    event_type: str | None = None,
    severity: str | None = None,
    stage: str | None = None,
    task_id: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> ApiResponse:
    rows, total = agent_dashboard_events(
        session,
        event_type=event_type,
        severity=severity,
        stage=stage,
        task_id=task_id,
        page=page,
        page_size=page_size,
    )
    items = [
        {
            "event_id": row.event_id,
            "task_id": row.task_id,
            "batch_id": row.batch_id,
            "pr_no": row.pr_no,
            "po_no": row.po_no,
            "stage": row.stage,
            "event_type": row.event_type,
            "severity": row.severity,
            "expected": row.expected,
            "actual": row.actual,
            "action_taken": row.action_taken,
            "retry_count": row.retry_count,
            "created_at": row.created_at,
        }
        for row in rows
    ]
    return response(page_payload(items, total, page, page_size))


@router.get("/erp/dashboard/po/summary", response_model=ApiResponse)
def get_dashboard_po_summary(
    date_from: str | None = None,
    date_to: str | None = None,
    department: str | None = None,
    supplier: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        po_dashboard_summary(
            session,
            date_from=_parse_optional_dt(date_from),
            date_to=_parse_optional_dt(date_to),
            department=department,
            supplier=supplier,
        )
    )


@router.get("/erp/dashboard/po/trend", response_model=ApiResponse)
def get_dashboard_po_trend(
    grain: str = Query("day"),
    date_from: str | None = None,
    date_to: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        {
            "items": po_dashboard_trend(
                session,
                grain=grain,
                date_from=_parse_optional_dt(date_from),
                date_to=_parse_optional_dt(date_to),
            )
        }
    )


@router.get("/erp/dashboard/po/by-department", response_model=ApiResponse)
def get_dashboard_po_by_department(
    metric: str = Query("amount"),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response({"items": po_dashboard_by_department(session, metric=metric)})


@router.get("/erp/dashboard/po/by-supplier", response_model=ApiResponse)
def get_dashboard_po_by_supplier(
    limit: int = Query(10, ge=1, le=50),
    metric: str = Query("amount"),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        {"items": po_dashboard_by_supplier(session, limit=limit, metric=metric)}
    )


@router.get("/erp/dashboard/po/by-method", response_model=ApiResponse)
def get_dashboard_po_by_method(session: Session = Depends(get_session)) -> ApiResponse:
    return response({"items": po_dashboard_by_method(session)})


@router.get("/erp/purchase-orders", response_model=ApiResponse)
def get_erp_purchase_orders(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    items, total = recent_pos(session, page=page, page_size=page_size, status=status)
    return response(page_payload(items, total, page, page_size))


@router.get("/erp/purchase-orders/{po_no}", response_model=ApiResponse)
def get_erp_purchase_order_detail(
    po_no: str, session: Session = Depends(get_session)
) -> ApiResponse:
    detail = get_po_detail(session, po_no)
    lineage = get_po_lineage(session, po_no)
    transfers = lineage.get("transfers") or []
    lineage["transfers"] = [TransferOut.model_validate(item) for item in transfers]
    return response({**detail, "lineage": lineage})


# ---- v1.0 compatibility aliases ----


@router.get("/erp/po-candidates", response_model=ApiResponse)
def get_po_candidates_alias(
    status: str | None = None,
    q: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_procurement_erp_po_candidates(status, q, page, page_size, session)


@router.post("/erp/po-tasks/batch", response_model=ApiResponse)
def post_po_tasks_batch_alias(
    payload: POBatchCreateInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return post_erp_po_batches(payload, session)


@router.post("/erp/po-tasks/{task_id}/run", response_model=ApiResponse)
def post_po_task_run_alias(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(run_po_task(session, task_id))


@router.post("/erp/po-tasks/{task_id}/create-po", response_model=ApiResponse)
def post_create_po_alias(
    task_id: str,
    payload: POCreateFormInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return post_erp_po_task_save_and_create(task_id, payload, session)


@router.post("/erp/po-tasks/{task_id}/mark-created", response_model=ApiResponse)
def post_mark_po_created(
    task_id: str,
    payload: POMarkCreatedInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        mark_po_created(
            session,
            task_id,
            po_no=payload.po_no,
            success=payload.success,
            error_code=payload.error_code,
            message=payload.message,
        )
    )


@router.get("/erp/pos/{po_no}", response_model=ApiResponse)
def get_erp_pos_alias(po_no: str, session: Session = Depends(get_session)) -> ApiResponse:
    return response(get_po_detail(session, po_no))


@router.get("/erp/pos/{po_no}/lineage", response_model=ApiResponse)
def get_erp_pos_lineage_alias(
    po_no: str, session: Session = Depends(get_session)
) -> ApiResponse:
    payload = get_po_lineage(session, po_no)
    transfers = payload.get("transfers") or []
    payload["transfers"] = [TransferOut.model_validate(item) for item in transfers]
    return response(payload)


@router.get("/erp/agent-dashboard/summary", response_model=ApiResponse)
def get_agent_dashboard_summary_alias(
    date_from: str | None = None,
    date_to: str | None = None,
    department: str | None = None,
    batch_id: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_agent_summary(date_from, date_to, department, batch_id, session)


@router.get("/erp/agent-dashboard/funnel", response_model=ApiResponse)
def get_agent_dashboard_funnel_alias(
    date_from: str | None = None,
    date_to: str | None = None,
    batch_id: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_agent_funnel(date_from, date_to, batch_id, session)


@router.get("/erp/agent-dashboard/events", response_model=ApiResponse)
def get_agent_dashboard_events_alias(
    event_type: str | None = None,
    severity: str | None = None,
    stage: str | None = None,
    task_id: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_agent_security_events(
        event_type, severity, stage, task_id, page, page_size, session
    )


@router.get("/erp/agent-dashboard/tasks", response_model=ApiResponse)
def get_agent_dashboard_tasks_alias(
    status: str | None = None,
    batch_id: str | None = None,
    pr_no: str | None = None,
    po_no: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_agent_tasks(status, batch_id, pr_no, po_no, session)


@router.get("/erp/agent-dashboard/tasks/{task_id}/steps", response_model=ApiResponse)
def get_agent_dashboard_task_steps_alias(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return get_erp_agent_task_steps(task_id, session)


@router.post("/erp/agent-dashboard/tasks/{task_id}/retry", response_model=ApiResponse)
def post_agent_dashboard_task_retry(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(retry_po_task(session, task_id))


@router.post("/erp/agent-dashboard/batches/{batch_id}/stop", response_model=ApiResponse)
def post_agent_dashboard_batch_stop(
    batch_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(stop_batch(session, batch_id))


@router.get("/erp/po-dashboard/summary", response_model=ApiResponse)
def get_po_dashboard_summary_alias(
    date_from: str | None = None,
    date_to: str | None = None,
    department: str | None = None,
    supplier: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_po_summary(date_from, date_to, department, supplier, session)


@router.get("/erp/po-dashboard/trend", response_model=ApiResponse)
def get_po_dashboard_trend_alias(
    grain: str = Query("day"),
    date_from: str | None = None,
    date_to: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_po_trend(grain, date_from, date_to, session)


@router.get("/erp/po-dashboard/by-department", response_model=ApiResponse)
def get_po_dashboard_by_department_alias(
    metric: str = Query("amount"),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_po_by_department(metric, session)


@router.get("/erp/po-dashboard/by-supplier", response_model=ApiResponse)
def get_po_dashboard_by_supplier_alias(
    limit: int = Query(10, ge=1, le=50),
    metric: str = Query("amount"),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return get_dashboard_po_by_supplier(limit, metric, session)


@router.get("/erp/po-dashboard/by-material", response_model=ApiResponse)
def get_po_dashboard_by_material(
    limit: int = Query(10, ge=1, le=50),
    metric: str = Query("amount"),
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        {"items": po_dashboard_by_material(session, limit=limit, metric=metric)}
    )


@router.get("/erp/po-dashboard/recent-pos", response_model=ApiResponse)
def get_po_dashboard_recent(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    items, total = recent_pos(session, page=page, page_size=page_size, status=status)
    return response(page_payload(items, total, page, page_size))


@router.post("/erp/po-tasks/{task_id}/retry-writeback", response_model=ApiResponse)
def post_retry_writeback(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(retry_upstream_writeback(session, task_id))
