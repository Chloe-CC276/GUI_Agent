from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, selectinload

from .errors import AppError
from .models import (
    OAApplication,
    OAApplicationLine,
    OAApprovalHistory,
    OAOutbox,
    utcnow,
)
from .schemas import OAApplicationCreate, OAApplicationUpdate, OALineInput

OA_STATUS_DRAFT = "DRAFT"
OA_STATUS_PENDING_APPROVAL = "PENDING_APPROVAL"
OA_STATUS_IN_APPROVAL = "IN_APPROVAL"
OA_STATUS_APPROVED = "APPROVED"
OA_STATUS_REJECTED = "REJECTED"
# Legacy values formerly written into OA.status (migrated to procurement_status).
OA_STATUS_PROCUREMENT_PREP = "PROCUREMENT_PREP"
OA_STATUS_AWARDED = "AWARDED"
OA_STATUS_ORDER_CREATED = "ORDER_CREATED"

PROCUREMENT_STATUS_NOT_STARTED = "NOT_STARTED"
PROCUREMENT_STATUS_PREPARING = "PREPARING"
PROCUREMENT_STATUS_AWARDED = "AWARDED"

OA_APPROVAL_STATUSES = {
    OA_STATUS_DRAFT,
    OA_STATUS_PENDING_APPROVAL,
    OA_STATUS_IN_APPROVAL,
    OA_STATUS_APPROVED,
    OA_STATUS_REJECTED,
}

OA_PIPELINE_STATUSES = {
    OA_STATUS_APPROVED,
    OA_STATUS_PROCUREMENT_PREP,
    OA_STATUS_AWARDED,
    OA_STATUS_ORDER_CREATED,
}

_STATUS_ALIASES = {
    "draft": OA_STATUS_DRAFT,
    "DRAFT": OA_STATUS_DRAFT,
    "pending_approval": OA_STATUS_PENDING_APPROVAL,
    "PENDING_APPROVAL": OA_STATUS_PENDING_APPROVAL,
    "pending": OA_STATUS_IN_APPROVAL,
    "approving": OA_STATUS_IN_APPROVAL,
    "IN_APPROVAL": OA_STATUS_IN_APPROVAL,
    "approved": OA_STATUS_APPROVED,
    "APPROVED": OA_STATUS_APPROVED,
    "rejected": OA_STATUS_REJECTED,
    "REJECTED": OA_STATUS_REJECTED,
    # Legacy mixed statuses map to APPROVED for approval-gate checks.
    "procurement_prep": OA_STATUS_APPROVED,
    "PROCUREMENT_PREP": OA_STATUS_APPROVED,
    "awarded": OA_STATUS_APPROVED,
    "AWARDED": OA_STATUS_APPROVED,
    "order_created": OA_STATUS_APPROVED,
    "ORDER_CREATED": OA_STATUS_APPROVED,
}

_PROCUREMENT_STATUS_ALIASES = {
    "not_started": PROCUREMENT_STATUS_NOT_STARTED,
    "NOT_STARTED": PROCUREMENT_STATUS_NOT_STARTED,
    "preparing": PROCUREMENT_STATUS_PREPARING,
    "PREPARING": PROCUREMENT_STATUS_PREPARING,
    "procurement_prep": PROCUREMENT_STATUS_PREPARING,
    "PROCUREMENT_PREP": PROCUREMENT_STATUS_PREPARING,
    "awarded": PROCUREMENT_STATUS_AWARDED,
    "AWARDED": PROCUREMENT_STATUS_AWARDED,
    "order_created": PROCUREMENT_STATUS_AWARDED,
    "ORDER_CREATED": PROCUREMENT_STATUS_AWARDED,
}


def normalize_oa_status(status: str | None) -> str | None:
    if status is None:
        return None
    mapped = _STATUS_ALIASES.get(status)
    if mapped:
        return mapped
    upper = status.upper()
    return _STATUS_ALIASES.get(upper, upper)


def normalize_procurement_status(status: str | None) -> str:
    if status is None or status == "":
        return PROCUREMENT_STATUS_NOT_STARTED
    mapped = _PROCUREMENT_STATUS_ALIASES.get(status)
    if mapped:
        return mapped
    upper = status.upper()
    return _PROCUREMENT_STATUS_ALIASES.get(upper, upper)


def oa_status_variants(status: str | None) -> list[str]:
    """Return DB status values that map to the same normalized status."""
    normalized = normalize_oa_status(status)
    if not normalized:
        return []
    variants = {normalized}
    for raw, mapped in _STATUS_ALIASES.items():
        if mapped == normalized:
            variants.add(raw)
    # Keep legacy mixed labels searchable when filtering APPROVED.
    if normalized == OA_STATUS_APPROVED:
        variants.update(
            {
                OA_STATUS_PROCUREMENT_PREP,
                OA_STATUS_AWARDED,
                OA_STATUS_ORDER_CREATED,
                "procurement_prep",
                "awarded",
                "order_created",
            }
        )
    return sorted(variants)


def is_oa_approved(status: str | None) -> bool:
    return normalize_oa_status(status) == OA_STATUS_APPROVED


def is_oa_in_procurement_pipeline(status: str | None) -> bool:
    """True for APPROVED OA (including legacy mixed status labels)."""
    return normalize_oa_status(status) == OA_STATUS_APPROVED


def set_procurement_status(
    application: OAApplication, procurement_status: str
) -> None:
    application.procurement_status = normalize_procurement_status(procurement_status)
    application.procurement_updated_at = utcnow()


def is_pending_approval(application: OAApplication) -> bool:
    status = normalize_oa_status(application.status)
    if status == OA_STATUS_PENDING_APPROVAL:
        return True
    # Legacy: submitted draft before PENDING_APPROVAL existed.
    return status == OA_STATUS_DRAFT and bool(application.is_submitted)


def is_handed_off_to_procurement(application: OAApplication) -> bool:
    """True once a PR has been created / linked for this OA."""
    if application.linked_pr_no:
        return True
    if normalize_procurement_status(application.procurement_status) in {
        PROCUREMENT_STATUS_PREPARING,
        PROCUREMENT_STATUS_AWARDED,
    }:
        return True
    status = (application.procurement_transfer_status or "").lower()
    return status in {"success", "pending", "target_created", "callback_failed"}


def can_submit_procurement(application: OAApplication) -> bool:
    if not is_oa_approved(application.status):
        return False
    if application.linked_po_no:
        return False
    return normalize_procurement_status(application.procurement_status) in {
        PROCUREMENT_STATUS_NOT_STARTED,
        PROCUREMENT_STATUS_PREPARING,
    }


def next_oa_apply_no(session: Session) -> str:
    year = datetime.now(timezone.utc).year
    prefix = f"OA-{year}-"
    rows = session.scalars(
        select(OAApplication.application_no)
        .where(OAApplication.application_no.like(f"{prefix}%"))
        .with_for_update()
    ).all()
    max_seq = 0
    for application_no in rows:
        try:
            max_seq = max(max_seq, int(str(application_no).rsplit("-", 1)[-1]))
        except ValueError:
            continue
    return f"{prefix}{max_seq + 1:04d}"


def _load_application(session: Session, application_id: int) -> OAApplication:
    application = session.scalar(
        select(OAApplication)
        .where(OAApplication.id == application_id)
        .options(
            selectinload(OAApplication.lines),
            selectinload(OAApplication.attachments),
        )
    )
    if application is None:
        raise AppError(404, "OA_NOT_FOUND", "OA application not found")
    return application


def _check_row_version(application: OAApplication, row_version: int) -> None:
    if application.row_version != row_version:
        raise AppError(
            409,
            "STATE_CONFLICT",
            "OA application row_version conflict",
            {
                "expected": row_version,
                "actual": application.row_version,
                "status": application.status,
            },
        )


def _bump(application: OAApplication) -> None:
    application.row_version += 1
    application.updated_at = utcnow()


def _snapshot(application: OAApplication) -> dict[str, Any]:
    return {
        "id": application.id,
        "application_no": application.application_no,
        "title": application.title,
        "status": application.status,
        "oa_version": application.oa_version,
        "is_submitted": application.is_submitted,
        "total_budget": str(application.total_budget),
        "row_version": application.row_version,
        "purchase_reason": application.purchase_reason,
        "urgency_level": application.urgency_level,
        "budget_project_code": application.budget_project_code,
        "budget_project_name": application.budget_project_name,
    }


def _add_history(
    session: Session,
    application: OAApplication,
    action: str,
    from_status: str | None,
    to_status: str | None,
    *,
    operator_id: str | None = None,
    operator_name: str | None = None,
    opinion: str | None = None,
) -> OAApprovalHistory:
    history = OAApprovalHistory(
        oa_apply_no=application.application_no,
        oa_version=application.oa_version,
        action=action,
        from_status=from_status,
        to_status=to_status,
        operator_id=operator_id,
        operator_name=operator_name,
        opinion=opinion,
        snapshot_json=_snapshot(application),
    )
    session.add(history)
    return history


def _replace_lines(application: OAApplication, lines: list[OALineInput] | None) -> None:
    if lines is None:
        return
    application.lines.clear()
    application.lines.extend(
        OAApplicationLine(
            item_name=line.item_name,
            specification=line.specification,
            quantity=line.quantity,
            estimated_unit_price=line.estimated_unit_price,
        )
        for line in lines
    )


def _apply_fields(application: OAApplication, payload: OAApplicationCreate | OAApplicationUpdate) -> None:
    title = payload.resolved_title()
    if title is not None:
        application.title = title
    amount = payload.resolved_amount()
    if amount is not None:
        application.total_budget = amount
    if payload.applicant is not None:
        application.applicant = payload.applicant
    if payload.department is not None:
        application.department = payload.department
    if payload.purchase_reason is not None:
        application.purchase_reason = payload.purchase_reason
    if payload.urgency_level is not None:
        application.urgency_level = payload.urgency_level
    if payload.budget_project_code is not None:
        application.budget_project_code = payload.budget_project_code
    if payload.budget_project_name is not None:
        application.budget_project_name = payload.budget_project_name
    if payload.cost_center_code is not None:
        application.cost_center_code = payload.cost_center_code
    if payload.requested_method is not None:
        application.requested_method = payload.requested_method
    if payload.expected_completion_date is not None:
        application.expected_completion_date = payload.expected_completion_date
    if payload.remark is not None:
        application.remark = payload.remark


def _ensure_editable(application: OAApplication) -> None:
    status = normalize_oa_status(application.status)
    if status not in {OA_STATUS_DRAFT, OA_STATUS_REJECTED}:
        raise AppError(
            409,
            "INVALID_STATE",
            "Only DRAFT or REJECTED applications can be edited",
            {"status": application.status},
        )


def _validate_for_submit(application: OAApplication) -> None:
    missing: list[str] = []
    if not (application.title or "").strip():
        missing.append("title")
    if not (application.department or "").strip():
        missing.append("department")
    if not (application.applicant or "").strip():
        missing.append("applicant")
    if application.total_budget is None or Decimal(application.total_budget) <= 0:
        missing.append("total_budget")
    reason = (application.purchase_reason or "").strip()
    if len(reason) < 10:
        missing.append("purchase_reason")
    if not (application.urgency_level or "").strip():
        missing.append("urgency_level")
    has_budget = bool(
        (application.budget_project_name or "").strip()
        or (application.budget_project_code or "").strip()
    )
    if not has_budget:
        missing.append("budget_project_name|budget_project_code")
    valid_lines = [
        line
        for line in application.lines
        if (line.item_name or "").strip()
        and line.quantity is not None
        and Decimal(line.quantity) > 0
        and line.estimated_unit_price is not None
        and Decimal(line.estimated_unit_price) >= 0
    ]
    if not valid_lines:
        missing.append("lines")
    if missing:
        raise AppError(
            422,
            "VALIDATION_ERROR",
            "OA application submit validation failed",
            {"missing_or_invalid": missing},
        )


def create_application(session: Session, payload: OAApplicationCreate) -> OAApplication:
    title = payload.resolved_title() or ""
    amount = payload.resolved_amount()
    if amount is None:
        amount = Decimal("0")
    application = OAApplication(
        application_no=next_oa_apply_no(session),
        title=title,
        applicant=payload.applicant or "",
        department=payload.department or "",
        status=OA_STATUS_DRAFT,
        procurement_status=PROCUREMENT_STATUS_NOT_STARTED,
        total_budget=amount,
        oa_version=1,
        is_submitted=False,
        urgency_level=payload.urgency_level or "NORMAL",
        purchase_reason=payload.purchase_reason,
        budget_project_code=payload.budget_project_code,
        budget_project_name=payload.budget_project_name,
        cost_center_code=payload.cost_center_code,
        requested_method=payload.requested_method,
        expected_completion_date=payload.expected_completion_date,
        remark=payload.remark,
        row_version=1,
        updated_at=utcnow(),
    )
    _replace_lines(application, payload.lines)
    session.add(application)
    session.commit()
    return _load_application(session, application.id)


def update_application(
    session: Session, application_id: int, payload: OAApplicationUpdate
) -> OAApplication:
    application = _load_application(session, application_id)
    _ensure_editable(application)
    if payload.row_version is not None:
        _check_row_version(application, payload.row_version)
    _apply_fields(application, payload)
    _replace_lines(application, payload.lines)
    # REJECTED edits keep REJECTED until resubmit; DRAFT stays DRAFT.
    _bump(application)
    session.commit()
    return _load_application(session, application.id)


def submit_application(
    session: Session,
    application_id: int,
    row_version: int,
    *,
    operator_id: str | None = None,
    operator_name: str | None = None,
) -> OAApplication:
    application = _load_application(session, application_id)
    if normalize_oa_status(application.status) != OA_STATUS_DRAFT:
        raise AppError(
            409,
            "INVALID_STATE",
            "Only DRAFT applications can be submitted",
            {"status": application.status},
        )
    _check_row_version(application, row_version)
    _validate_for_submit(application)
    from_status = application.status
    application.is_submitted = True
    application.submitted_at = utcnow()
    application.status = OA_STATUS_PENDING_APPROVAL
    _bump(application)
    _add_history(
        session,
        application,
        "SUBMIT",
        from_status,
        application.status,
        operator_id=operator_id,
        operator_name=operator_name,
    )
    session.commit()
    return _load_application(session, application.id)


def start_approval(
    session: Session,
    application_id: int,
    row_version: int,
    *,
    operator_id: str | None = None,
    operator_name: str | None = None,
    current_approver_id: str | None = None,
    current_approver_name: str | None = None,
    opinion: str | None = None,
) -> OAApplication:
    application = _load_application(session, application_id)
    if not is_pending_approval(application):
        raise AppError(
            409,
            "INVALID_STATE",
            "Only pending-approval applications can start approval",
            {"status": application.status, "is_submitted": application.is_submitted},
        )
    _check_row_version(application, row_version)
    from_status = application.status
    application.status = OA_STATUS_IN_APPROVAL
    application.approval_started_at = utcnow()
    if current_approver_id is not None:
        application.current_approver_id = current_approver_id
    if current_approver_name is not None:
        application.current_approver_name = current_approver_name
    _bump(application)
    _add_history(
        session,
        application,
        "START",
        from_status,
        application.status,
        operator_id=operator_id,
        operator_name=operator_name,
        opinion=opinion,
    )
    session.commit()
    return _load_application(session, application.id)


def approve_application(
    session: Session,
    application_id: int,
    row_version: int,
    *,
    operator_id: str | None = None,
    operator_name: str | None = None,
    opinion: str | None = None,
) -> OAApplication:
    application = _load_application(session, application_id)
    if normalize_oa_status(application.status) != OA_STATUS_IN_APPROVAL:
        raise AppError(
            409,
            "INVALID_STATE",
            "Only IN_APPROVAL applications can be approved",
            {"status": application.status},
        )
    _check_row_version(application, row_version)
    from_status = application.status
    application.status = OA_STATUS_APPROVED
    application.approved_time = utcnow()
    application.approved_by = operator_name or operator_id
    application.approval_opinion = opinion
    _bump(application)
    _add_history(
        session,
        application,
        "APPROVE",
        from_status,
        application.status,
        operator_id=operator_id,
        operator_name=operator_name,
        opinion=opinion,
    )
    session.add(
        OAOutbox(
            event_id=str(uuid.uuid4()),
            event_type="oa.application.approved",
            payload={
                "oa_apply_no": application.application_no,
                "oa_version": application.oa_version,
                "application_id": application.id,
                "approved_time": application.approved_time.isoformat()
                if application.approved_time
                else None,
                "total_budget": str(application.total_budget),
                "title": application.title,
            },
            status="pending",
        )
    )
    session.commit()
    return _load_application(session, application.id)


def reject_application(
    session: Session,
    application_id: int,
    row_version: int,
    reason: str,
    *,
    operator_id: str | None = None,
    operator_name: str | None = None,
) -> OAApplication:
    if not (reason or "").strip():
        raise AppError(422, "VALIDATION_ERROR", "Reject reason is required")
    application = _load_application(session, application_id)
    if normalize_oa_status(application.status) != OA_STATUS_IN_APPROVAL:
        raise AppError(
            409,
            "INVALID_STATE",
            "Only IN_APPROVAL applications can be rejected",
            {"status": application.status},
        )
    _check_row_version(application, row_version)
    from_status = application.status
    application.status = OA_STATUS_REJECTED
    application.approval_opinion = reason.strip()
    application.approved_by = operator_name or operator_id
    application.approved_time = None
    _bump(application)
    _add_history(
        session,
        application,
        "REJECT",
        from_status,
        application.status,
        operator_id=operator_id,
        operator_name=operator_name,
        opinion=reason.strip(),
    )
    session.commit()
    return _load_application(session, application.id)


def resubmit_application(
    session: Session,
    application_id: int,
    row_version: int,
    *,
    operator_id: str | None = None,
    operator_name: str | None = None,
    opinion: str | None = None,
) -> OAApplication:
    application = _load_application(session, application_id)
    if normalize_oa_status(application.status) != OA_STATUS_REJECTED:
        raise AppError(
            409,
            "INVALID_STATE",
            "Only REJECTED applications can be resubmitted",
            {"status": application.status},
        )
    _check_row_version(application, row_version)
    _validate_for_submit(application)
    from_status = application.status
    application.status = OA_STATUS_PENDING_APPROVAL
    application.oa_version += 1
    application.is_submitted = True
    application.submitted_at = utcnow()
    application.approval_started_at = None
    application.approved_time = None
    application.approved_by = None
    application.approval_opinion = None
    application.current_approver_id = None
    application.current_approver_name = None
    _bump(application)
    _add_history(
        session,
        application,
        "RESUBMIT",
        from_status,
        application.status,
        operator_id=operator_id,
        operator_name=operator_name,
        opinion=opinion,
    )
    session.commit()
    return _load_application(session, application.id)


def list_approvals(session: Session, queue: str) -> list[OAApplication]:
    statement = select(OAApplication).options(selectinload(OAApplication.lines))
    if queue == "pending_start":
        # 待审批：正式态 PENDING_APPROVAL，兼容旧 DRAFT+is_submitted
        statement = statement.where(
            or_(
                OAApplication.status.in_(
                    oa_status_variants(OA_STATUS_PENDING_APPROVAL)
                ),
                and_(
                    OAApplication.status.in_(oa_status_variants(OA_STATUS_DRAFT)),
                    OAApplication.is_submitted.is_(True),
                ),
            )
        )
    elif queue == "in_approval":
        # Include legacy pending/approving so list and workbench stay in sync.
        statement = statement.where(
            OAApplication.status.in_(oa_status_variants(OA_STATUS_IN_APPROVAL))
        )
    elif queue == "done":
        statement = statement.where(
            OAApplication.status.in_(
                [
                    *oa_status_variants(OA_STATUS_APPROVED),
                    *oa_status_variants(OA_STATUS_REJECTED),
                ]
            )
        )
    else:
        raise AppError(
            422,
            "VALIDATION_ERROR",
            "queue must be pending_start, in_approval, or done",
            {"queue": queue},
        )
    return list(session.scalars(statement.order_by(OAApplication.id)).all())


def list_approved_for_procurement(
    session: Session, since: datetime | date | None = None
) -> list[OAApplication]:
    statement = (
        select(OAApplication)
        .where(OAApplication.status.in_(oa_status_variants(OA_STATUS_APPROVED)))
        .options(selectinload(OAApplication.lines), selectinload(OAApplication.attachments))
    )
    if since is not None:
        if isinstance(since, date) and not isinstance(since, datetime):
            since_dt = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
        else:
            since_dt = since
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        statement = statement.where(
            or_(
                OAApplication.approved_time >= since_dt,
                OAApplication.created_at >= since_dt,
            )
        )
    return list(session.scalars(statement.order_by(OAApplication.id)).all())


def list_approval_history(
    session: Session, application_no: str
) -> list[OAApprovalHistory]:
    return list(
        session.scalars(
            select(OAApprovalHistory)
            .where(OAApprovalHistory.oa_apply_no == application_no)
            .order_by(OAApprovalHistory.id)
        ).all()
    )


def get_application_detail(
    session: Session, application_id: int
) -> tuple[OAApplication, list[OAApprovalHistory]]:
    application = _load_application(session, application_id)
    history = list_approval_history(session, application.application_no)
    return application, history
