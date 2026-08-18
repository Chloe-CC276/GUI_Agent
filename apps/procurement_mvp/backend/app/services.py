from __future__ import annotations

import secrets
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .errors import AppError
from .oa_services import (
    PROCUREMENT_STATUS_AWARDED,
    PROCUREMENT_STATUS_PREPARING,
    can_submit_procurement,
    is_oa_approved,
    is_oa_in_procurement_pipeline,
    normalize_procurement_status,
    set_procurement_status,
)
from .models import (
    AgentTask,
    AuditEvent,
    CrossSystemDifference,
    BusinessLineage,
    ERPMaterial,
    ERPPurchaseOrder,
    ERPPurchaseOrderLine,
    AwardSource,
    ERPSupplier,
    IntegrationTransfer,
    OAApplication,
    OAApplicationLine,
    ProcurementAttachment,
    ProcurementRequest,
    ProcurementRequestLine,
    utcnow,
)
from .schemas import AttachmentInput, RequestLineInput

AWARD_SOURCE_ALIASES = {
    "线下询比价": "offline_inquiry",
    "框架协议": "framework",
    "商城": "mall",
    "直接采购": "direct",
    "其他": "other",
}

AWARD_SOURCES = set(AWARD_SOURCE_ALIASES.values()) | set(AWARD_SOURCE_ALIASES.keys())

PURCHASE_METHODS = {
    "online",
    "inquiry",
    "bidding",
    "centralized",
    "single",
    "framework",
    "网购",
    "比价",
    "询比价",
    "招标",
    "集中采购",
    "单一来源",
    "框架协议",
}

PURCHASE_METHOD_ALIASES = {
    "网购": "online",
    "比价": "inquiry",
    "询比价": "inquiry",
    "招标": "bidding",
    "集中采购": "centralized",
    "单一来源": "single",
    "框架协议": "framework",
}

MONEY = Decimal("0.01")


def normalize_award_source(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return AWARD_SOURCE_ALIASES.get(text, text)


def normalize_purchase_method(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return PURCHASE_METHOD_ALIASES.get(text, text)


def apply_oa_purchase_method_default(request: ProcurementRequest) -> bool:
    """Default PR purchase method from OA requested_method when still empty."""
    if request.purchase_method_confirmed or request.purchase_type:
        return False
    oa = request.oa_application
    method = normalize_purchase_method(oa.requested_method if oa else None)
    if not method:
        return False
    request.purchase_type = method
    request.purchase_method_confirmed = method
    return True


def money(value: Decimal) -> Decimal:
    return Decimal(value).quantize(MONEY, rounding=ROUND_HALF_UP)


def _unique_number(prefix: str) -> str:
    # UUID-backed numbers avoid count/max races; database uniqueness is the final guard.
    return f"{prefix}{uuid.uuid4().hex[:12].upper()}"


def get_request(session: Session, request_id: int) -> ProcurementRequest:
    request = session.scalar(
        select(ProcurementRequest)
        .where(ProcurementRequest.id == request_id)
        .options(
            selectinload(ProcurementRequest.lines),
            selectinload(ProcurementRequest.attachments),
            selectinload(ProcurementRequest.oa_application),
        )
    )
    if request is None:
        raise AppError(404, "REQUEST_NOT_FOUND", "Procurement request not found")
    return request


def get_existing_task(
    session: Session, task_id: str, business_key: str, operation: str
) -> AgentTask | None:
    return session.scalar(
        select(AgentTask).where(
            AgentTask.task_id == task_id,
            AgentTask.business_key == business_key,
            AgentTask.operation == operation,
        )
    )


def complete_task(
    session: Session,
    task_id: str,
    business_key: str,
    operation: str,
    result: dict[str, Any],
) -> AgentTask:
    task = AgentTask(
        task_id=task_id,
        business_key=business_key,
        operation=operation,
        status="completed",
        result=result,
    )
    session.add(task)
    session.add(
        AuditEvent(
            event_type=operation,
            business_key=business_key,
            task_id=task_id,
            payload=result,
        )
    )
    return task


def _workflow_event(
    session: Session,
    *,
    business_key: str,
    event_type: str,
    status: str = "info",
    operator: str | None = None,
    detail: dict | None = None,
) -> None:
    from .v21_services import record_workflow_event

    record_workflow_event(
        session,
        business_key=business_key,
        event_type=event_type,
        status=status,
        operator=operator,
        detail=detail,
    )


def _record_transfer_task(session: Session, transfer: IntegrationTransfer) -> AgentTask:
    operation = (
        "push-to-procurement"
        if transfer.transfer_type == "OA_TO_PR"
        else "push-to-erp"
    )
    task_status = {
        "pending": "pending",
        "success": "completed",
        "failed": "failed",
        "callback_failed": "partial",
    }[transfer.status]
    result = {
        "transfer_id": transfer.transfer_id,
        "transfer_status": transfer.status,
        "phase": transfer.phase,
        "source_key": transfer.source_key,
        "target_key": transfer.target_key,
        "error_code": transfer.error_code,
        **(transfer.result or {}),
    }
    task = get_existing_task(
        session, transfer.task_id, transfer.source_key, operation
    )
    changed = task is None or task.status != task_status or task.result != result
    if task is None:
        task = AgentTask(
            task_id=transfer.task_id,
            business_key=transfer.source_key,
            operation=operation,
            status=task_status,
            result=result,
        )
        session.add(task)
    else:
        task.status = task_status
        task.result = result
    if changed:
        session.add(
            AuditEvent(
                event_type=f"{operation}:{transfer.phase}",
                business_key=transfer.source_key,
                task_id=transfer.task_id,
                payload=result,
            )
        )
    return task


def _difference(
    session: Session,
    request_id: int,
    request_line_id: int,
    field_name: str,
    source_system: str,
    provided: object,
    authoritative: object,
) -> None:
    if provided is None:
        return
    if isinstance(provided, Decimal) and isinstance(authoritative, Decimal):
        if provided == authoritative:
            return
    elif str(provided) == str(authoritative):
        return
    session.add(
        CrossSystemDifference(
            request_id=request_id,
            request_line_id=request_line_id,
            field_name=field_name,
            source_system=source_system,
            provided_value=str(provided),
            authoritative_value=str(authoritative),
        )
    )


def _replace_lines(
    session: Session,
    request: ProcurementRequest,
    inputs: list[RequestLineInput],
) -> None:
    request.lines.clear()
    session.flush()
    total = Decimal("0")
    for item in inputs:
        material = session.scalar(
            select(ERPMaterial).where(
                ERPMaterial.material_code == item.material_code,
                ERPMaterial.status == "active",
            )
        )
        if material is None:
            raise AppError(
                422,
                "INVALID_ERP_MATERIAL",
                "Material code must reference active ERP master data",
                {"material_code": item.material_code},
            )

        oa_line = None
        if item.oa_line_id is not None:
            oa_line = session.scalar(
                select(OAApplicationLine).where(
                    OAApplicationLine.id == item.oa_line_id,
                    OAApplicationLine.application_id == request.oa_application_id,
                )
            )
            if oa_line is None:
                raise AppError(
                    422,
                    "INVALID_OA_LINE",
                    "OA line does not belong to the selected application",
                    {"oa_line_id": item.oa_line_id},
                )

        unit_price = money(material.standard_price)
        line_amount = money(item.quantity * unit_price)
        line = ProcurementRequestLine(
            oa_line_id=item.oa_line_id,
            material_code=material.material_code,
            material_name=material.material_name,
            specification=material.specification,
            unit=material.unit,
            quantity=item.quantity,
            unit_price=unit_price,
            line_amount=line_amount,
            raw_material_name=oa_line.item_name if oa_line else item.material_name,
            raw_specification=oa_line.specification if oa_line else item.specification,
            raw_unit=item.unit,
            raw_quantity=oa_line.quantity if oa_line else item.quantity,
            raw_estimated_unit_price=(
                oa_line.estimated_unit_price if oa_line else item.unit_price
            ),
        )
        request.lines.append(line)
        session.flush()
        _difference(
            session,
            request.id,
            line.id,
            "material_name",
            "ERP",
            item.material_name,
            material.material_name,
        )
        _difference(
            session,
            request.id,
            line.id,
            "specification",
            "ERP",
            item.specification,
            material.specification,
        )
        _difference(
            session, request.id, line.id, "unit", "ERP", item.unit, material.unit
        )
        _difference(
            session,
            request.id,
            line.id,
            "unit_price",
            "ERP",
            money(item.unit_price) if item.unit_price is not None else None,
            unit_price,
        )
        _difference(
            session,
            request.id,
            line.id,
            "line_amount",
            "SERVER",
            money(item.line_amount) if item.line_amount is not None else None,
            line_amount,
        )
        if oa_line is not None:
            _difference(
                session,
                request.id,
                line.id,
                "quantity",
                "OA",
                item.quantity,
                oa_line.quantity,
            )
            _difference(
                session,
                request.id,
                line.id,
                "oa_item_name",
                "OA",
                item.material_name,
                oa_line.item_name,
            )
        total += line_amount
    request.total_amount = money(total)


def _replace_attachments(
    request: ProcurementRequest, inputs: list[AttachmentInput]
) -> None:
    request.attachments.clear()
    request.attachments.extend(
        ProcurementAttachment(file_name=item.file_name, file_url=item.file_url)
        for item in inputs
    )


def create_request(
    session: Session,
    task_id: str,
    business_key: str,
    oa_application_id: int,
    lines: list[RequestLineInput],
    attachments: list[AttachmentInput],
) -> tuple[ProcurementRequest, bool]:
    existing = get_existing_task(session, task_id, business_key, "create")
    if existing:
        return get_request(session, int(existing.result["request_id"])), True
    oa = session.get(OAApplication, oa_application_id)
    if oa is None:
        raise AppError(404, "OA_NOT_FOUND", "OA application not found")
    request = ProcurementRequest(
        request_no=_unique_number("PR-2026-"),
        oa_application_id=oa_application_id,
        oa_apply_no=oa.application_no,
        oa_title=oa.title,
        oa_applicant=oa.applicant,
        oa_department=oa.department,
        oa_total_budget=oa.total_budget,
        oa_version=oa.oa_version,
        status="draft",
        total_amount=Decimal("0"),
    )
    session.add(request)
    session.flush()
    _replace_lines(session, request, lines)
    _replace_attachments(request, attachments)
    complete_task(
        session, task_id, business_key, "create", {"request_id": request.id}
    )
    _workflow_event(
        session,
        business_key=business_key,
        event_type="pr_created",
        status="success",
        operator=task_id,
        detail={"pr_no": request.request_no, "request_id": request.id},
    )
    session.commit()
    return get_request(session, request.id), False


def update_draft(
    session: Session,
    request_id: int,
    task_id: str,
    business_key: str,
    lines: list[RequestLineInput],
    attachments: list[AttachmentInput] | None,
) -> tuple[ProcurementRequest, bool]:
    existing = get_existing_task(session, task_id, business_key, "update-draft")
    if existing:
        return get_request(session, int(existing.result["request_id"])), True
    request = get_request(session, request_id)
    if request.status != "draft":
        raise AppError(409, "NOT_EDITABLE", "Only draft requests can be edited")
    _replace_lines(session, request, lines)
    if attachments is not None:
        _replace_attachments(request, attachments)
    complete_task(
        session, task_id, business_key, "update-draft", {"request_id": request.id}
    )
    session.commit()
    return get_request(session, request.id), False


def _assert_award_ready(session: Session, request: ProcurementRequest) -> None:
    oa = request.oa_application
    if oa is not None and not is_oa_approved(oa.status):
        raise AppError(409, "OA_NOT_APPROVED", "OA approval_status must be APPROVED")
    apply_oa_purchase_method_default(request)
    method = normalize_purchase_method(
        request.purchase_method_confirmed or request.purchase_type
    )
    if not method:
        raise AppError(
            422,
            "PR_VALIDATION_FAILED",
            "purchase_method is required before ERP submit",
            {"field": "purchase_method"},
        )
    request.purchase_method_confirmed = method
    request.purchase_type = method
    award_source = normalize_award_source(request.award_source)
    if not award_source or award_source not in set(AWARD_SOURCE_ALIASES.values()):
        raise AppError(
            422,
            "PR_VALIDATION_FAILED",
            "award_source is required before ERP submit",
            {"field": "award_source"},
        )
    request.award_source = award_source
    if not request.supplier_code:
        raise AppError(
            422,
            "SUPPLIER_INVALID",
            "supplier_code is required before ERP submit",
        )
    supplier = session.scalar(
        select(ERPSupplier)
        .where(ERPSupplier.supplier_code == request.supplier_code)
        .options(selectinload(ERPSupplier.award_sources))
    )
    if supplier is None or supplier.status != "active":
        raise AppError(
            422,
            "SUPPLIER_INVALID",
            "supplier_code is missing or inactive in ERP master data",
            {"supplier_code": request.supplier_code},
        )
    allowed = {item.code for item in supplier.award_sources if item.status == "active"}
    if allowed and award_source not in allowed:
        raise AppError(
            422,
            "PR_VALIDATION_FAILED",
            "award_source is not allowed for the selected supplier",
            {
                "field": "award_source",
                "supplier_code": request.supplier_code,
                "award_source": award_source,
                "allowed": sorted(allowed),
            },
        )


def validate_request(
    session: Session,
    request_id: int | str,
    task_id: str,
    business_key: str,
    *,
    require_award: bool = False,
) -> tuple[ProcurementRequest, bool]:
    existing = get_existing_task(session, task_id, business_key, "validate")
    if existing:
        return get_request(session, int(existing.result["request_id"])), True
    request = get_request_by_ref(session, request_id)
    if request.status not in {"draft", "validated", "ready"}:
        raise AppError(409, "INVALID_STATE", "Request cannot be validated now")
    if not request.lines:
        raise AppError(422, "EMPTY_REQUEST", "At least one line is required")
    active_codes = set(
        session.scalars(
            select(ERPMaterial.material_code).where(ERPMaterial.status == "active")
        )
    )
    invalid = [line.material_code for line in request.lines if line.material_code not in active_codes]
    if invalid:
        raise AppError(
            422,
            "INVALID_ERP_MATERIAL",
            "Request contains inactive or missing ERP material",
            {"material_codes": invalid},
        )
    invalid_lines = [
        line.id
        for line in request.lines
        if line.quantity <= 0
        or not line.unit
        or line.unit_price < 0
        or line.line_amount != money(line.quantity * line.unit_price)
    ]
    if invalid_lines:
        raise AppError(
            422,
            "INVALID_REQUEST_LINES",
            "Quantity, unit and amount must be valid",
            {"line_ids": invalid_lines},
        )
    if require_award:
        _assert_award_ready(session, request)
    request.status = "validated" if str(request_id).isdigit() else "ready"
    request.final_total_amount_tax = money(
        sum((line.line_amount for line in request.lines), Decimal("0"))
    )
    request.total_amount = request.final_total_amount_tax
    complete_task(
        session, task_id, business_key, "validate", {"request_id": request.id}
    )
    _workflow_event(
        session,
        business_key=business_key,
        event_type="pr_validated",
        status="success",
        operator=task_id,
        detail={"pr_no": request.request_no, "status": request.status},
    )
    session.commit()
    return get_request(session, request.id), False


def prepare_submit(
    session: Session, request_id: int, task_id: str, business_key: str
) -> tuple[ProcurementRequest, str, bool]:
    existing = get_existing_task(session, task_id, business_key, "prepare-submit")
    if existing:
        request = get_request(session, int(existing.result["request_id"]))
        return request, str(existing.result["confirmation_token"]), True
    request = get_request(session, request_id)
    if not is_oa_in_procurement_pipeline(request.oa_application.status):
        raise AppError(
            409,
            "OA_NOT_APPROVED",
            "Only an approved OA application can enter ready state",
            {"oa_status": request.oa_application.status},
        )
    if request.status not in {"draft", "validated"}:
        raise AppError(409, "INVALID_STATE", "Request cannot be prepared now")
    token = secrets.token_urlsafe(32)
    request.confirmation_token = token
    request.status = "ready"
    complete_task(
        session,
        task_id,
        business_key,
        "prepare-submit",
        {"request_id": request.id, "confirmation_token": token},
    )
    session.commit()
    return get_request(session, request.id), token, False


def confirm_submit(
    session: Session,
    request_id: int,
    task_id: str,
    business_key: str,
    confirmation_token: str,
) -> tuple[ProcurementRequest, bool]:
    existing = get_existing_task(session, task_id, business_key, "confirm-submit")
    if existing:
        return get_request(session, int(existing.result["request_id"])), True
    request = get_request(session, request_id)
    if request.status != "ready":
        raise AppError(409, "NOT_READY", "Request must be prepared before submission")
    if not secrets.compare_digest(request.confirmation_token or "", confirmation_token):
        raise AppError(
            409, "INVALID_CONFIRMATION_TOKEN", "Confirmation token is invalid"
        )
    request.status = "submitted"
    request.submitted_at = utcnow()
    request.confirmation_token = None
    complete_task(
        session, task_id, business_key, "confirm-submit", {"request_id": request.id}
    )
    _workflow_event(
        session,
        business_key=business_key,
        event_type="pr_submitted",
        status="success",
        operator=task_id,
        detail={"pr_no": request.request_no},
    )
    session.commit()
    return get_request(session, request.id), False


def get_request_by_ref(session: Session, reference: str | int) -> ProcurementRequest:
    if isinstance(reference, int) or str(reference).isdigit():
        return get_request(session, int(reference))
    request = session.scalar(
        select(ProcurementRequest)
        .where(ProcurementRequest.request_no == str(reference))
        .options(
            selectinload(ProcurementRequest.lines),
            selectinload(ProcurementRequest.attachments),
            selectinload(ProcurementRequest.oa_application),
        )
    )
    if request is None:
        raise AppError(404, "REQUEST_NOT_FOUND", "Procurement request not found")
    return request


def _upsert_lineage(
    session: Session,
    oa_apply_no: str,
    *,
    pr_no: str | None = None,
    po_no: str | None = None,
    task_id: str | None = None,
    status: str,
) -> BusinessLineage:
    lineage = session.scalar(
        select(BusinessLineage).where(BusinessLineage.oa_apply_no == oa_apply_no)
    )
    if lineage is None:
        lineage = BusinessLineage(oa_apply_no=oa_apply_no, latest_status=status)
        session.add(lineage)
    if pr_no:
        lineage.pr_no = pr_no
    if po_no:
        lineage.po_no = po_no
    if task_id:
        lineage.task_id = task_id
    lineage.latest_status = status
    return lineage


def _new_transfer(
    session: Session,
    *,
    source_system: str,
    source_key: str,
    target_system: str,
    transfer_type: str,
    idempotency_key: str,
    task_id: str,
    payload: dict,
) -> IntegrationTransfer:
    transfer = IntegrationTransfer(
        transfer_id=f"TR-{uuid.uuid4().hex}",
        source_system=source_system,
        source_key=source_key,
        target_system=target_system,
        transfer_type=transfer_type,
        status="pending",
        phase="target_pending",
        idempotency_key=idempotency_key,
        task_id=task_id,
        payload=payload,
        result={},
    )
    session.add(transfer)
    _record_transfer_task(session, transfer)
    session.commit()
    return transfer


def _oa_callback(
    session: Session,
    transfer: IntegrationTransfer,
    request: ProcurementRequest,
    simulate_failure: bool,
) -> None:
    oa = session.scalar(
        select(OAApplication).where(OAApplication.application_no == transfer.source_key)
    )
    if oa is None:
        raise AppError(404, "OA_NOT_FOUND", "OA application not found")
    # PR already exists in procurement cloud — link immediately for monitoring.
    oa.linked_pr_no = request.request_no
    set_procurement_status(oa, PROCUREMENT_STATUS_PREPARING)
    if simulate_failure:
        transfer.status = "callback_failed"
        transfer.phase = "source_callback"
        transfer.error_code = "OA_CALLBACK_FAILED"
        transfer.error_message = "Simulated OA callback failure"
        oa.procurement_transfer_status = "callback_failed"
        _upsert_lineage(
            session,
            transfer.source_key,
            pr_no=request.request_no,
            task_id=transfer.task_id,
            status="callback_failed",
        )
        _record_transfer_task(session, transfer)
        session.commit()
        return
    oa.procurement_transfer_status = "success"
    request.procurement_transfer_status = "success"
    _upsert_lineage(
        session,
        oa.application_no,
        pr_no=request.request_no,
        task_id=transfer.task_id,
        status="pr_created",
    )
    transfer.status = "success"
    transfer.phase = "completed"
    transfer.error_code = None
    transfer.error_message = None
    transfer.result = {"pr_no": request.request_no}
    _record_transfer_task(session, transfer)
    session.commit()


def _create_pr_target(
    session: Session,
    transfer: IntegrationTransfer,
    simulate_failure: bool,
) -> ProcurementRequest | None:
    if transfer.target_key:
        return get_request_by_ref(session, transfer.target_key)
    if simulate_failure:
        transfer.status = "failed"
        transfer.phase = "target_create"
        transfer.error_code = "PR_CREATE_FAILED"
        transfer.error_message = "Simulated procurement target failure"
        oa = session.scalar(
            select(OAApplication).where(
                OAApplication.application_no == transfer.source_key
            )
        )
        if oa:
            oa.procurement_transfer_status = "failed"
        _upsert_lineage(
            session,
            transfer.source_key,
            task_id=transfer.task_id,
            status="failed",
        )
        _record_transfer_task(session, transfer)
        session.commit()
        return None
    oa = session.scalar(
        select(OAApplication)
        .where(OAApplication.application_no == transfer.source_key)
        .options(selectinload(OAApplication.lines), selectinload(OAApplication.attachments))
    )
    if oa is None:
        raise AppError(404, "OA_NOT_FOUND", "OA application not found")
    method = normalize_purchase_method(oa.requested_method)
    request = ProcurementRequest(
        request_no=_unique_number("PR-2026-"),
        oa_application_id=oa.id,
        oa_apply_no=oa.application_no,
        oa_title=oa.title,
        oa_applicant=oa.applicant,
        oa_department=oa.department,
        oa_total_budget=oa.total_budget,
        oa_version=oa.oa_version,
        status="draft",
        total_amount=Decimal("0"),
        procurement_transfer_status="target_created",
        budget_project=oa.budget_project_name,
        cost_center=oa.cost_center_code,
        purchase_reason=oa.purchase_reason,
        purchase_type=method,
        purchase_method_confirmed=method,
    )
    total = Decimal("0")
    for oa_line in oa.lines:
        material = session.scalar(
            select(ERPMaterial).where(
                ERPMaterial.material_name == oa_line.item_name,
                ERPMaterial.status == "active",
            )
        )
        quantity = oa_line.quantity
        price = money(material.standard_price if material else oa_line.estimated_unit_price)
        amount = money(quantity * price)
        request.lines.append(
            ProcurementRequestLine(
                oa_line_id=oa_line.id,
                material_code=material.material_code if material else "",
                material_name=material.material_name if material else "",
                specification=material.specification if material else "",
                unit=material.unit if material else "",
                quantity=quantity,
                unit_price=price,
                line_amount=amount,
                raw_material_name=oa_line.item_name,
                raw_specification=oa_line.specification,
                raw_unit=None,
                raw_quantity=oa_line.quantity,
                raw_estimated_unit_price=oa_line.estimated_unit_price,
            )
        )
        total += amount
    request.total_amount = money(total)
    request.attachments.extend(
        ProcurementAttachment(file_name=item.file_name, file_url=item.file_url)
        for item in oa.attachments
    )
    session.add(request)
    session.flush()
    transfer.target_key = request.request_no
    transfer.phase = "target_created"
    transfer.result = {"pr_no": request.request_no}
    oa.linked_pr_no = request.request_no
    oa.procurement_transfer_status = "target_created"
    set_procurement_status(oa, PROCUREMENT_STATUS_PREPARING)
    _upsert_lineage(
        session,
        oa.application_no,
        pr_no=request.request_no,
        task_id=transfer.task_id,
        status="target_created",
    )
    _record_transfer_task(session, transfer)
    session.commit()
    return get_request_by_ref(session, request.request_no)


def push_oa_to_procurement(
    session: Session,
    oa_apply_no: str,
    task_id: str,
    simulate_target_failure: bool = False,
    simulate_callback_failure: bool = False,
) -> tuple[IntegrationTransfer, ProcurementRequest | None, bool]:
    oa = session.scalar(
        select(OAApplication).where(OAApplication.application_no == oa_apply_no)
    )
    if oa is None:
        raise AppError(404, "OA_NOT_FOUND", "OA application not found")
    key = f"OA_TO_PR:{oa.application_no}:{oa.oa_version}"
    transfer = session.scalar(
        select(IntegrationTransfer).where(IntegrationTransfer.idempotency_key == key)
    )
    if transfer:
        request = get_request_by_ref(session, transfer.target_key) if transfer.target_key else None
        return transfer, request, True
    if not is_oa_approved(oa.status):
        raise AppError(409, "OA_NOT_APPROVED", "Only approved OA proposals can be pushed")
    oa.procurement_transfer_status = "pending"
    _upsert_lineage(
        session,
        oa.application_no,
        task_id=task_id,
        status="pending",
    )
    transfer = _new_transfer(
        session,
        source_system="OA",
        source_key=oa.application_no,
        target_system="PROCUREMENT",
        transfer_type="OA_TO_PR",
        idempotency_key=key,
        task_id=task_id,
        payload={"oa_apply_no": oa.application_no, "oa_version": oa.oa_version},
    )
    request = _create_pr_target(session, transfer, simulate_target_failure)
    if request:
        _oa_callback(session, transfer, request, simulate_callback_failure)
        _workflow_event(
            session,
            business_key=oa_apply_no,
            event_type="oa_to_pr_transfer",
            status=transfer.status,
            operator=task_id,
            detail={
                "transfer_id": transfer.transfer_id,
                "pr_no": request.request_no,
                "phase": transfer.phase,
            },
        )
        session.commit()
    else:
        _workflow_event(
            session,
            business_key=oa_apply_no,
            event_type="oa_to_pr_transfer",
            status=transfer.status,
            operator=task_id,
            detail={
                "transfer_id": transfer.transfer_id,
                "error_code": transfer.error_code,
                "phase": transfer.phase,
            },
        )
        session.commit()
    return transfer, request, False


def prepare_erp_submit(
    session: Session, reference: str, task_id: str, business_key: str
) -> tuple[ProcurementRequest, str, bool]:
    request = get_request_by_ref(session, reference)
    if request.status != "ready":
        raise AppError(409, "NOT_READY", "Request must be ready before ERP preparation")
    operation = "prepare-erp-submit"
    existing = get_existing_task(session, task_id, business_key, operation)
    if existing:
        return request, str(existing.result["confirmation_token"]), True
    token = secrets.token_urlsafe(32)
    request.confirmation_token = token
    complete_task(
        session,
        task_id,
        business_key,
        operation,
        {"request_id": request.id, "confirmation_token": token},
    )
    session.commit()
    return get_request_by_ref(session, reference), token, False


def _po_callback(
    session: Session,
    transfer: IntegrationTransfer,
    order: ERPPurchaseOrder,
    simulate_failure: bool,
) -> None:
    request = get_request_by_ref(session, order.pr_no)
    oa = request.oa_application
    # Order already exists in ERP — advance OA monitoring status immediately.
    request.po_no = order.po_no
    oa.linked_po_no = order.po_no
    set_procurement_status(oa, PROCUREMENT_STATUS_AWARDED)
    if simulate_failure:
        transfer.status = "callback_failed"
        transfer.phase = "source_callback"
        transfer.error_code = "PROCUREMENT_CALLBACK_FAILED"
        transfer.error_message = "Simulated procurement/OA callback failure"
        # Keep erp_status as pending until callback retry succeeds.
        _upsert_lineage(
            session,
            order.oa_apply_no or "",
            pr_no=order.pr_no,
            po_no=order.po_no,
            task_id=transfer.task_id,
            status="callback_failed",
        )
        _record_transfer_task(session, transfer)
        session.commit()
        return
    request.erp_status = "success"
    request.erp_sync_status = "SUCCESS"
    request.status = "submitted"
    request.submitted_at = utcnow()
    request.confirmation_token = None
    request.final_total_amount_tax = request.total_amount
    if not request.award_confirmed_at:
        request.award_confirmed_at = utcnow()
    if not request.award_confirmed_by:
        request.award_confirmed_by = transfer.task_id
    oa.erp_status = "success"
    set_procurement_status(oa, PROCUREMENT_STATUS_AWARDED)
    _upsert_lineage(
        session,
        oa.application_no,
        pr_no=request.request_no,
        po_no=order.po_no,
        task_id=transfer.task_id,
        status="po_created",
    )
    transfer.status = "success"
    transfer.phase = "completed"
    transfer.error_code = None
    transfer.error_message = None
    transfer.result = {"po_no": order.po_no}
    _record_transfer_task(session, transfer)
    session.commit()


def _create_po_target(
    session: Session,
    transfer: IntegrationTransfer,
    simulate_failure: bool,
) -> ERPPurchaseOrder | None:
    if transfer.target_key:
        return session.scalar(
            select(ERPPurchaseOrder)
            .where(ERPPurchaseOrder.po_no == transfer.target_key)
            .options(selectinload(ERPPurchaseOrder.lines))
        )
    if simulate_failure:
        transfer.status = "failed"
        transfer.phase = "target_create"
        transfer.error_code = "PO_CREATE_FAILED"
        transfer.error_message = "Simulated ERP target failure"
        request = get_request_by_ref(session, transfer.source_key)
        request.erp_status = "failed"
        request.erp_sync_status = "FAILED"
        _upsert_lineage(
            session,
            request.oa_apply_no or request.oa_application.application_no,
            pr_no=request.request_no,
            task_id=transfer.task_id,
            status="failed",
        )
        _record_transfer_task(session, transfer)
        session.commit()
        return None
    request = get_request_by_ref(session, transfer.source_key)
    order = ERPPurchaseOrder(
        po_no=_unique_number("PO-2026-"),
        pr_no=request.request_no,
        oa_apply_no=request.oa_apply_no,
        submission_version=request.submission_version,
        task_id=transfer.task_id,
        status="created",
        total_amount=request.total_amount,
    )
    for index, line in enumerate(request.lines, 1):
        order.lines.append(
            ERPPurchaseOrderLine(
                line_no=index,
                material_code=line.material_code,
                material_name=line.material_name,
                specification=line.specification,
                unit=line.unit,
                quantity=line.quantity,
                unit_price=line.unit_price,
                line_amount=line.line_amount,
            )
        )
    session.add(order)
    session.flush()
    transfer.target_key = order.po_no
    transfer.phase = "target_created"
    transfer.result = {"po_no": order.po_no}
    _upsert_lineage(
        session,
        request.oa_apply_no or request.oa_application.application_no,
        pr_no=request.request_no,
        po_no=order.po_no,
        task_id=transfer.task_id,
        status="target_created",
    )
    _record_transfer_task(session, transfer)
    session.commit()
    return order


def push_pr_to_erp(
    session: Session,
    reference: str,
    task_id: str,
    business_key: str,
    confirmation_token: str,
    simulate_target_failure: bool = False,
    simulate_callback_failure: bool = False,
) -> tuple[IntegrationTransfer, ERPPurchaseOrder | None, bool]:
    request = get_request_by_ref(session, reference)
    key = f"PR_TO_PO:{request.request_no}:{request.submission_version}"
    transfer = session.scalar(
        select(IntegrationTransfer).where(IntegrationTransfer.idempotency_key == key)
    )
    if transfer:
        order = (
            session.scalar(
                select(ERPPurchaseOrder)
                .where(ERPPurchaseOrder.po_no == transfer.target_key)
                .options(selectinload(ERPPurchaseOrder.lines))
            )
            if transfer.target_key
            else None
        )
        return transfer, order, True
    if request.status != "ready":
        raise AppError(409, "NOT_READY", "Request must be ready before ERP submission")
    if not secrets.compare_digest(request.confirmation_token or "", confirmation_token):
        raise AppError(409, "INVALID_CONFIRMATION_TOKEN", "Confirmation token is invalid")
    if business_key not in {request.request_no, request.oa_apply_no}:
        raise AppError(422, "BUSINESS_KEY_MISMATCH", "business_key does not identify the request")
    request.erp_status = "pending"
    request.erp_sync_status = "SENDING"
    oa = request.oa_application
    if oa is not None:
        # Keep approval status APPROVED; procurement stays PREPARING until PO success.
        set_procurement_status(oa, PROCUREMENT_STATUS_PREPARING)
    _upsert_lineage(
        session,
        request.oa_apply_no or request.oa_application.application_no,
        pr_no=request.request_no,
        task_id=task_id,
        status="pending",
    )
    transfer = _new_transfer(
        session,
        source_system="PROCUREMENT",
        source_key=request.request_no,
        target_system="ERP",
        transfer_type="PR_TO_PO",
        idempotency_key=key,
        task_id=task_id,
        payload={
            "pr_no": request.request_no,
            "submission_version": request.submission_version,
        },
    )
    order = _create_po_target(session, transfer, simulate_target_failure)
    if order:
        _po_callback(session, transfer, order, simulate_callback_failure)
        _workflow_event(
            session,
            business_key=request.request_no,
            event_type="pr_to_po_transfer",
            status=transfer.status,
            operator=task_id,
            detail={
                "transfer_id": transfer.transfer_id,
                "po_no": order.po_no,
                "phase": transfer.phase,
            },
        )
        session.commit()
    else:
        _workflow_event(
            session,
            business_key=request.request_no,
            event_type="pr_to_po_transfer",
            status=transfer.status,
            operator=task_id,
            detail={
                "transfer_id": transfer.transfer_id,
                "error_code": transfer.error_code,
                "phase": transfer.phase,
            },
        )
        session.commit()
    return transfer, order, False


def retry_transfer(
    session: Session, transfer_id: str, task_id: str | None = None
) -> IntegrationTransfer:
    transfer = session.scalar(
        select(IntegrationTransfer).where(IntegrationTransfer.transfer_id == transfer_id)
    )
    if transfer is None:
        raise AppError(404, "TRANSFER_NOT_FOUND", "Integration transfer not found")
    if transfer.status == "success":
        return transfer
    if task_id:
        transfer.task_id = task_id
    transfer.retry_count += 1
    session.commit()
    if transfer.transfer_type == "OA_TO_PR":
        request = _create_pr_target(session, transfer, False)
        if request:
            _oa_callback(session, transfer, request, False)
    elif transfer.transfer_type == "PR_TO_PO":
        order = _create_po_target(session, transfer, False)
        if order:
            _po_callback(session, transfer, order, False)
    else:
        raise AppError(422, "UNSUPPORTED_TRANSFER", "Unsupported transfer type")
    session.refresh(transfer)
    return transfer


def submit_oa_procurement(
    session: Session,
    oa_apply_no: str,
    task_id: str,
    simulate_target_failure: bool = False,
    simulate_callback_failure: bool = False,
) -> dict[str, Any]:
    """Command: OA APPROVED → create/get PR → procurement_status=PREPARING."""
    oa = session.scalar(
        select(OAApplication).where(OAApplication.application_no == oa_apply_no)
    )
    if oa is None:
        raise AppError(404, "OA_NOT_FOUND", "OA application not found")
    if not is_oa_approved(oa.status):
        raise AppError(
            409,
            "OA_NOT_APPROVED",
            "Only approved OA applications can submit procurement",
        )
    if oa.linked_po_no or normalize_procurement_status(oa.procurement_status) == PROCUREMENT_STATUS_AWARDED:
        raise AppError(
            409,
            "ERP_PO_ALREADY_EXISTS",
            "Procurement already awarded; open existing PR/PO",
            {"pr_no": oa.linked_pr_no, "po_no": oa.linked_po_no},
        )
    if not can_submit_procurement(oa):
        raise AppError(
            409,
            "OA_NOT_APPROVED",
            "OA is not eligible to submit procurement",
        )
    transfer, request, replay = push_oa_to_procurement(
        session,
        oa_apply_no,
        task_id,
        simulate_target_failure=simulate_target_failure,
        simulate_callback_failure=simulate_callback_failure,
    )
    oa = session.scalar(
        select(OAApplication).where(OAApplication.application_no == oa_apply_no)
    )
    pr_no = request.request_no if request else (oa.linked_pr_no if oa else None)
    if oa and pr_no and normalize_procurement_status(oa.procurement_status) == "NOT_STARTED":
        # Ensure PREPARING even if callback was simulated failed after target create.
        if oa.linked_pr_no:
            set_procurement_status(oa, PROCUREMENT_STATUS_PREPARING)
            session.commit()
    return {
        "oa_apply_no": oa_apply_no,
        "approval_status": oa.status if oa else None,
        "procurement_status": (
            normalize_procurement_status(oa.procurement_status) if oa else None
        ),
        "pr_no": pr_no,
        "redirect_url": f"/procurement/requests/{pr_no}" if pr_no else None,
        "transfer": transfer,
        "replay": replay,
    }


def patch_procurement_request(
    session: Session,
    reference: str,
    payload: dict[str, Any],
    *,
    confirmed_by: str | None = None,
) -> ProcurementRequest:
    request = get_request_by_ref(session, reference)
    oa = request.oa_application
    if oa is not None and normalize_procurement_status(oa.procurement_status) == PROCUREMENT_STATUS_AWARDED:
        raise AppError(409, "INVALID_STATE", "Awarded procurement request is read-only")
    if request.po_no or request.status == "submitted":
        raise AppError(409, "INVALID_STATE", "Submitted procurement request is read-only")

    header_fields = (
        "budget_project",
        "cost_center",
        "purchase_type",
        "expected_delivery_date",
        "receive_address",
        "purchase_reason",
        "purchase_method_confirmed",
        "award_source",
        "supplier_code",
        "supplier_name",
    )
    for field in header_fields:
        if field in payload and payload[field] is not None:
            setattr(request, field, payload[field])

    if "purchase_method" in payload and payload["purchase_method"] is not None:
        request.purchase_method_confirmed = normalize_purchase_method(
            payload["purchase_method"]
        )
        request.purchase_type = request.purchase_method_confirmed
    if request.purchase_method_confirmed:
        request.purchase_method_confirmed = normalize_purchase_method(
            request.purchase_method_confirmed
        )
        request.purchase_type = request.purchase_method_confirmed
    apply_oa_purchase_method_default(request)

    supplier = None
    if request.supplier_code:
        supplier = session.scalar(
            select(ERPSupplier)
            .where(ERPSupplier.supplier_code == request.supplier_code)
            .options(selectinload(ERPSupplier.award_sources))
        )
        if supplier is None or supplier.status != "active":
            raise AppError(
                422,
                "SUPPLIER_INVALID",
                "supplier_code is missing or inactive in ERP master data",
                {"supplier_code": request.supplier_code},
            )
        request.supplier_name = supplier.supplier_name

    if request.award_source:
        request.award_source = normalize_award_source(request.award_source)
        if request.award_source not in set(AWARD_SOURCE_ALIASES.values()):
            raise AppError(
                422,
                "PR_VALIDATION_FAILED",
                "Invalid award_source",
                {"award_source": request.award_source},
            )
        if supplier is not None:
            allowed = {
                item.code for item in supplier.award_sources if item.status == "active"
            }
            if allowed and request.award_source not in allowed:
                raise AppError(
                    422,
                    "PR_VALIDATION_FAILED",
                    "award_source is not allowed for the selected supplier",
                    {
                        "field": "award_source",
                        "supplier_code": request.supplier_code,
                        "award_source": request.award_source,
                        "allowed": sorted(allowed),
                    },
                )

    lines_payload = payload.get("lines")
    if lines_payload is not None:
        by_id = {line.id: line for line in request.lines}
        for item in lines_payload:
            line_id = item.get("id") or item.get("line_id")
            if line_id is None:
                continue
            line = by_id.get(int(line_id))
            if line is None:
                continue
            if "material_code" in item and item["material_code"] is not None:
                line.material_code = item["material_code"]
            if "material_name" in item and item["material_name"] is not None:
                line.material_name = item["material_name"]
            if "specification" in item and item["specification"] is not None:
                line.specification = item["specification"]
            if "unit" in item and item["unit"] is not None:
                line.unit = item["unit"]
            if "quantity" in item and item["quantity"] is not None:
                line.quantity = Decimal(str(item["quantity"]))
            unit_price = item.get("final_unit_price_tax", item.get("unit_price"))
            if unit_price is not None:
                line.unit_price = money(Decimal(str(unit_price)))
            line.line_amount = money(line.quantity * line.unit_price)

    request.total_amount = money(
        sum((line.line_amount for line in request.lines), Decimal("0"))
    )
    request.final_total_amount_tax = request.total_amount
    if confirmed_by and (
        request.supplier_code
        or request.award_source
        or request.purchase_method_confirmed
    ):
        request.award_confirmed_by = confirmed_by
        request.award_confirmed_at = utcnow()
    session.commit()
    return get_request_by_ref(session, request.request_no)


def submit_procurement_to_erp(
    session: Session,
    pr_no: str,
    task_id: str,
    business_key: str,
    *,
    confirmed_by: str | None = None,
    simulate_target_failure: bool = False,
    simulate_callback_failure: bool = False,
) -> dict[str, Any]:
    """方案 A：定标确认后仅进入 ERP 待建 PO，不经业务 API 直接建单。

    simulate_* 保留兼容入参，方案 A 下不再触发采购云→ERP 直推建 PO。
    """
    _ = (simulate_target_failure, simulate_callback_failure)
    request = get_request_by_ref(session, pr_no)
    oa = request.oa_application

    if request.po_no:
        order = session.scalar(
            select(ERPPurchaseOrder)
            .where(ERPPurchaseOrder.po_no == request.po_no)
            .options(selectinload(ERPPurchaseOrder.lines))
        )
        return {
            "po_no": request.po_no,
            "pr_no": request.request_no,
            "erp_sync_status": request.erp_sync_status or "SUCCESS",
            "procurement_status": (
                normalize_procurement_status(oa.procurement_status) if oa else None
            ),
            "transfer": None,
            "replay": True,
            "upstream_sync_failed": False,
            "order": order,
            "message": "PO already exists",
        }

    if (request.erp_sync_status or "") == "WAITING_PO" and normalize_procurement_status(
        oa.procurement_status if oa else None
    ) == PROCUREMENT_STATUS_AWARDED:
        complete_task(
            session,
            task_id,
            business_key,
            "confirm_award_waiting_po",
            {"pr_no": request.request_no, "erp_sync_status": "WAITING_PO"},
        )
        session.commit()
        return {
            "po_no": None,
            "pr_no": request.request_no,
            "erp_sync_status": "WAITING_PO",
            "procurement_status": PROCUREMENT_STATUS_AWARDED,
            "transfer": None,
            "replay": True,
            "upstream_sync_failed": False,
            "order": None,
            "message": "Already waiting for ERP PO creation",
        }

    if request.status not in {"draft", "validated", "ready"}:
        raise AppError(409, "INVALID_STATE", "Request cannot be submitted to ERP now")

    validate_request(
        session,
        pr_no,
        f"{task_id}-validate",
        business_key,
        require_award=True,
    )
    request = get_request_by_ref(session, pr_no)
    oa = request.oa_application
    request.award_confirmed_by = confirmed_by or task_id
    request.award_confirmed_at = utcnow()
    request.final_total_amount_tax = request.total_amount
    request.status = "ready"
    request.erp_status = "waiting_po"
    request.erp_sync_status = "WAITING_PO"
    if oa is not None:
        set_procurement_status(oa, PROCUREMENT_STATUS_AWARDED)
    _upsert_lineage(
        session,
        request.oa_apply_no or (oa.application_no if oa else ""),
        pr_no=request.request_no,
        task_id=task_id,
        status="waiting_po",
    )
    _workflow_event(
        session,
        business_key=business_key,
        event_type="award_confirmed_waiting_po",
        status="success",
        operator=task_id,
        detail={"pr_no": request.request_no, "erp_sync_status": "WAITING_PO"},
    )
    complete_task(
        session,
        task_id,
        business_key,
        "confirm_award_waiting_po",
        {"pr_no": request.request_no, "erp_sync_status": "WAITING_PO"},
    )
    session.commit()
    return {
        "po_no": None,
        "pr_no": request.request_no,
        "erp_sync_status": "WAITING_PO",
        "procurement_status": PROCUREMENT_STATUS_AWARDED,
        "transfer": None,
        "replay": False,
        "upstream_sync_failed": False,
        "order": None,
        "message": "Award confirmed; entered ERP waiting-PO list",
    }


def list_active_suppliers(
    session: Session, search: str | None = None
) -> list[ERPSupplier]:
    statement = (
        select(ERPSupplier)
        .where(ERPSupplier.status == "active")
        .options(selectinload(ERPSupplier.award_sources))
    )
    if search:
        term = f"%{search}%"
        statement = statement.where(
            (ERPSupplier.supplier_code.ilike(term))
            | (ERPSupplier.supplier_name.ilike(term))
        )
    return list(session.scalars(statement.order_by(ERPSupplier.supplier_code)).all())


def serialize_supplier(supplier: ERPSupplier) -> dict[str, Any]:
    return {
        "supplier_code": supplier.supplier_code,
        "supplier_name": supplier.supplier_name,
        "status": supplier.status,
        "award_sources": sorted(
            item.code for item in supplier.award_sources if item.status == "active"
        ),
    }


def list_award_sources(session: Session) -> list[AwardSource]:
    return list(
        session.scalars(
            select(AwardSource)
            .where(AwardSource.status == "active")
            .order_by(AwardSource.code)
        ).all()
    )
