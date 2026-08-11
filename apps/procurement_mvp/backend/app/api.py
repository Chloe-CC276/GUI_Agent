from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from .errors import AppError
from .models import (
    AgentTask,
    AuditEvent,
    CrossSystemDifference,
    BusinessLineage,
    ERPMaterial,
    ERPPurchaseOrder,
    IntegrationTransfer,
    OAApplication,
    OAApprovalHistory,
    ProcurementRequest,
)
from .oa_services import (
    approve_application,
    create_application,
    get_application_detail,
    list_approvals,
    list_approved_for_procurement,
    normalize_procurement_status as _normalize_proc_status,
    oa_status_variants,
    reject_application,
    resubmit_application,
    start_approval,
    submit_application,
    update_application,
)
from .schemas import (
    AgentChatInput,
    AgentContinueInput,
    AgentStepResultInput,
    ApiResponse,
    AwardSourceOut,
    BatchExportInput,
    BatchValidateInput,
    ConfirmInput,
    CreateRequestInput,
    DifferenceOut,
    ERPMaterialOut,
    ERPSubmitInput,
    ERPSupplierOut,
    FaultInjectionInput,
    ImportConfirmInput,
    OAApplicationCreate,
    OAApplicationOut,
    OAApplicationUpdate,
    OAApprovalActionInput,
    OAApprovalHistoryOut,
    OARejectInput,
    OASubmitInput,
    OperationInput,
    ProcurementPatchInput,
    ProcurementRequestOut,
    PurchaseMethodPatchInput,
    PurchaseOrderOut,
    RetryInput,
    SubmitErpInput,
    SubmitProcurementInput,
    TaskOut,
    TransferOut,
    UpdateDraftInput,
)
from .agent_runtime import (
    continue_agent_task,
    create_task_from_message,
    get_agent_task_view,
    report_step_result,
)
from .agent_runtime.excel_reader import list_excel_files
from .agent_runtime.intents import QUICK_CHIPS
from .seed import reset_database
from .services import (
    apply_oa_purchase_method_default,
    confirm_submit,
    create_request,
    get_request,
    get_request_by_ref,
    list_active_suppliers,
    list_award_sources,
    patch_procurement_request,
    prepare_erp_submit,
    prepare_submit,
    push_oa_to_procurement,
    push_pr_to_erp,
    retry_transfer,
    serialize_supplier,
    submit_oa_procurement,
    submit_procurement_to_erp,
    update_draft,
    validate_request,
)
from .v21_services import (
    batch_export_requests,
    batch_validate_requests,
    confirm_import,
    get_purchase_method_rules,
    list_active_agent_tasks,
    list_export_candidates,
    list_workflow_events,
    pause_agent_task,
    patch_purchase_method,
    preview_import,
    resolve_export_file,
    resume_agent_task,
    stop_agent_task,
    workbench_summary,
)

router = APIRouter()
s0_router = APIRouter()
v21_router = APIRouter()


def get_session(request: Request) -> Generator[Session, None, None]:
    yield from request.app.state.database.session()


def page_payload(items: list, total: int, page: int, page_size: int) -> dict:
    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": (total + page_size - 1) // page_size,
        },
    }


def response(
    data=None,
    task_id: str | None = None,
    business_key: str | None = None,
    replay: bool = False,
) -> ApiResponse:
    return ApiResponse(
        data=data,
        task_id=task_id,
        business_key=business_key,
        idempotent_replay=replay,
    )


def _oa_out(
    application: OAApplication,
    history: list[OAApprovalHistory] | None = None,
) -> OAApplicationOut:
    payload = OAApplicationOut.model_validate(application)
    if history is None:
        return payload
    return payload.model_copy(
        update={
            "approval_history": [
                OAApprovalHistoryOut.model_validate(item) for item in history
            ]
        }
    )


@router.get("/oa/applications", response_model=ApiResponse)
def list_oa_applications(
    search: str | None = None,
    keyword: str | None = None,
    status: str | None = None,
    procurement_status: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> ApiResponse:
    """List all OA applications for end-to-end monitoring (draft → ERP order)."""
    statement = select(OAApplication)
    count_statement = select(func.count(OAApplication.id))
    filters = []
    term_value = keyword or search
    if term_value:
        term = f"%{term_value}%"
        filters.append(
            or_(
                OAApplication.application_no.ilike(term),
                OAApplication.title.ilike(term),
                OAApplication.applicant.ilike(term),
                OAApplication.department.ilike(term),
            )
        )
    if status:
        filters.append(OAApplication.status.in_(oa_status_variants(status)))
    if procurement_status:
        filters.append(
            OAApplication.procurement_status
            == _normalize_proc_status(procurement_status)
        )
    if filters:
        statement = statement.where(*filters)
        count_statement = count_statement.where(*filters)
    total = session.scalar(count_statement) or 0
    applications = session.scalars(
        statement.options(selectinload(OAApplication.lines))
        .order_by(OAApplication.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    items = [_oa_out(item) for item in applications]
    return response(page_payload(items, total, page, page_size))


@router.post("/oa/applications", response_model=ApiResponse, status_code=201)
def post_oa_application(
    payload: OAApplicationCreate, session: Session = Depends(get_session)
) -> ApiResponse:
    application = create_application(session, payload)
    return response(_oa_out(application))


@router.get("/oa/applications/approved", response_model=ApiResponse)
def list_oa_applications_approved(
    since: datetime | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    applications = list_approved_for_procurement(session, since=since)
    return response({"items": [_oa_out(item) for item in applications]})


@router.get("/oa/applications/{application_id}", response_model=ApiResponse)
def get_oa_application(
    application_id: int, session: Session = Depends(get_session)
) -> ApiResponse:
    application, history = get_application_detail(session, application_id)
    return response(_oa_out(application, history))


@router.put("/oa/applications/{application_id}", response_model=ApiResponse)
def put_oa_application(
    application_id: int,
    payload: OAApplicationUpdate,
    session: Session = Depends(get_session),
) -> ApiResponse:
    application = update_application(session, application_id, payload)
    return response(_oa_out(application))


@router.post("/oa/applications/{application_id}/submit", response_model=ApiResponse)
def post_oa_submit(
    application_id: int,
    payload: OASubmitInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    application = submit_application(
        session,
        application_id,
        payload.row_version,
        operator_id=payload.operator_id,
        operator_name=payload.operator_name,
    )
    application, history = get_application_detail(session, application.id)
    return response(_oa_out(application, history))


@router.post("/oa/applications/{application_id}/resubmit", response_model=ApiResponse)
def post_oa_resubmit(
    application_id: int,
    payload: OAApprovalActionInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    application = resubmit_application(
        session,
        application_id,
        payload.row_version,
        operator_id=payload.operator_id,
        operator_name=payload.operator_name,
        opinion=payload.opinion,
    )
    application, history = get_application_detail(session, application.id)
    return response(_oa_out(application, history))


@router.get("/oa/approvals", response_model=ApiResponse)
def get_oa_approvals(
    queue: str = Query(..., pattern="^(pending_start|in_approval|done)$"),
    session: Session = Depends(get_session),
) -> ApiResponse:
    applications = list_approvals(session, queue)
    return response({"items": [_oa_out(item) for item in applications], "queue": queue})


@router.post(
    "/oa/applications/{application_id}/start-approval", response_model=ApiResponse
)
def post_oa_start_approval(
    application_id: int,
    payload: OAApprovalActionInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    application = start_approval(
        session,
        application_id,
        payload.row_version,
        operator_id=payload.operator_id,
        operator_name=payload.operator_name,
        current_approver_id=payload.current_approver_id,
        current_approver_name=payload.current_approver_name,
        opinion=payload.opinion,
    )
    application, history = get_application_detail(session, application.id)
    return response(_oa_out(application, history))


@router.post("/oa/applications/{application_id}/approve", response_model=ApiResponse)
def post_oa_approve(
    application_id: int,
    payload: OAApprovalActionInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    application = approve_application(
        session,
        application_id,
        payload.row_version,
        operator_id=payload.operator_id,
        operator_name=payload.operator_name,
        opinion=payload.opinion,
    )
    application, history = get_application_detail(session, application.id)
    return response(_oa_out(application, history))


@router.post("/oa/applications/{application_id}/reject", response_model=ApiResponse)
def post_oa_reject(
    application_id: int,
    payload: OARejectInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    application = reject_application(
        session,
        application_id,
        payload.row_version,
        payload.reason,
        operator_id=payload.operator_id,
        operator_name=payload.operator_name,
    )
    application, history = get_application_detail(session, application.id)
    return response(_oa_out(application, history))


@s0_router.post(
    "/oa/proposals/{oa_apply_no}/push-to-procurement",
    response_model=ApiResponse,
)
def post_oa_to_procurement(
    oa_apply_no: str,
    payload: FaultInjectionInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    transfer, request, replay = push_oa_to_procurement(
        session,
        oa_apply_no,
        payload.task_id,
        payload.simulate_target_failure,
        payload.simulate_callback_failure,
    )
    data = {
        "pr_no": request.request_no if request else None,
        "transfer": TransferOut.model_validate(transfer),
    }
    return response(data, payload.task_id, oa_apply_no, replay)


@s0_router.post(
    "/oa/applications/{oa_apply_no}/submit-procurement",
    response_model=ApiResponse,
)
def post_submit_procurement(
    oa_apply_no: str,
    payload: SubmitProcurementInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    result = submit_oa_procurement(
        session,
        oa_apply_no,
        payload.task_id,
        payload.simulate_target_failure,
        payload.simulate_callback_failure,
    )
    transfer = result.pop("transfer")
    replay = bool(result.pop("replay", False))
    data = {
        **result,
        "transfer": TransferOut.model_validate(transfer) if transfer else None,
    }
    return response(data, payload.task_id, oa_apply_no, replay)


@s0_router.get("/oa/proposals/{oa_apply_no}/lineage", response_model=ApiResponse)
def get_oa_lineage(
    oa_apply_no: str, session: Session = Depends(get_session)
) -> ApiResponse:
    lineage = session.scalar(
        select(BusinessLineage).where(BusinessLineage.oa_apply_no == oa_apply_no)
    )
    transfers = session.scalars(
        select(IntegrationTransfer)
        .where(
            or_(
                IntegrationTransfer.source_key == oa_apply_no,
                IntegrationTransfer.source_key
                == (lineage.pr_no if lineage is not None else "__none__"),
            )
        )
        .order_by(IntegrationTransfer.id)
    ).all()
    if lineage is None and not transfers:
        raise AppError(404, "LINEAGE_NOT_FOUND", "Business lineage not found")
    data = {
        "oa_apply_no": oa_apply_no,
        "pr_no": lineage.pr_no if lineage else None,
        "po_no": lineage.po_no if lineage else None,
        "task_id": lineage.task_id if lineage else None,
        "task_ids": list(dict.fromkeys(item.task_id for item in transfers)),
        "latest_status": lineage.latest_status if lineage else None,
        "transfers": [TransferOut.model_validate(item) for item in transfers],
    }
    return response(data, business_key=oa_apply_no)


@router.get("/erp/suppliers", response_model=ApiResponse)
def get_erp_suppliers(
    search: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    items = [
        ERPSupplierOut.model_validate(serialize_supplier(item))
        for item in list_active_suppliers(session, search)
    ]
    return response({"items": items})


@router.get("/erp/award-sources", response_model=ApiResponse)
def get_erp_award_sources(session: Session = Depends(get_session)) -> ApiResponse:
    items = [AwardSourceOut.model_validate(item) for item in list_award_sources(session)]
    return response({"items": items})


@router.get("/erp/materials", response_model=ApiResponse)
def list_erp_materials(
    search: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> ApiResponse:
    statement = select(ERPMaterial)
    count_statement = select(func.count(ERPMaterial.id))
    if search:
        term = f"%{search}%"
        condition = or_(
            ERPMaterial.material_code.ilike(term),
            ERPMaterial.material_name.ilike(term),
            ERPMaterial.specification.ilike(term),
        )
        statement = statement.where(condition)
        count_statement = count_statement.where(condition)
    total = session.scalar(count_statement) or 0
    materials = session.scalars(
        statement.order_by(ERPMaterial.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    items = [ERPMaterialOut.model_validate(item) for item in materials]
    return response(page_payload(items, total, page, page_size))


@router.get("/erp/materials/{material_code}", response_model=ApiResponse)
def get_erp_material(
    material_code: str, session: Session = Depends(get_session)
) -> ApiResponse:
    material = session.scalar(
        select(ERPMaterial).where(ERPMaterial.material_code == material_code)
    )
    if material is None:
        raise AppError(404, "MATERIAL_NOT_FOUND", "ERP material not found")
    return response(ERPMaterialOut.model_validate(material))


@s0_router.get("/erp/orders", response_model=ApiResponse)
def list_erp_orders(
    pr_no: str | None = None,
    oa_apply_no: str | None = None,
    po_no: str | None = None,
    status: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> ApiResponse:
    filters = []
    for value, column in (
        (pr_no, ERPPurchaseOrder.pr_no),
        (oa_apply_no, ERPPurchaseOrder.oa_apply_no),
        (po_no, ERPPurchaseOrder.po_no),
        (status, ERPPurchaseOrder.status),
    ):
        if value:
            filters.append(column == value)
    total = session.scalar(select(func.count(ERPPurchaseOrder.id)).where(*filters)) or 0
    orders = session.scalars(
        select(ERPPurchaseOrder)
        .where(*filters)
        .options(selectinload(ERPPurchaseOrder.lines))
        .order_by(ERPPurchaseOrder.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return response(
        page_payload(
            [PurchaseOrderOut.model_validate(item) for item in orders],
            total,
            page,
            page_size,
        )
    )


@s0_router.get("/erp/orders/{po_no}", response_model=ApiResponse)
def get_erp_order(po_no: str, session: Session = Depends(get_session)) -> ApiResponse:
    order = session.scalar(
        select(ERPPurchaseOrder)
        .where(ERPPurchaseOrder.po_no == po_no)
        .options(selectinload(ERPPurchaseOrder.lines))
    )
    if order is None:
        raise AppError(404, "PO_NOT_FOUND", "ERP purchase order not found")
    return response(PurchaseOrderOut.model_validate(order))


@router.post("/procurement/requests", response_model=ApiResponse, status_code=201)
def post_procurement_request(
    payload: CreateRequestInput, session: Session = Depends(get_session)
) -> ApiResponse:
    request, replay = create_request(
        session,
        payload.task_id,
        payload.business_key,
        payload.oa_application_id,
        payload.lines,
        payload.attachments,
    )
    return response(
        ProcurementRequestOut.model_validate(request),
        payload.task_id,
        payload.business_key,
        replay,
    )


@s0_router.get("/procurement/requests", response_model=ApiResponse)
def list_procurement_requests(
    oa_apply_no: str | None = None,
    pr_no: str | None = None,
    status: str | None = None,
    erp_status: str | None = None,
    procurement_status: str | None = None,
    q: str | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> ApiResponse:
    filters = []
    if oa_apply_no:
        filters.append(ProcurementRequest.oa_apply_no == oa_apply_no)
    if pr_no:
        filters.append(ProcurementRequest.request_no == pr_no)
    if status in {"PREPARING", "AWARDED", "NOT_STARTED"}:
        procurement_status = procurement_status or status
    elif status:
        filters.append(ProcurementRequest.status == status)
    if erp_status:
        filters.append(ProcurementRequest.erp_status == erp_status)
    term_value = q or search
    if term_value:
        term = f"%{term_value}%"
        filters.append(
            or_(
                ProcurementRequest.request_no.ilike(term),
                ProcurementRequest.oa_apply_no.ilike(term),
                ProcurementRequest.oa_title.ilike(term),
                ProcurementRequest.oa_applicant.ilike(term),
            )
        )
    statement = select(ProcurementRequest)
    if procurement_status:
        normalized = _normalize_proc_status(procurement_status)
        statement = statement.join(
            OAApplication, ProcurementRequest.oa_application_id == OAApplication.id
        )
        filters.append(OAApplication.procurement_status == normalized)
        # List page only shows OA-submitted procurement tasks.
        if normalized == "PREPARING":
            filters.append(ProcurementRequest.po_no.is_(None))
        elif normalized == "AWARDED":
            filters.append(ProcurementRequest.po_no.is_not(None))
    if filters:
        statement = statement.where(*filters)
    count_statement = select(func.count()).select_from(statement.order_by(None).subquery())
    total = session.scalar(count_statement) or 0
    requests = session.scalars(
        statement.options(
            selectinload(ProcurementRequest.lines),
            selectinload(ProcurementRequest.attachments),
            selectinload(ProcurementRequest.oa_application),
        )
        .order_by(ProcurementRequest.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    items = []
    for item in requests:
        payload = ProcurementRequestOut.model_validate(item).model_dump()
        oa = item.oa_application
        if oa is not None:
            payload["oa_approval_status"] = oa.status
            payload["procurement_status"] = _normalize_proc_status(oa.procurement_status)
        elif item.po_no:
            payload["procurement_status"] = "AWARDED"
        else:
            payload["procurement_status"] = "PREPARING"
        items.append(payload)
    return response(page_payload(items, total, page, page_size))


@s0_router.get("/procurement/requests/{request_id}", response_model=ApiResponse)
def get_procurement_request(
    request_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    request = get_request_by_ref(session, request_id)
    if apply_oa_purchase_method_default(request):
        session.commit()
        request = get_request_by_ref(session, request_id)
    payload = ProcurementRequestOut.model_validate(request).model_dump()
    oa = request.oa_application
    if oa is not None:
        payload["oa_approval_status"] = oa.status
        payload["procurement_status"] = _normalize_proc_status(oa.procurement_status)
        payload["oa_approved_time"] = oa.approved_time
        payload["oa_requested_method"] = oa.requested_method
    return response(payload)


@s0_router.patch("/procurement/requests/{pr_no}", response_model=ApiResponse)
def patch_procurement(
    pr_no: str,
    payload: ProcurementPatchInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    request = patch_procurement_request(
        session,
        pr_no,
        payload.model_dump(exclude_unset=True),
        confirmed_by=payload.award_confirmed_by,
    )
    return response(ProcurementRequestOut.model_validate(request))


@s0_router.post("/procurement/requests/{pr_no}/submit-erp", response_model=ApiResponse)
def post_submit_erp(
    pr_no: str,
    payload: SubmitErpInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    result = submit_procurement_to_erp(
        session,
        pr_no,
        payload.task_id,
        payload.business_key,
        confirmed_by=payload.confirmed_by,
        simulate_target_failure=payload.simulate_target_failure,
        simulate_callback_failure=payload.simulate_callback_failure,
    )
    transfer = result.get("transfer")
    data = {
        "po_no": result.get("po_no"),
        "pr_no": result.get("pr_no"),
        "erp_sync_status": result.get("erp_sync_status"),
        "procurement_status": result.get("procurement_status"),
        "upstream_sync_failed": result.get("upstream_sync_failed", False),
        "transfer": TransferOut.model_validate(transfer) if transfer else None,
    }
    return response(data, payload.task_id, payload.business_key, bool(result.get("replay")))


@router.put("/procurement/requests/{request_id}/draft", response_model=ApiResponse)
def put_procurement_draft(
    request_id: int,
    payload: UpdateDraftInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    request, replay = update_draft(
        session,
        request_id,
        payload.task_id,
        payload.business_key,
        payload.lines,
        payload.attachments,
    )
    return response(
        ProcurementRequestOut.model_validate(request),
        payload.task_id,
        payload.business_key,
        replay,
    )


@s0_router.post("/procurement/requests/{request_id}/validate", response_model=ApiResponse)
def post_validate(
    request_id: str,
    payload: OperationInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    request, replay = validate_request(
        session, request_id, payload.task_id, payload.business_key
    )
    return response(
        ProcurementRequestOut.model_validate(request),
        payload.task_id,
        payload.business_key,
        replay,
    )


@s0_router.post(
    "/procurement/requests/{pr_no}/prepare-erp-submit", response_model=ApiResponse
)
def post_prepare_erp_submit(
    pr_no: str,
    payload: OperationInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    request, token, replay = prepare_erp_submit(
        session, pr_no, payload.task_id, payload.business_key
    )
    data = ProcurementRequestOut.model_validate(request).model_dump()
    data["confirmation_token"] = token
    return response(data, payload.task_id, payload.business_key, replay)


@s0_router.post("/procurement/requests/{pr_no}/push-to-erp", response_model=ApiResponse)
def post_push_to_erp(
    pr_no: str,
    payload: ERPSubmitInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    transfer, order, replay = push_pr_to_erp(
        session,
        pr_no,
        payload.task_id,
        payload.business_key,
        payload.confirmation_token,
        payload.simulate_target_failure,
        payload.simulate_callback_failure,
    )
    data = {
        "po_no": order.po_no if order else None,
        "transfer": TransferOut.model_validate(transfer),
    }
    return response(data, payload.task_id, payload.business_key, replay)


@router.post(
    "/procurement/requests/{request_id}/prepare-submit", response_model=ApiResponse
)
def post_prepare_submit(
    request_id: int,
    payload: OperationInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    request, token, replay = prepare_submit(
        session, request_id, payload.task_id, payload.business_key
    )
    data = ProcurementRequestOut.model_validate(request).model_dump()
    data["confirmation_token"] = token
    return response(data, payload.task_id, payload.business_key, replay)


@router.post(
    "/procurement/requests/{request_id}/confirm-submit", response_model=ApiResponse
)
def post_confirm_submit(
    request_id: int,
    payload: ConfirmInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    request, replay = confirm_submit(
        session,
        request_id,
        payload.task_id,
        payload.business_key,
        payload.confirmation_token,
    )
    return response(
        ProcurementRequestOut.model_validate(request),
        payload.task_id,
        payload.business_key,
        replay,
    )


@router.get("/differences", response_model=ApiResponse)
def list_differences(
    request_id: int | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> ApiResponse:
    statement = select(CrossSystemDifference)
    count_statement = select(func.count(CrossSystemDifference.id))
    if request_id is not None:
        statement = statement.where(CrossSystemDifference.request_id == request_id)
        count_statement = count_statement.where(
            CrossSystemDifference.request_id == request_id
        )
    total = session.scalar(count_statement) or 0
    differences = session.scalars(
        statement.order_by(CrossSystemDifference.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    items = [DifferenceOut.model_validate(item) for item in differences]
    return response(page_payload(items, total, page, page_size))


@router.get("/agent/tasks/active", response_model=ApiResponse)
def get_active_agent_tasks(session: Session = Depends(get_session)) -> ApiResponse:
    return response({"items": list_active_agent_tasks(session)})


@router.get("/agent/tasks/{task_id}", response_model=ApiResponse)
def get_agent_task(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    view = get_agent_task_view(session, task_id)
    return response(view, view["task_id"], view["business_key"], False)


@router.get("/agent/chips", response_model=ApiResponse)
def get_agent_chips() -> ApiResponse:
    return response({"items": QUICK_CHIPS})


@router.post("/agent/chat", response_model=ApiResponse)
def post_agent_chat(
    payload: AgentChatInput, session: Session = Depends(get_session)
) -> ApiResponse:
    data = create_task_from_message(
        session,
        message=payload.message,
        page_context={"route": payload.route, "business_key": payload.business_key},
        folder_path=payload.folder_path,
        excel_path=payload.excel_path,
    )
    task = data.get("task") or {}
    return response(
        data,
        task.get("task_id"),
        task.get("business_key") or payload.business_key,
        False,
    )


@router.post("/agent/tasks/{task_id}/continue", response_model=ApiResponse)
async def post_agent_continue(
    task_id: str,
    folder_path: str | None = Form(default=None),
    excel_path: str | None = Form(default=None),
    oa_id: int | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    session: Session = Depends(get_session),
) -> ApiResponse:
    upload = await file.read() if file is not None else None
    view = continue_agent_task(
        session,
        task_id,
        payload={
            "folder_path": folder_path,
            "excel_path": excel_path,
            "oa_id": oa_id,
        },
        upload=upload if upload else None,
        upload_name=file.filename if file is not None else None,
    )
    return response(view, view["task_id"], view["business_key"], False)


@router.post("/agent/tasks/{task_id}/step-result", response_model=ApiResponse)
def post_agent_step_result(
    task_id: str,
    payload: AgentStepResultInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    view = report_step_result(
        session,
        task_id,
        step_id=payload.step_id,
        status=payload.status,
        actual=payload.actual,
        detail=payload.detail,
    )
    return response(view, view["task_id"], view["business_key"], False)


@router.get("/agent/fs/list-excels", response_model=ApiResponse)
def get_agent_list_excels(path: str = Query(min_length=1)) -> ApiResponse:
    return response({"items": list_excel_files(path), "path": path})


@s0_router.get("/integration/transfers", response_model=ApiResponse)
def list_transfers(
    business_key: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> ApiResponse:
    keys = [business_key] if business_key else []
    if business_key:
        lineage = session.scalar(
            select(BusinessLineage).where(
                or_(
                    BusinessLineage.oa_apply_no == business_key,
                    BusinessLineage.pr_no == business_key,
                    BusinessLineage.po_no == business_key,
                )
            )
        )
        if lineage:
            keys = [
                key
                for key in (lineage.oa_apply_no, lineage.pr_no, lineage.po_no)
                if key
            ]
    condition = (
        or_(
            IntegrationTransfer.source_key.in_(keys),
            IntegrationTransfer.target_key.in_(keys),
        )
        if keys
        else None
    )
    filters = [condition] if condition is not None else []
    total = session.scalar(select(func.count(IntegrationTransfer.id)).where(*filters)) or 0
    transfers = session.scalars(
        select(IntegrationTransfer)
        .where(*filters)
        .order_by(IntegrationTransfer.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return response(
        page_payload(
            [TransferOut.model_validate(item) for item in transfers],
            total,
            page,
            page_size,
        )
    )


@s0_router.post("/integration/transfers/{transfer_id}/retry", response_model=ApiResponse)
def post_retry_transfer(
    transfer_id: str,
    payload: RetryInput | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    transfer = retry_transfer(session, transfer_id, payload.task_id if payload else None)
    return response(
        TransferOut.model_validate(transfer),
        transfer.task_id,
        transfer.source_key,
        transfer.status == "success",
    )


@s0_router.get("/lineage", response_model=ApiResponse)
def get_lineage_alias(
    oa_apply_no: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return get_oa_lineage(oa_apply_no, session)


@v21_router.get("/workbench/summary", response_model=ApiResponse)
def get_workbench_summary(session: Session = Depends(get_session)) -> ApiResponse:
    return response(workbench_summary(session))


@v21_router.get("/workbench/events", response_model=ApiResponse)
def get_workbench_events(
    business_key: str | None = None,
    session: Session = Depends(get_session),
) -> ApiResponse:
    return response(
        {"items": list_workflow_events(session, business_key)},
        business_key=business_key,
    )


@v21_router.post("/procurement/imports/preview", response_model=ApiResponse)
async def post_import_preview(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> ApiResponse:
    filename = file.filename or "import.xlsx"
    if not filename.lower().endswith(".xlsx"):
        raise AppError(422, "INVALID_FILE_TYPE", "Only .xlsx files are accepted")
    content = await file.read()
    if not content:
        raise AppError(422, "EMPTY_FILE", "Uploaded file is empty")
    if len(content) > 5 * 1024 * 1024:
        raise AppError(422, "FILE_TOO_LARGE", "Excel file exceeds 5MB limit")
    data = preview_import(session, filename, content)
    return response(data, business_key=data["import_batch_id"])


@v21_router.post("/procurement/imports/confirm", response_model=ApiResponse)
def post_import_confirm(
    payload: ImportConfirmInput, session: Session = Depends(get_session)
) -> ApiResponse:
    data = confirm_import(
        session,
        import_batch_id=payload.import_batch_id,
        task_id=payload.task_id,
        business_key=payload.business_key,
        pr_no=payload.pr_no,
        oa_application_id=payload.oa_application_id,
        row_nos=payload.row_nos,
    )
    return response(
        data,
        payload.task_id,
        payload.business_key,
        bool(data.get("idempotent_replay")),
    )


@v21_router.get("/config/purchase-method-rules", response_model=ApiResponse)
def get_purchase_method_rules_api() -> ApiResponse:
    return response(get_purchase_method_rules())


@v21_router.get("/procurement/requests/export-candidates", response_model=ApiResponse)
def get_export_candidates(
    department: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
    min_amount: Decimal | None = None,
    max_amount: Decimal | None = None,
    exportable_only: bool = False,
    session: Session = Depends(get_session),
) -> ApiResponse:
    items = list_export_candidates(
        session,
        department=department,
        status=status,
        keyword=keyword,
        min_amount=min_amount,
        max_amount=max_amount,
        exportable_only=exportable_only,
    )
    return response({"items": items, "total": len(items)})


@v21_router.post("/procurement/requests/batch-validate", response_model=ApiResponse)
def post_batch_validate(
    payload: BatchValidateInput, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(batch_validate_requests(session, payload.pr_nos))


@v21_router.post("/procurement/requests/batch-export", response_model=ApiResponse)
def post_batch_export(
    payload: BatchExportInput, session: Session = Depends(get_session)
) -> ApiResponse:
    data = batch_export_requests(
        session,
        pr_nos=payload.pr_nos,
        template_version=payload.template_version,
        filters=payload.filters,
    )
    return response(data, business_key=data["export_task_id"])


@v21_router.patch(
    "/procurement/requests/{pr_no}/purchase-method", response_model=ApiResponse
)
def patch_pr_purchase_method(
    pr_no: str,
    payload: PurchaseMethodPatchInput,
    session: Session = Depends(get_session),
) -> ApiResponse:
    data = patch_purchase_method(
        session,
        pr_no,
        purchase_method_confirmed=payload.purchase_method_confirmed,
        task_id=payload.task_id,
    )
    return response(data, payload.task_id, pr_no)


@v21_router.post("/agent/tasks/{task_id}/pause", response_model=ApiResponse)
def post_pause_agent_task(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(pause_agent_task(session, task_id), task_id=task_id)


@v21_router.post("/agent/tasks/{task_id}/resume", response_model=ApiResponse)
def post_resume_agent_task(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(resume_agent_task(session, task_id), task_id=task_id)


@v21_router.post("/agent/tasks/{task_id}/stop", response_model=ApiResponse)
def post_stop_agent_task(
    task_id: str, session: Session = Depends(get_session)
) -> ApiResponse:
    return response(stop_agent_task(session, task_id), task_id=task_id)


@v21_router.get("/exports/{filename}")
def download_export(filename: str) -> FileResponse:
    path = resolve_export_file(filename)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=Path(path).name,
    )


@router.post("/demo/reset", response_model=ApiResponse)
def reset_demo(request: Request) -> ApiResponse:
    reset_database(request.app.state.database)
    return response({"reset": True})


@router.get("/demo/state", response_model=ApiResponse)
def demo_state(session: Session = Depends(get_session)) -> ApiResponse:
    counts = {
        "oa_applications": session.scalar(select(func.count(OAApplication.id))) or 0,
        "erp_materials": session.scalar(select(func.count(ERPMaterial.id))) or 0,
        "procurement_requests": session.scalar(
            select(func.count(ProcurementRequest.id))
        )
        or 0,
        "differences": session.scalar(
            select(func.count(CrossSystemDifference.id))
        )
        or 0,
        "agent_tasks": session.scalar(select(func.count(AgentTask.id))) or 0,
        "audit_events": session.scalar(select(func.count(AuditEvent.id))) or 0,
        "erp_purchase_orders": session.scalar(select(func.count(ERPPurchaseOrder.id)))
        or 0,
        "integration_transfers": session.scalar(
            select(func.count(IntegrationTransfer.id))
        )
        or 0,
        "business_lineages": session.scalar(select(func.count(BusinessLineage.id)))
        or 0,
    }
    oa_statuses = dict(
        session.execute(
            select(OAApplication.status, func.count(OAApplication.id)).group_by(
                OAApplication.status
            )
        ).all()
    )
    return response({"counts": counts, "oa_statuses": oa_statuses})
