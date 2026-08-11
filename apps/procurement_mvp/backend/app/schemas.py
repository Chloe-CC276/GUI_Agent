from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class OALineOut(ORMModel):
    id: int
    item_name: str
    specification: str | None
    quantity: Decimal
    estimated_unit_price: Decimal


class OALineInput(BaseModel):
    item_name: str
    specification: str | None = None
    quantity: Decimal = Field(gt=0)
    estimated_unit_price: Decimal = Field(ge=0)


class OAAttachmentOut(ORMModel):
    id: int
    file_name: str
    file_url: str
    source_attachment_id: str | None
    size: int | None = None
    mime_type: str | None = None


class OAApprovalHistoryOut(ORMModel):
    id: int
    oa_apply_no: str
    oa_version: int
    action: str
    from_status: str | None
    to_status: str | None
    operator_id: str | None
    operator_name: str | None
    opinion: str | None
    snapshot_json: dict[str, Any] | None = None
    created_at: datetime


class OAApplicationOut(ORMModel):
    id: int
    application_no: str
    title: str
    applicant: str
    department: str
    status: str
    total_budget: Decimal
    created_at: datetime
    lines: list[OALineOut] = []
    attachments: list[OAAttachmentOut] = []
    oa_version: int = 1
    procurement_transfer_status: str | None = None
    linked_pr_no: str | None = None
    linked_po_no: str | None = None
    erp_status: str | None = None
    procurement_status: str = "NOT_STARTED"
    procurement_updated_at: datetime | None = None
    is_submitted: bool = False
    submitted_at: datetime | None = None
    approval_started_at: datetime | None = None
    approved_time: datetime | None = None
    current_approver_id: str | None = None
    current_approver_name: str | None = None
    approved_by: str | None = None
    approval_opinion: str | None = None
    budget_project_code: str | None = None
    budget_project_name: str | None = None
    cost_center_code: str | None = None
    purchase_reason: str | None = None
    requested_method: str | None = None
    urgency_level: str = "NORMAL"
    expected_completion_date: date | None = None
    remark: str | None = None
    row_version: int = 1
    updated_at: datetime | None = None
    approval_history: list[OAApprovalHistoryOut] = []

    @computed_field
    @property
    def oa_apply_no(self) -> str:
        return self.application_no

    @computed_field
    @property
    def approval_status(self) -> str:
        return self.status

    @computed_field
    @property
    def proposal_title(self) -> str:
        return self.title

    @computed_field
    @property
    def proposal_amount(self) -> Decimal:
        return self.total_budget


class OAApplicationCreate(BaseModel):
    title: str | None = None
    proposal_title: str | None = None
    applicant: str | None = None
    department: str | None = None
    total_budget: Decimal | None = Field(default=None, ge=0)
    proposal_amount: Decimal | None = Field(default=None, ge=0)
    purchase_reason: str | None = None
    urgency_level: str | None = "NORMAL"
    budget_project_code: str | None = None
    budget_project_name: str | None = None
    cost_center_code: str | None = None
    requested_method: str | None = None
    expected_completion_date: date | None = None
    remark: str | None = None
    lines: list[OALineInput] = []

    def resolved_title(self) -> str | None:
        value = self.title if self.title is not None else self.proposal_title
        return value

    def resolved_amount(self) -> Decimal | None:
        if self.total_budget is not None:
            return self.total_budget
        return self.proposal_amount


class OAApplicationUpdate(OAApplicationCreate):
    row_version: int | None = None
    lines: list[OALineInput] | None = None  # type: ignore[assignment]


class OASubmitInput(BaseModel):
    row_version: int
    operator_id: str | None = None
    operator_name: str | None = None


class OAApprovalActionInput(BaseModel):
    row_version: int
    operator_id: str | None = None
    operator_name: str | None = None
    opinion: str | None = None
    current_approver_id: str | None = None
    current_approver_name: str | None = None


class OARejectInput(BaseModel):
    row_version: int
    reason: str = Field(min_length=1)
    operator_id: str | None = None
    operator_name: str | None = None


class ERPMaterialOut(ORMModel):
    material_code: str
    material_name: str
    specification: str
    unit: str
    standard_price: Decimal
    status: str


class RequestLineInput(BaseModel):
    oa_line_id: int | None = None
    material_code: str
    quantity: Decimal = Field(gt=0)
    material_name: str | None = None
    specification: str | None = None
    unit: str | None = None
    unit_price: Decimal | None = Field(default=None, ge=0)
    line_amount: Decimal | None = Field(default=None, ge=0)


class AttachmentInput(BaseModel):
    file_name: str
    file_url: str


class CreateRequestInput(BaseModel):
    task_id: str = Field(min_length=1)
    business_key: str = Field(min_length=1)
    oa_application_id: int
    lines: list[RequestLineInput] = Field(min_length=1)
    attachments: list[AttachmentInput] = []


class UpdateDraftInput(BaseModel):
    task_id: str = Field(min_length=1)
    business_key: str = Field(min_length=1)
    lines: list[RequestLineInput] = Field(min_length=1)
    attachments: list[AttachmentInput] | None = None


class OperationInput(BaseModel):
    task_id: str = Field(min_length=1)
    business_key: str = Field(min_length=1)


class ConfirmInput(OperationInput):
    confirmation_token: str = Field(min_length=1)


class RequestLineOut(ORMModel):
    id: int
    oa_line_id: int | None
    material_code: str
    material_name: str
    specification: str
    unit: str
    quantity: Decimal
    unit_price: Decimal
    line_amount: Decimal
    raw_material_name: str | None = None
    raw_specification: str | None = None
    raw_unit: str | None = None
    raw_quantity: Decimal | None = None
    raw_estimated_unit_price: Decimal | None = None
    import_batch_id: str | None = None
    match_confidence: Decimal | None = None

    @computed_field
    @property
    def final_unit_price_tax(self) -> Decimal:
        return self.unit_price


class AttachmentOut(ORMModel):
    id: int
    file_name: str
    file_url: str


class ProcurementRequestOut(ORMModel):
    id: int
    request_no: str
    oa_application_id: int
    status: str
    total_amount: Decimal
    submitted_at: datetime | None
    created_at: datetime
    updated_at: datetime
    lines: list[RequestLineOut]
    attachments: list[AttachmentOut]
    oa_apply_no: str | None = None
    oa_title: str | None = None
    oa_applicant: str | None = None
    oa_department: str | None = None
    oa_total_budget: Decimal | None = None
    oa_version: int = 1
    po_no: str | None = None
    erp_status: str | None = None
    submission_version: int = 1
    procurement_transfer_status: str | None = None
    purchase_method_suggested: str | None = None
    purchase_method_confirmed: str | None = None
    rule_version: str | None = None
    export_status: str | None = None
    budget_project: str | None = None
    cost_center: str | None = None
    purchase_type: str | None = None
    expected_delivery_date: date | None = None
    receive_address: str | None = None
    purchase_reason: str | None = None
    supplier_code: str | None = None
    supplier_name: str | None = None
    award_source: str | None = None
    final_total_amount_tax: Decimal | None = None
    award_confirmed_by: str | None = None
    award_confirmed_at: datetime | None = None
    erp_sync_status: str | None = "NOT_SENT"

    @computed_field
    @property
    def pr_no(self) -> str:
        return self.request_no

    @computed_field
    @property
    def purchase_method(self) -> str | None:
        return self.purchase_method_confirmed or self.purchase_type


class AwardSourceOut(ORMModel):
    code: str
    name: str
    status: str = "active"


class ERPSupplierOut(ORMModel):
    supplier_code: str
    supplier_name: str
    status: str
    award_sources: list[str] = Field(default_factory=list)


class ProcurementPatchInput(BaseModel):
    budget_project: str | None = None
    cost_center: str | None = None
    purchase_type: str | None = None
    purchase_method: str | None = None
    purchase_method_confirmed: str | None = None
    expected_delivery_date: date | None = None
    receive_address: str | None = None
    purchase_reason: str | None = None
    supplier_code: str | None = None
    supplier_name: str | None = None
    award_source: str | None = None
    award_confirmed_by: str | None = None
    lines: list[dict[str, Any]] | None = None


class SubmitProcurementInput(BaseModel):
    task_id: str = Field(min_length=1)
    simulate_target_failure: bool = False
    simulate_callback_failure: bool = False


class SubmitErpInput(BaseModel):
    task_id: str = Field(min_length=1)
    business_key: str = Field(min_length=1)
    confirmed_by: str | None = None
    simulate_target_failure: bool = False
    simulate_callback_failure: bool = False


class ImportConfirmInput(BaseModel):
    task_id: str = Field(min_length=1)
    business_key: str = Field(min_length=1)
    import_batch_id: str = Field(min_length=1)
    pr_no: str | None = None
    oa_application_id: int | None = None
    row_nos: list[int] | None = None


class BatchValidateInput(BaseModel):
    pr_nos: list[str] = Field(min_length=1)


class BatchExportInput(BaseModel):
    pr_nos: list[str] = Field(min_length=1)
    template_version: str = "V3.2"
    filters: dict[str, Any] = Field(default_factory=dict)


class PurchaseMethodPatchInput(BaseModel):
    purchase_method_confirmed: str = Field(min_length=1)
    task_id: str | None = None


class FaultInjectionInput(BaseModel):
    task_id: str = Field(min_length=1)
    simulate_target_failure: bool = False
    simulate_callback_failure: bool = False


class ERPSubmitInput(BaseModel):
    task_id: str = Field(min_length=1)
    business_key: str = Field(min_length=1)
    confirmation_token: str = Field(min_length=1)
    simulate_target_failure: bool = False
    simulate_callback_failure: bool = False


class RetryInput(BaseModel):
    task_id: str | None = None


class PurchaseOrderLineOut(ORMModel):
    id: int
    line_no: int
    material_code: str
    material_name: str
    specification: str
    unit: str
    quantity: Decimal
    unit_price: Decimal
    line_amount: Decimal
    po_item_no: int | None = None
    uom: str | None = None
    unit_price_tax: Decimal | None = None
    tax_rate: Decimal | None = None
    line_amount_tax: Decimal | None = None
    delivery_date: date | None = None


class PurchaseOrderOut(ORMModel):
    po_no: str
    pr_no: str
    oa_apply_no: str | None
    submission_version: int
    task_id: str
    status: str
    total_amount: Decimal
    created_at: datetime
    updated_at: datetime
    lines: list[PurchaseOrderLineOut]
    supplier_code: str | None = None
    supplier_name: str | None = None
    request_dept: str | None = None
    purchasing_org: str | None = None
    purchasing_group: str | None = None
    currency_code: str | None = "CNY"
    payment_terms: str | None = None
    buyer_id: str | None = None
    total_amount_tax: Decimal | None = None
    created_by_agent_task_id: str | None = None
    batch_id: str | None = None


class POBatchCreateInput(BaseModel):
    pr_nos: list[str] = Field(min_length=1)
    operator: str | None = None


class POCreateFormInput(BaseModel):
    header: dict[str, Any] = Field(default_factory=dict)
    lines: list[dict[str, Any]] = Field(default_factory=list)
    simulate_readback_fail: bool = False


class POMarkCreatedInput(BaseModel):
    po_no: str | None = None
    success: bool = True
    error_code: str | None = None
    message: str | None = None


class TransferOut(ORMModel):
    transfer_id: str
    source_system: str
    source_key: str
    target_system: str
    target_key: str | None
    transfer_type: str
    status: str
    phase: str
    idempotency_key: str
    retry_count: int
    error_code: str | None
    error_message: str | None
    task_id: str
    payload: dict
    result: dict
    created_at: datetime
    updated_at: datetime


class DifferenceOut(ORMModel):
    id: int
    request_id: int
    request_line_id: int | None
    field_name: str
    source_system: str
    provided_value: str | None
    authoritative_value: str | None
    created_at: datetime


class TaskOut(ORMModel):
    task_id: str
    business_key: str
    operation: str
    status: str
    result: dict[str, Any]
    created_at: datetime
    current_route: str | None = None
    context_json: dict[str, Any] | None = None
    is_paused: bool = False


class AgentChatInput(BaseModel):
    message: str = Field(min_length=1)
    route: str | None = None
    business_key: str | None = None
    folder_path: str | None = None
    excel_path: str | None = None


class AgentContinueInput(BaseModel):
    folder_path: str | None = None
    excel_path: str | None = None
    oa_id: int | None = None
    note: str | None = None


class AgentStepResultInput(BaseModel):
    step_id: str = Field(min_length=1)
    status: Literal["passed", "failed"]
    actual: Any = None
    detail: dict[str, Any] | None = None


class ApiResponse(BaseModel):
    ok: bool = True
    data: Any = None
    task_id: str | None = None
    business_key: str | None = None
    idempotent_replay: bool = False


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    ok: Literal[False] = False
    error: ErrorBody
