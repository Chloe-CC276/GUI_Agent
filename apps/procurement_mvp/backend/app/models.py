from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


erp_supplier_award_sources = Table(
    "erp_supplier_award_sources",
    Base.metadata,
    Column("supplier_id", ForeignKey("erp_suppliers.id", ondelete="CASCADE"), primary_key=True),
    Column("award_source_id", ForeignKey("award_sources.id", ondelete="CASCADE"), primary_key=True),
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OAApplication(Base):
    __tablename__ = "oa_applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_no: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    applicant: Mapped[str] = mapped_column(String(80))
    department: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), index=True)
    total_budget: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    oa_version: Mapped[int] = mapped_column(Integer, default=1)
    procurement_transfer_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    linked_pr_no: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    linked_po_no: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    erp_status: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    procurement_status: Mapped[str] = mapped_column(
        String(20), default="NOT_STARTED", index=True
    )
    procurement_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_submitted: Mapped[bool] = mapped_column(Boolean, default=False)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approval_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approved_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    current_approver_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    current_approver_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    approval_opinion: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget_project_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    budget_project_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    cost_center_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    purchase_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_method: Mapped[str | None] = mapped_column(String(40), nullable=True)
    urgency_level: Mapped[str] = mapped_column(String(20), default="NORMAL")
    expected_completion_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    remark: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=True
    )
    lines: Mapped[list["OAApplicationLine"]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )
    attachments: Mapped[list["OAAttachmentReference"]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )


class OAApplicationLine(Base):
    __tablename__ = "oa_application_lines"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("oa_applications.id"))
    item_name: Mapped[str] = mapped_column(String(200))
    specification: Mapped[str | None] = mapped_column(String(300), nullable=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    estimated_unit_price: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    application: Mapped["OAApplication"] = relationship(back_populates="lines")


class OAAttachmentReference(Base):
    __tablename__ = "oa_attachment_references"

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("oa_applications.id"), index=True)
    file_name: Mapped[str] = mapped_column(String(255))
    file_url: Mapped[str] = mapped_column(String(500))
    source_attachment_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    application: Mapped["OAApplication"] = relationship(back_populates="attachments")


class OAApprovalHistory(Base):
    __tablename__ = "oa_approval_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    oa_apply_no: Mapped[str] = mapped_column(String(40), index=True)
    oa_version: Mapped[int] = mapped_column(Integer, default=1)
    action: Mapped[str] = mapped_column(String(20), index=True)
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    operator_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    operator_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    opinion: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OAOutbox(Base):
    __tablename__ = "oa_outbox"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ERPMaterial(Base):
    __tablename__ = "erp_materials"

    id: Mapped[int] = mapped_column(primary_key=True)
    material_code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    material_name: Mapped[str] = mapped_column(String(200), index=True)
    specification: Mapped[str] = mapped_column(String(300), default="")
    unit: Mapped[str] = mapped_column(String(30))
    standard_price: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)


class AwardSource(Base):
    __tablename__ = "award_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    suppliers: Mapped[list["ERPSupplier"]] = relationship(
        secondary=erp_supplier_award_sources,
        back_populates="award_sources",
    )


class ERPSupplier(Base):
    __tablename__ = "erp_suppliers"

    id: Mapped[int] = mapped_column(primary_key=True)
    supplier_code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    supplier_name: Mapped[str] = mapped_column(String(200), index=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    award_sources: Mapped[list[AwardSource]] = relationship(
        secondary=erp_supplier_award_sources,
        back_populates="suppliers",
    )


class ProcurementRequest(Base):
    __tablename__ = "procurement_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_no: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    oa_application_id: Mapped[int] = mapped_column(ForeignKey("oa_applications.id"))
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0"))
    oa_apply_no: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    oa_title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    oa_applicant: Mapped[str | None] = mapped_column(String(80), nullable=True)
    oa_department: Mapped[str | None] = mapped_column(String(100), nullable=True)
    oa_total_budget: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    oa_version: Mapped[int] = mapped_column(Integer, default=1)
    po_no: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    erp_status: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    submission_version: Mapped[int] = mapped_column(Integer, default=1)
    procurement_transfer_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    confirmation_token: Mapped[str | None] = mapped_column(String(120), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    purchase_method_suggested: Mapped[str | None] = mapped_column(String(40), nullable=True)
    purchase_method_confirmed: Mapped[str | None] = mapped_column(String(40), nullable=True)
    rule_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    export_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    budget_project: Mapped[str | None] = mapped_column(String(120), nullable=True)
    cost_center: Mapped[str | None] = mapped_column(String(120), nullable=True)
    purchase_type: Mapped[str | None] = mapped_column(String(60), nullable=True)
    expected_delivery_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    receive_address: Mapped[str | None] = mapped_column(String(300), nullable=True)
    purchase_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    supplier_code: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    supplier_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    award_source: Mapped[str | None] = mapped_column(String(40), nullable=True)
    final_total_amount_tax: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True)
    award_confirmed_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    award_confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    erp_sync_status: Mapped[str] = mapped_column(
        String(30), default="NOT_SENT", index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    oa_application: Mapped["OAApplication"] = relationship()
    lines: Mapped[list["ProcurementRequestLine"]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )
    attachments: Mapped[list["ProcurementAttachment"]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )


class ProcurementRequestLine(Base):
    __tablename__ = "procurement_request_lines"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("procurement_requests.id"))
    oa_line_id: Mapped[int | None] = mapped_column(
        ForeignKey("oa_application_lines.id"), nullable=True
    )
    material_code: Mapped[str] = mapped_column(String(50), index=True)
    material_name: Mapped[str] = mapped_column(String(200))
    specification: Mapped[str] = mapped_column(String(300), default="")
    unit: Mapped[str] = mapped_column(String(30))
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    unit_price: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    line_amount: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    raw_material_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    raw_specification: Mapped[str | None] = mapped_column(String(300), nullable=True)
    raw_unit: Mapped[str | None] = mapped_column(String(30), nullable=True)
    raw_quantity: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    raw_estimated_unit_price: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 2), nullable=True
    )
    import_batch_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    match_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    request: Mapped["ProcurementRequest"] = relationship(back_populates="lines")


class ProcurementAttachment(Base):
    __tablename__ = "procurement_attachments"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("procurement_requests.id"))
    file_name: Mapped[str] = mapped_column(String(255))
    file_url: Mapped[str] = mapped_column(String(500))
    request: Mapped["ProcurementRequest"] = relationship(back_populates="attachments")


class CrossSystemDifference(Base):
    __tablename__ = "cross_system_differences"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(
        ForeignKey("procurement_requests.id"), index=True
    )
    request_line_id: Mapped[int | None] = mapped_column(
        ForeignKey("procurement_request_lines.id"), nullable=True
    )
    field_name: Mapped[str] = mapped_column(String(80))
    source_system: Mapped[str] = mapped_column(String(30))
    provided_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    authoritative_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AgentTask(Base):
    __tablename__ = "agent_tasks"
    __table_args__ = (
        UniqueConstraint(
            "task_id", "business_key", "operation", name="uq_agent_task_idempotency"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(String(80), index=True)
    business_key: Mapped[str] = mapped_column(String(100), index=True)
    operation: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="completed")
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    current_route: Mapped[str | None] = mapped_column(String(200), nullable=True)
    context_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_paused: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WorkflowEvent(Base):
    __tablename__ = "workflow_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    business_key: Mapped[str] = mapped_column(String(100), index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(30), default="info")
    operator: Mapped[str | None] = mapped_column(String(80), nullable=True)
    detail_json: Mapped[dict] = mapped_column(JSON, default=dict)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ImportBatch(Base):
    __tablename__ = "import_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    import_batch_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    sheet_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    success_rows: Mapped[int] = mapped_column(Integer, default=0)
    failed_rows: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="review", index=True)
    preview_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExportBatch(Base):
    __tablename__ = "export_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    export_task_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    filters_json: Mapped[dict] = mapped_column(JSON, default=dict)
    selected_prs: Mapped[list] = mapped_column(JSON, default=list)
    template_version: Mapped[str] = mapped_column(String(40), default="V3.2")
    rule_version: Mapped[str] = mapped_column(String(40), default="v1")
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="generated", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    business_key: Mapped[str] = mapped_column(String(100), index=True)
    task_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ERPPurchaseOrder(Base):
    __tablename__ = "erp_purchase_orders"
    __table_args__ = (
        UniqueConstraint("pr_no", "submission_version", name="uq_po_pr_submission"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    po_no: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    pr_no: Mapped[str] = mapped_column(String(40), index=True)
    oa_apply_no: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    submission_version: Mapped[int] = mapped_column(Integer, default=1)
    task_id: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(30), default="created", index=True)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    lines: Mapped[list["ERPPurchaseOrderLine"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class ERPPurchaseOrderLine(Base):
    __tablename__ = "erp_purchase_order_lines"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("erp_purchase_orders.id"), index=True)
    line_no: Mapped[int] = mapped_column(Integer)
    material_code: Mapped[str] = mapped_column(String(50), index=True)
    material_name: Mapped[str] = mapped_column(String(200))
    specification: Mapped[str] = mapped_column(String(300), default="")
    unit: Mapped[str] = mapped_column(String(30))
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 4))
    unit_price: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    line_amount: Mapped[Decimal] = mapped_column(Numeric(18, 2))
    order: Mapped["ERPPurchaseOrder"] = relationship(back_populates="lines")


class IntegrationTransfer(Base):
    __tablename__ = "integration_transfers"

    id: Mapped[int] = mapped_column(primary_key=True)
    transfer_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    source_system: Mapped[str] = mapped_column(String(30))
    source_key: Mapped[str] = mapped_column(String(100), index=True)
    target_system: Mapped[str] = mapped_column(String(30))
    target_key: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    transfer_type: Mapped[str] = mapped_column(String(30), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    phase: Mapped[str] = mapped_column(String(40))
    idempotency_key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_id: Mapped[str] = mapped_column(String(80), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class BusinessLineage(Base):
    __tablename__ = "business_lineages"

    id: Mapped[int] = mapped_column(primary_key=True)
    oa_apply_no: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    pr_no: Mapped[str | None] = mapped_column(String(40), unique=True, nullable=True, index=True)
    po_no: Mapped[str | None] = mapped_column(String(40), unique=True, nullable=True, index=True)
    task_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    latest_status: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class SchemaVersion(Base):
    __tablename__ = "schema_versions"

    version: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str] = mapped_column(String(200))
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
