from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Engine, inspect, text

S0_VERSION = 1
S0_BACKFILL_VERSION = 2
V21_VERSION = 3
OA_CLOSURE_VERSION = 4
PROCUREMENT_CLOUD_VERSION = 5
SUPPLIER_AWARD_SOURCE_VERSION = 6
ERP_PO_AGENT_VERSION = 7

DEFAULT_AWARD_SOURCES = (
    ("offline_inquiry", "线下询比价"),
    ("framework", "框架协议"),
    ("mall", "商城"),
    ("direct", "直接采购"),
    ("other", "其他"),
)

# supplier_code -> award_source codes (many-to-many seed links)
DEFAULT_SUPPLIER_AWARD_LINKS = {
    "SUP-001": ("offline_inquiry", "framework", "direct"),
    "SUP-002": ("offline_inquiry", "mall", "other"),
    "SUP-003": ("framework", "direct", "other"),
    "SUP-004": ("mall", "direct"),
}

OA_COLUMNS = {
    "oa_version": "INTEGER NOT NULL DEFAULT 1",
    "procurement_transfer_status": "VARCHAR(30)",
    "linked_pr_no": "VARCHAR(40)",
    "linked_po_no": "VARCHAR(40)",
    "erp_status": "VARCHAR(30)",
}

PR_COLUMNS = {
    "oa_apply_no": "VARCHAR(40)",
    "oa_title": "VARCHAR(200)",
    "oa_applicant": "VARCHAR(80)",
    "oa_department": "VARCHAR(100)",
    "oa_total_budget": "NUMERIC(18, 2)",
    "oa_version": "INTEGER NOT NULL DEFAULT 1",
    "po_no": "VARCHAR(40)",
    "erp_status": "VARCHAR(30)",
    "submission_version": "INTEGER NOT NULL DEFAULT 1",
    "procurement_transfer_status": "VARCHAR(30)",
}

PR_V21_COLUMNS = {
    "purchase_method_suggested": "VARCHAR(40)",
    "purchase_method_confirmed": "VARCHAR(40)",
    "rule_version": "VARCHAR(40)",
    "export_status": "VARCHAR(30)",
    "budget_project": "VARCHAR(120)",
    "cost_center": "VARCHAR(120)",
    "purchase_type": "VARCHAR(60)",
    "expected_delivery_date": "DATE",
    "receive_address": "VARCHAR(300)",
    "purchase_reason": "TEXT",
}

PR_LINE_COLUMNS = {
    "raw_material_name": "VARCHAR(200)",
    "raw_specification": "VARCHAR(300)",
    "raw_unit": "VARCHAR(30)",
    "raw_quantity": "NUMERIC(18, 4)",
    "raw_estimated_unit_price": "NUMERIC(18, 2)",
}

PR_LINE_V21_COLUMNS = {
    "import_batch_id": "VARCHAR(80)",
    "match_confidence": "NUMERIC(5, 4)",
}

AGENT_TASK_V21_COLUMNS = {
    "current_route": "VARCHAR(200)",
    "context_json": "JSON",
    "is_paused": "BOOLEAN NOT NULL DEFAULT 0",
}

OA_CLOSURE_COLUMNS = {
    "is_submitted": "BOOLEAN NOT NULL DEFAULT 0",
    "submitted_at": "DATETIME",
    "approval_started_at": "DATETIME",
    "approved_time": "DATETIME",
    "current_approver_id": "VARCHAR(80)",
    "current_approver_name": "VARCHAR(80)",
    "approved_by": "VARCHAR(80)",
    "approval_opinion": "TEXT",
    "budget_project_code": "VARCHAR(80)",
    "budget_project_name": "VARCHAR(200)",
    "cost_center_code": "VARCHAR(80)",
    "purchase_reason": "TEXT",
    "requested_method": "VARCHAR(40)",
    "urgency_level": "VARCHAR(20) DEFAULT 'NORMAL'",
    "expected_completion_date": "DATE",
    "remark": "TEXT",
    "row_version": "INTEGER NOT NULL DEFAULT 1",
    "updated_at": "DATETIME",
}

OA_ATTACHMENT_CLOSURE_COLUMNS = {
    "size": "INTEGER",
    "mime_type": "VARCHAR(120)",
}

OA_PROCUREMENT_STATUS_COLUMNS = {
    "procurement_status": "VARCHAR(20) NOT NULL DEFAULT 'NOT_STARTED'",
    "procurement_updated_at": "DATETIME",
}

PR_AWARD_COLUMNS = {
    "supplier_code": "VARCHAR(50)",
    "supplier_name": "VARCHAR(200)",
    "award_source": "VARCHAR(40)",
    "final_total_amount_tax": "NUMERIC(18, 2)",
    "award_confirmed_by": "VARCHAR(80)",
    "award_confirmed_at": "DATETIME",
    "erp_sync_status": "VARCHAR(30) NOT NULL DEFAULT 'NOT_SENT'",
}


def _add_columns(connection, table: str, definitions: dict[str, str]) -> None:
    existing = {column["name"] for column in inspect(connection).get_columns(table)}
    for name, definition in definitions.items():
        if name not in existing:
            connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}'))


def migrate_s0(engine: Engine) -> None:
    """Apply the additive S0 migration. Safe to call at every startup."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_versions ("
                "version INTEGER PRIMARY KEY, description VARCHAR(200) NOT NULL, "
                "applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        tables = set(inspect(connection).get_table_names())
        if "oa_applications" in tables:
            _add_columns(connection, "oa_applications", OA_COLUMNS)
        if "procurement_requests" in tables:
            _add_columns(connection, "procurement_requests", PR_COLUMNS)
        if "procurement_request_lines" in tables:
            _add_columns(connection, "procurement_request_lines", PR_LINE_COLUMNS)

        if "procurement_requests" in tables and "oa_applications" in tables:
            connection.execute(
                text(
                    "UPDATE procurement_requests SET "
                    "oa_apply_no=COALESCE(oa_apply_no, (SELECT application_no FROM oa_applications "
                    "WHERE id=procurement_requests.oa_application_id)), "
                    "oa_title=COALESCE(oa_title, (SELECT title FROM oa_applications "
                    "WHERE id=procurement_requests.oa_application_id)), "
                    "oa_applicant=COALESCE(oa_applicant, (SELECT applicant FROM oa_applications "
                    "WHERE id=procurement_requests.oa_application_id)), "
                    "oa_department=COALESCE(oa_department, (SELECT department FROM oa_applications "
                    "WHERE id=procurement_requests.oa_application_id)), "
                    "oa_total_budget=COALESCE(oa_total_budget, (SELECT total_budget FROM oa_applications "
                    "WHERE id=procurement_requests.oa_application_id)), "
                    "oa_version=COALESCE(oa_version, 1), submission_version=COALESCE(submission_version, 1)"
                )
            )
        if "procurement_request_lines" in tables:
            connection.execute(
                text(
                    "UPDATE procurement_request_lines SET "
                    "raw_material_name=COALESCE(raw_material_name, material_name), "
                    "raw_specification=COALESCE(raw_specification, specification), "
                    "raw_unit=COALESCE(raw_unit, unit), "
                    "raw_quantity=COALESCE(raw_quantity, quantity), "
                    "raw_estimated_unit_price=COALESCE(raw_estimated_unit_price, unit_price)"
                )
            )
        connection.execute(
            text(
                "INSERT OR IGNORE INTO schema_versions(version, description) "
                "VALUES (:version, 'S0 procurement integration schema')"
            ),
            {"version": S0_VERSION},
        )


def migrate_v21(engine: Engine) -> None:
    """Apply additive v2.1 columns. Idempotent; never drops data."""
    with engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        if "procurement_requests" in tables:
            _add_columns(connection, "procurement_requests", PR_V21_COLUMNS)
        if "procurement_request_lines" in tables:
            _add_columns(connection, "procurement_request_lines", PR_LINE_V21_COLUMNS)
        if "agent_tasks" in tables:
            _add_columns(connection, "agent_tasks", AGENT_TASK_V21_COLUMNS)
        connection.execute(
            text(
                "INSERT OR IGNORE INTO schema_versions(version, description) "
                "VALUES (:version, 'v2.1 workbench import export agent columns')"
            ),
            {"version": V21_VERSION},
        )


def migrate_oa_closure(engine: Engine) -> None:
    """Apply OA procurement-application closed-loop columns. Idempotent; never drops."""
    with engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        if "oa_applications" in tables:
            _add_columns(connection, "oa_applications", OA_CLOSURE_COLUMNS)
            connection.execute(
                text("UPDATE oa_applications SET status='DRAFT' WHERE status='draft'")
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET status='IN_APPROVAL' "
                    "WHERE status IN ('pending', 'approving')"
                )
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET status='APPROVED' WHERE status='approved'"
                )
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET status='REJECTED' WHERE status='rejected'"
                )
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET is_submitted=1 "
                    "WHERE status IN ('IN_APPROVAL', 'APPROVED')"
                )
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET row_version=1 "
                    "WHERE row_version IS NULL OR row_version < 1"
                )
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET urgency_level='NORMAL' "
                    "WHERE urgency_level IS NULL OR urgency_level=''"
                )
            )
            # Submitted drafts become 待审批; keep pure drafts as DRAFT.
            connection.execute(
                text(
                    "UPDATE oa_applications SET status='PENDING_APPROVAL' "
                    "WHERE status IN ('DRAFT', 'draft') AND is_submitted=1"
                )
            )
            # Procurement lifecycle must NOT overwrite approval status.
            # See migrate_procurement_cloud for procurement_status backfill.
        if "oa_attachment_references" in tables:
            _add_columns(
                connection, "oa_attachment_references", OA_ATTACHMENT_CLOSURE_COLUMNS
            )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS oa_approval_history ("
                "id INTEGER PRIMARY KEY, "
                "oa_apply_no VARCHAR(40) NOT NULL, "
                "oa_version INTEGER NOT NULL DEFAULT 1, "
                "action VARCHAR(20) NOT NULL, "
                "from_status VARCHAR(20), "
                "to_status VARCHAR(20), "
                "operator_id VARCHAR(80), "
                "operator_name VARCHAR(80), "
                "opinion TEXT, "
                "snapshot_json JSON, "
                "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS oa_outbox ("
                "id INTEGER PRIMARY KEY, "
                "event_id VARCHAR(80) NOT NULL UNIQUE, "
                "event_type VARCHAR(80) NOT NULL, "
                "payload JSON NOT NULL, "
                "status VARCHAR(20) NOT NULL DEFAULT 'pending', "
                "retry_count INTEGER NOT NULL DEFAULT 0, "
                "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "sent_at DATETIME)"
            )
        )
        connection.execute(
            text(
                "INSERT OR IGNORE INTO schema_versions(version, description) "
                "VALUES (:version, 'OA application closed-loop schema')"
            ),
            {"version": OA_CLOSURE_VERSION},
        )


def migrate_procurement_cloud(engine: Engine) -> None:
    """Split OA approval vs procurement execution; add award/supplier fields."""
    now = datetime.now(timezone.utc).isoformat()
    with engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        if "oa_applications" in tables:
            _add_columns(connection, "oa_applications", OA_PROCUREMENT_STATUS_COLUMNS)
            connection.execute(
                text(
                    "UPDATE oa_applications SET procurement_status='NOT_STARTED' "
                    "WHERE procurement_status IS NULL OR procurement_status=''"
                )
            )
            # Legacy mixed statuses → restore APPROVED + procurement_status.
            connection.execute(
                text(
                    "UPDATE oa_applications SET "
                    "procurement_status='AWARDED', "
                    "status='APPROVED', "
                    f"procurement_updated_at=COALESCE(procurement_updated_at, '{now}') "
                    "WHERE status IN ('AWARDED', 'ORDER_CREATED', 'awarded', 'order_created') "
                    "OR (linked_po_no IS NOT NULL AND linked_po_no!='')"
                )
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET "
                    "procurement_status='PREPARING', "
                    "status='APPROVED', "
                    f"procurement_updated_at=COALESCE(procurement_updated_at, '{now}') "
                    "WHERE status IN ('PROCUREMENT_PREP', 'procurement_prep') "
                    "OR ("
                    "  status IN ('APPROVED', 'approved') "
                    "  AND linked_pr_no IS NOT NULL AND linked_pr_no!='' "
                    "  AND (linked_po_no IS NULL OR linked_po_no='') "
                    "  AND COALESCE(procurement_status, 'NOT_STARTED')='NOT_STARTED'"
                    ")"
                )
            )
            # Any remaining mixed procurement labels on status.
            connection.execute(
                text(
                    "UPDATE oa_applications SET status='APPROVED' "
                    "WHERE status IN ("
                    "'PROCUREMENT_PREP','AWARDED','ORDER_CREATED',"
                    "'procurement_prep','awarded','order_created'"
                    ")"
                )
            )
        if "procurement_requests" in tables:
            _add_columns(connection, "procurement_requests", PR_AWARD_COLUMNS)
            connection.execute(
                text(
                    "UPDATE procurement_requests SET erp_sync_status='NOT_SENT' "
                    "WHERE erp_sync_status IS NULL OR erp_sync_status=''"
                )
            )
            connection.execute(
                text(
                    "UPDATE procurement_requests SET erp_sync_status='SUCCESS' "
                    "WHERE po_no IS NOT NULL AND po_no!=''"
                )
            )
            connection.execute(
                text(
                    "UPDATE procurement_requests SET erp_sync_status='FAILED' "
                    "WHERE erp_status='failed' "
                    "AND (po_no IS NULL OR po_no='')"
                )
            )
            connection.execute(
                text(
                    "UPDATE procurement_requests SET "
                    "final_total_amount_tax=COALESCE(final_total_amount_tax, total_amount)"
                )
            )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS erp_suppliers ("
                "id INTEGER PRIMARY KEY, "
                "supplier_code VARCHAR(50) NOT NULL UNIQUE, "
                "supplier_name VARCHAR(200) NOT NULL, "
                "status VARCHAR(20) NOT NULL DEFAULT 'active')"
            )
        )
        connection.execute(
            text(
                "INSERT OR IGNORE INTO schema_versions(version, description) "
                "VALUES (:version, 'Procurement cloud award and OA status split')"
            ),
            {"version": PROCUREMENT_CLOUD_VERSION},
        )


def migrate_supplier_award_sources(engine: Engine) -> None:
    """Award source master + supplier↔source many-to-many links."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS award_sources ("
                "id INTEGER PRIMARY KEY, "
                "code VARCHAR(40) NOT NULL UNIQUE, "
                "name VARCHAR(80) NOT NULL, "
                "status VARCHAR(20) NOT NULL DEFAULT 'active')"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS erp_supplier_award_sources ("
                "supplier_id INTEGER NOT NULL, "
                "award_source_id INTEGER NOT NULL, "
                "PRIMARY KEY (supplier_id, award_source_id), "
                "FOREIGN KEY(supplier_id) REFERENCES erp_suppliers(id) ON DELETE CASCADE, "
                "FOREIGN KEY(award_source_id) REFERENCES award_sources(id) ON DELETE CASCADE)"
            )
        )
        for code, name in DEFAULT_AWARD_SOURCES:
            connection.execute(
                text(
                    "INSERT OR IGNORE INTO award_sources(code, name, status) "
                    "VALUES (:code, :name, 'active')"
                ),
                {"code": code, "name": name},
            )
        tables = set(inspect(connection).get_table_names())
        if "erp_suppliers" in tables:
            for supplier_code, source_codes in DEFAULT_SUPPLIER_AWARD_LINKS.items():
                supplier = connection.execute(
                    text("SELECT id FROM erp_suppliers WHERE supplier_code=:code"),
                    {"code": supplier_code},
                ).mappings().first()
                if supplier is None:
                    continue
                for source_code in source_codes:
                    source = connection.execute(
                        text("SELECT id FROM award_sources WHERE code=:code"),
                        {"code": source_code},
                    ).mappings().first()
                    if source is None:
                        continue
                    connection.execute(
                        text(
                            "INSERT OR IGNORE INTO erp_supplier_award_sources"
                            "(supplier_id, award_source_id) VALUES (:sid, :aid)"
                        ),
                        {"sid": supplier["id"], "aid": source["id"]},
                    )
        connection.execute(
            text(
                "INSERT OR IGNORE INTO schema_versions(version, description) "
                "VALUES (:version, 'Supplier award-source many-to-many')"
            ),
            {"version": SUPPLIER_AWARD_SOURCE_VERSION},
        )


AGENT_TASK_PO_COLUMNS = {
    "batch_id": "VARCHAR(80)",
    "started_at": "DATETIME",
    "finished_at": "DATETIME",
    "retry_count": "INTEGER NOT NULL DEFAULT 0",
    "po_no": "VARCHAR(40)",
    "error_code": "VARCHAR(80)",
    "executor_type": "VARCHAR(40) DEFAULT 'dom'",
    "takeover_flag": "BOOLEAN NOT NULL DEFAULT 0",
}

ERP_PO_HEADER_COLUMNS = {
    "supplier_code": "VARCHAR(50)",
    "supplier_name": "VARCHAR(200)",
    "request_dept": "VARCHAR(100)",
    "purchasing_org": "VARCHAR(80)",
    "purchasing_group": "VARCHAR(80)",
    "currency_code": "VARCHAR(10) DEFAULT 'CNY'",
    "payment_terms": "VARCHAR(80)",
    "buyer_id": "VARCHAR(80)",
    "total_amount_tax": "NUMERIC(18, 2)",
    "created_by_agent_task_id": "VARCHAR(80)",
    "batch_id": "VARCHAR(80)",
}

ERP_PO_LINE_COLUMNS = {
    "tax_rate": "NUMERIC(8, 4)",
    "unit_price_tax": "NUMERIC(18, 2)",
    "line_amount_tax": "NUMERIC(18, 2)",
    "uom": "VARCHAR(30)",
    "delivery_date": "DATE",
    "po_item_no": "INTEGER",
}


def migrate_erp_po_agent(engine: Engine) -> None:
    """ERP PO creation via Agent (Scheme A): task/step/safety + PO header extras."""
    with engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        if "agent_tasks" in tables:
            _add_columns(connection, "agent_tasks", AGENT_TASK_PO_COLUMNS)
        if "erp_purchase_orders" in tables:
            _add_columns(connection, "erp_purchase_orders", ERP_PO_HEADER_COLUMNS)
        if "erp_purchase_order_lines" in tables:
            _add_columns(connection, "erp_purchase_order_lines", ERP_PO_LINE_COLUMNS)

        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS agent_step_logs ("
                "id INTEGER PRIMARY KEY, "
                "step_id VARCHAR(80) NOT NULL UNIQUE, "
                "task_id VARCHAR(80) NOT NULL, "
                "step_name VARCHAR(80) NOT NULL, "
                "expected_json JSON, "
                "actual_json JSON, "
                "status VARCHAR(30) NOT NULL DEFAULT 'pending', "
                "retry_count INTEGER NOT NULL DEFAULT 0, "
                "duration_ms INTEGER, "
                "screenshot_hash VARCHAR(128), "
                "error_code VARCHAR(80), "
                "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE IF NOT EXISTS agent_safety_logs ("
                "id INTEGER PRIMARY KEY, "
                "event_id VARCHAR(80) NOT NULL UNIQUE, "
                "task_id VARCHAR(80), "
                "batch_id VARCHAR(80), "
                "pr_no VARCHAR(40), "
                "po_no VARCHAR(40), "
                "stage VARCHAR(80), "
                "event_type VARCHAR(80) NOT NULL, "
                "severity VARCHAR(20) NOT NULL DEFAULT 'INFO', "
                "expected TEXT, "
                "actual TEXT, "
                "action_taken VARCHAR(80), "
                "retry_count INTEGER NOT NULL DEFAULT 0, "
                "created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT OR IGNORE INTO schema_versions(version, description) "
                "VALUES (:version, 'ERP PO Agent creation and dashboard schema')"
            ),
            {"version": ERP_PO_AGENT_VERSION},
        )


def backfill_s0(engine: Engine) -> None:
    """Adopt deterministic legacy PRs after all S0 tables have been created."""
    now = datetime.now(timezone.utc).isoformat()
    with engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        required = {
            "oa_applications",
            "procurement_requests",
            "integration_transfers",
            "business_lineages",
            "schema_versions",
        }
        if not required.issubset(tables):
            return

        connection.execute(
            text(
                "UPDATE procurement_requests SET status='ready', erp_status='pending' "
                "WHERE status='submitted' AND (po_no IS NULL OR po_no='')"
            )
        )
        rows = connection.execute(
            text(
                "SELECT id, request_no, oa_apply_no, COALESCE(oa_version, 1) AS oa_version "
                "FROM procurement_requests "
                "WHERE oa_apply_no IS NOT NULL AND oa_apply_no <> '' "
                "ORDER BY oa_apply_no, COALESCE(oa_version, 1), id"
            )
        ).mappings()
        adopted: dict[tuple[str, int], dict] = {}
        duplicate_counts: dict[tuple[str, int], int] = {}
        for row in rows:
            key = (str(row["oa_apply_no"]), int(row["oa_version"]))
            duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
            adopted.setdefault(key, dict(row))

        for (oa_apply_no, oa_version), request in adopted.items():
            request_no = str(request["request_no"])
            idempotency_key = f"OA_TO_PR:{oa_apply_no}:{oa_version}"
            result = {"pr_no": request_no, "migrated": True}
            if duplicate_counts[(oa_apply_no, oa_version)] > 1:
                result["anomaly"] = "multiple_legacy_prs"
                result["legacy_pr_count"] = duplicate_counts[(oa_apply_no, oa_version)]
                result["selection"] = "lowest_id"
            values = {
                "transfer_id": f"TR-MIG-S0-{request['id']}",
                "source_key": oa_apply_no,
                "target_key": request_no,
                "idempotency_key": idempotency_key,
                "payload": json.dumps(
                    {"oa_apply_no": oa_apply_no, "oa_version": oa_version},
                    ensure_ascii=False,
                ),
                "result": json.dumps(result, ensure_ascii=False),
                "now": now,
            }
            existing = connection.execute(
                text(
                    "SELECT transfer_id, target_key FROM integration_transfers "
                    "WHERE idempotency_key=:idempotency_key"
                ),
                values,
            ).mappings().first()
            if existing is None:
                connection.execute(
                    text(
                        "INSERT INTO integration_transfers "
                        "(transfer_id, source_system, source_key, target_system, target_key, "
                        "transfer_type, status, phase, idempotency_key, retry_count, task_id, "
                        "payload, result, created_at, updated_at) VALUES "
                        "(:transfer_id, 'OA', :source_key, 'PROCUREMENT', :target_key, "
                        "'OA_TO_PR', 'success', 'migrated', :idempotency_key, 0, "
                        "'migration-s0', :payload, :result, :now, :now)"
                    ),
                    values,
                )
            elif not existing["target_key"]:
                connection.execute(
                    text(
                        "UPDATE integration_transfers SET target_key=:target_key, "
                        "status='success', phase='migrated', error_code=NULL, "
                        "error_message=NULL, result=:result, updated_at=:now "
                        "WHERE idempotency_key=:idempotency_key"
                    ),
                    values,
                )

            connection.execute(
                text(
                    "INSERT OR IGNORE INTO business_lineages "
                    "(oa_apply_no, pr_no, task_id, latest_status, created_at, updated_at) "
                    "VALUES (:source_key, :target_key, 'migration-s0', "
                    "'pr_migrated', :now, :now)"
                ),
                values,
            )
            connection.execute(
                text(
                    "UPDATE business_lineages SET pr_no=:target_key, "
                    "task_id=COALESCE(task_id, 'migration-s0'), "
                    "latest_status=CASE WHEN po_no IS NULL OR po_no='' "
                    "THEN 'pr_migrated' ELSE latest_status END, updated_at=:now "
                    "WHERE oa_apply_no=:source_key"
                ),
                values,
            )
            connection.execute(
                text(
                    "UPDATE oa_applications SET linked_pr_no=:target_key, "
                    "procurement_transfer_status='success' "
                    "WHERE application_no=:source_key"
                ),
                values,
            )

        connection.execute(
            text(
                "INSERT OR IGNORE INTO schema_versions(version, description) "
                "VALUES (:version, 'S0 legacy PR adoption backfill')"
            ),
            {"version": S0_BACKFILL_VERSION},
        )
