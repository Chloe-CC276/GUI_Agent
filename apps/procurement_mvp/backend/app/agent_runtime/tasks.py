"""Deterministic step graphs for OA GUI Agent tasks."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def _step(
    step_id: str,
    *,
    title: str,
    action: dict[str, Any],
    verify: dict[str, Any],
    expected: str,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "title": title,
        "action": action,
        "verify": verify,
        "expected": expected,
        "actual": None,
        "status": "pending",
        "retry_count": 0,
    }


def _urgency_label(raw: Any) -> str:
    text = str(raw or "NORMAL").strip()
    if text in {"普通", "加急", "特急"}:
        return text
    return {
        "NORMAL": "普通",
        "URGENT": "加急",
        "CRITICAL": "特急",
        "HIGH": "加急",
    }.get(text.upper(), "普通")


def _method_label(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text in {"网购", "比价", "招标", "集中采购", "单一来源", "框架协议"}:
        return text
    return {
        "online": "网购",
        "inquiry": "比价",
        "bidding": "招标",
        "centralized": "集中采购",
        "single": "单一来源",
        "framework": "框架协议",
    }.get(text.lower(), "")


def build_import_purchase_steps(application: dict[str, Any]) -> list[dict[str, Any]]:
    lines = application["lines"]
    steps: list[dict[str, Any]] = [
        _step(
            "OPEN_OA",
            title="打开 OA 申请列表",
            action={"type": "navigate", "path": "/oa"},
            verify={"type": "testid_visible", "testid": "oa-list-page"},
            expected="oa-list-page visible",
        ),
        _step(
            "OPEN_CREATE_FORM",
            title="打开新建申请表单",
            action={"type": "click", "testid": "oa-create-button"},
            verify={"type": "testid_visible", "testid": "oa-form-page"},
            expected="oa-form-page visible",
        ),
        _step(
            "FILL_HEADER",
            title="填写申请头",
            action={
                "type": "fill_fields",
                "fields": [
                    {"testid": "oa-form-title", "value": application["title"]},
                    {"testid": "oa-form-department", "value": application["department"]},
                    {"testid": "oa-form-applicant", "value": application["applicant"]},
                ],
            },
            verify={
                "type": "fields_equals",
                "fields": [
                    {"testid": "oa-form-title", "value": application["title"]},
                    {"testid": "oa-form-department", "value": application["department"]},
                    {"testid": "oa-form-applicant", "value": application["applicant"]},
                ],
            },
            expected="header fields match Excel",
        ),
    ]

    for index, line in enumerate(lines):
        if index > 0:
            steps.append(
                _step(
                    f"ADD_LINE_{index}",
                    title=f"添加明细行 #{index + 1}",
                    action={
                        "type": "click",
                        "testid": "oa-line-add-button",
                        "expect_line_count": index + 1,
                    },
                    verify={"type": "line_count_at_least", "count": index + 1},
                    expected=f"line count >= {index + 1}",
                )
            )
        steps.append(
            _step(
                f"FILL_ITEMS_{index}",
                title=f"填写物资明细 #{index + 1}",
                action={
                    "type": "fill_fields",
                    "fields": [
                        {"testid": f"oa-line-item-name-{index}", "value": line["item_name"]},
                        {
                            "testid": f"oa-line-spec-{index}",
                            "value": line.get("specification") or "",
                        },
                        {
                            "testid": f"oa-line-qty-{index}",
                            "value": str(line["quantity"]),
                            "input_type": "number",
                        },
                        {
                            "testid": f"oa-line-price-{index}",
                            "value": str(line["estimated_unit_price"]),
                            "input_type": "number",
                        },
                    ],
                },
                verify={
                    "type": "fields_equals",
                    "fields": [
                        {"testid": f"oa-line-item-name-{index}", "value": line["item_name"]},
                        {
                            "testid": f"oa-line-qty-{index}",
                            "value": str(Decimal(str(line["quantity"]))),
                            "loose_number": True,
                        },
                    ],
                },
                expected=f"line {index} filled",
            )
        )

    steps.extend(
        [
            _step(
                "FILL_BUDGET",
                title="填写预算信息",
                action={
                    "type": "fill_fields",
                    "fields": [
                        {
                            "testid": "oa-form-budget-project-name",
                            "value": application.get("budget_project_name") or "",
                        },
                        {
                            "testid": "oa-form-budget-project-code",
                            "value": application.get("budget_project_code") or "",
                        },
                        {
                            "testid": "oa-form-cost-center",
                            "value": application.get("cost_center_code") or "",
                        },
                        {
                            "testid": "oa-form-purchase-reason",
                            "value": application.get("purchase_reason") or "",
                        },
                    ],
                },
                verify={
                    "type": "fields_equals",
                    "fields": [
                        {
                            "testid": "oa-form-budget-project-name",
                            "value": application.get("budget_project_name") or "",
                        },
                        {
                            "testid": "oa-form-purchase-reason",
                            "value": application.get("purchase_reason") or "",
                        },
                    ],
                },
                expected="budget fields filled",
            ),
            _step(
                "FILL_OTHER_FIELDS",
                title="填写紧急程度与其他字段",
                action={
                    "type": "fill_fields",
                    "fields": [
                        {
                            "testid": "oa-form-urgency",
                            "value": _urgency_label(application.get("urgency_level")),
                            "input_type": "select",
                        },
                        {
                            "testid": "oa-form-requested-method",
                            "value": _method_label(application.get("requested_method")),
                            "input_type": "select",
                            "optional": True,
                        },
                        {
                            "testid": "oa-form-remark",
                            "value": application.get("remark") or "",
                            "optional": True,
                        },
                    ],
                },
                verify={"type": "testid_visible", "testid": "oa-form-page"},
                expected="form still available after filling other fields",
            ),
            _step(
                "VERIFY_FORM",
                title="提交前校验表单（不点提交审批）",
                action={
                    "type": "assert_no_click",
                    "forbidden_testid": "oa-submit-approval-button",
                },
                verify={
                    "type": "budget_matches",
                    "testid": "oa-form-total-budget",
                    "value": str(application.get("total_budget")),
                },
                expected="budget total matches and submit-approval not clicked",
            ),
            _step(
                "SAVE_DRAFT",
                title="保存草稿",
                action={"type": "click", "testid": "oa-save-draft-button"},
                verify={"type": "url_matches", "pattern": "/oa/applications/\\d+/edit|/oa/\\d+/edit"},
                expected="navigated to draft edit page",
            ),
            _step(
                "VERIFY_DRAFT",
                title="验证草稿状态与申请编号",
                action={"type": "read_draft_status"},
                verify={
                    "type": "draft_status",
                    "approval_status": "DRAFT",
                    "procurement_status": "NOT_STARTED",
                },
                expected="approval_status=DRAFT and procurement_status=NOT_STARTED with OA号",
            ),
        ]
    )
    return steps


def build_submit_approved_steps(
    *,
    oa_id: int,
    application_no: str,
    expected_header: dict[str, Any],
    expected_lines: list[dict[str, Any]],
    expected_total: str,
) -> list[dict[str, Any]]:
    return [
        _step(
            "OPEN_OA",
            title="打开 OA 申请列表",
            action={"type": "navigate", "path": "/oa"},
            verify={"type": "testid_visible", "testid": "oa-list-page"},
            expected="oa-list-page visible",
        ),
        _step(
            "OPEN_DETAIL",
            title=f"打开目标申请 {application_no}",
            action={"type": "click", "testid": f"oa-view-{oa_id}"},
            verify={"type": "testid_visible", "testid": "oa-detail-page"},
            expected="detail page visible",
        ),
        _step(
            "VERIFY_TARGET_OA",
            title="核对目标 OA 编号",
            action={"type": "noop"},
            verify={
                "type": "page_contains",
                "text": application_no,
            },
            expected=f"page contains {application_no}",
        ),
        _step(
            "VERIFY_HEADER_AND_LINES",
            title="核对申请头与物资明细金额",
            action={
                "type": "verify_detail_payload",
                "expected_header": expected_header,
                "expected_lines": expected_lines,
                "expected_total": expected_total,
            },
            verify={
                "type": "detail_amount_match",
                "expected_total": expected_total,
                "line_count": len(expected_lines),
            },
            expected="header/lines/total match",
        ),
        _step(
            "SUBMIT_PROCUREMENT",
            title="提交采购",
            action={"type": "click", "testid": "enter-procurement-button", "text_includes": "提交采购"},
            verify={
                "type": "procurement_preparing",
                "approval_status": "APPROVED",
                "procurement_status": "PREPARING",
                "text": "采购准备中",
            },
            expected="APPROVED + PREPARING + 采购准备中",
        ),
    ]


def build_create_erp_po_gui_steps(
    *,
    pr_no: str,
    task_id: str,
    form: dict[str, Any],
) -> list[dict[str, Any]]:
    """DOM steps for Scheme A: Agent fills ERP create page (no procurement→ERP business API)."""
    header = form.get("header") or {}
    lines = form.get("lines") or []
    route = f"/erp/po-create/{task_id}"
    fields = [
        {"testid": "erp-po-supplier", "value": str(header.get("supplier_name") or "")},
        {"testid": "erp-po-supplier-code", "value": str(header.get("supplier_code") or "")},
        {"testid": "erp-po-purchasing-org", "value": str(header.get("purchasing_org") or "1000")},
        {"testid": "erp-po-purchasing-group", "value": str(header.get("purchasing_group") or "P01")},
        {"testid": "erp-po-currency", "value": str(header.get("currency_code") or "CNY")},
        {"testid": "erp-po-payment-terms", "value": str(header.get("payment_terms") or "NET30")},
        {"testid": "erp-po-request-dept", "value": str(header.get("request_dept") or "")},
        {"testid": "erp-po-buyer", "value": str(header.get("buyer_id") or "BUYER-01")},
    ]
    line_fields: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        line_fields.extend(
            [
                {
                    "testid": f"erp-po-line-material-{index}",
                    "value": str(line.get("material_code") or ""),
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
        _step(
            "READ_PR_DATA",
            title=f"确认待建 PR {pr_no}",
            action={"type": "noop", "pr_no": pr_no},
            verify={"type": "always"},
            expected=f"PR {pr_no} ready for ERP GUI create",
        ),
        _step(
            "OPEN_ERP_FORM",
            title="打开 ERP 建单草稿页",
            action={"type": "navigate", "path": route},
            verify={"type": "testid_visible", "testid": "erp-po-create-form"},
            expected="ERP create form visible",
        ),
        _step(
            "FILL_HEADER",
            title="填写 PO Header",
            action={"type": "fill_fields", "fields": fields},
            verify={
                "type": "fields_equals",
                "fields": [
                    {"testid": "erp-po-supplier", "value": str(header.get("supplier_name") or "")},
                    {"testid": "erp-po-purchasing-org", "value": str(header.get("purchasing_org") or "1000")},
                ],
            },
            expected="header fields match",
        ),
        _step(
            "FILL_LINES",
            title="填写物资行",
            action={"type": "fill_fields", "fields": line_fields},
            verify={
                "type": "line_count_at_least",
                "count": max(len(lines), 1),
                "testid": "erp-po-line-row",
            },
            expected=f"line count >= {len(lines)}",
        ),
        _step(
            "PRE_SAVE_VERIFY",
            title="保存前校验",
            action={"type": "click", "testid": "erp-po-verify-button"},
            verify={"type": "always"},
            expected="pre-save verify clicked",
        ),
        _step(
            "SAVE_PO",
            title="保存并创建 PO",
            action={"type": "click", "testid": "erp-po-create-button"},
            verify={"type": "po_created", "pr_no": pr_no, "testid": "erp-po-created-po-no"},
            expected="PO created and po_no readable",
        ),
        _step(
            "READ_BACK_PO_NO",
            title="回读 PO 号",
            action={"type": "read_text", "testid": "erp-po-created-po-no", "pr_no": pr_no},
            verify={"type": "po_created", "pr_no": pr_no, "testid": "erp-po-created-po-no"},
            expected="po_no non-empty",
        ),
    ]
