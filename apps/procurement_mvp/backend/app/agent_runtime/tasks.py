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
