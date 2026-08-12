"""Natural-language intent routing via LangChain Runnable (rule-based LCEL)."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

QUICK_CHIPS = [
    {"id": "import_purchase_to_oa", "label": "帮我导入生产部的采购申请到 OA"},
    {"id": "submit_approved_purchase", "label": "处理已通过的采购申请"},
    {"id": "create_erp_po", "label": "帮我把待建 PR 创建成 ERP PO"},
    {"id": "view_current_task", "label": "查看当前任务"},
    {"id": "resume_last_task", "label": "继续上次任务"},
]


def _route(payload: dict[str, Any]) -> dict[str, Any]:
    message = str(payload.get("message") or "").strip()
    text = message.lower()
    department = None
    for name in ("生产部", "研发部", "行政部", "综合管理部", "市场部", "财务部", "信息中心"):
        if name in message:
            department = name
            break

    if any(key in message for key in ("查看当前任务", "当前任务", "任务状态")) or "view_current" in text:
        return {"intent": "view_current_task", "params": {}}

    if any(key in message for key in ("继续上次", "继续任务", "恢复任务")) or "resume" in text:
        return {"intent": "resume_last_task", "params": {}}

    if any(key in message for key in ("提交采购", "已通过", "处理已通过", "submit_approved")):
        return {
            "intent": "submit_approved_purchase",
            "params": {"department": department},
        }

    if any(
        key in message
        for key in (
            "创建ERP",
            "创建 PO",
            "创建PO",
            "创建成 ERP PO",
            "创建成ERP PO",
            "创建成 ERP",
            "待建 PO",
            "待建PO",
            "待建 PR",
            "待建PR",
            "建 PO",
            "建PO",
            "create_erp_po",
            "ERP采购订单",
            "ERP PO",
        )
    ) or ("PR-" in message and any(k in message for k in ("创建", "建单", "PO"))):
        pr_no = None
        for token in message.replace(",", " ").replace("，", " ").split():
            if token.upper().startswith("PR-"):
                pr_no = token.strip()
                break
        return {
            "intent": "create_erp_po",
            "params": {"department": department, "pr_no": pr_no},
        }

    if any(key in message for key in ("导入", "excel", "xlsx", "到 oa", "到oa", "import_purchase")):
        return {
            "intent": "import_purchase_to_oa",
            "params": {"department": department or "生产部"},
        }

    return {
        "intent": "unknown",
        "params": {},
        "reply": "暂未识别指令。可尝试：导入生产部采购申请到 OA / 处理已通过的采购申请 / 帮我把待建 PR 创建成 ERP PO / 查看当前任务。",
    }


intent_router = RunnableLambda(_route)


def route_message(message: str, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return intent_router.invoke({"message": message, "context": context or {}})
