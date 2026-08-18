"""Create a one-application Excel fixture for Agent OA E2E."""

from __future__ import annotations

import sys
from pathlib import Path

from openpyxl import Workbook

MVP_ROOT = Path(__file__).resolve().parents[1]
OUT = MVP_ROOT / ".cache" / "tmp" / "agent_import_prod.xlsx"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        [
            "title",
            "applicant",
            "department",
            "budget_project_name",
            "budget_project_code",
            "purchase_reason",
            "urgency_level",
            "requested_method",
            "item_name",
            "specification",
            "quantity",
            "estimated_unit_price",
            "total_budget",
        ]
    )
    title = "生产部办公设备采购-Agent"
    reason = "生产岗位扩编后需补充办公设备，保障日常作业效率。"
    rows = [
        [title, "张明", "生产部", "安全生产专项", "BUD-SAFE-2026", reason, "NORMAL", "inquiry", "商务笔记本电脑", "14英寸/32GB/1TB", 2, 9000, 26000],
        [title, "张明", "生产部", "安全生产专项", "BUD-SAFE-2026", reason, "NORMAL", "inquiry", "27英寸显示器", "4K IPS", 2, 4000, 26000],
    ]
    for row in rows:
        sheet.append(row)
    workbook.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
    sys.exit(0)
