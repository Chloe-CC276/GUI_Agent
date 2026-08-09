from __future__ import annotations

from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import Base, Database
from .migrations import DEFAULT_AWARD_SOURCES, DEFAULT_SUPPLIER_AWARD_LINKS
from .models import AwardSource, ERPMaterial, ERPSupplier, OAApplication, OAApplicationLine


def _seed_award_sources(session: Session) -> dict[str, AwardSource]:
    by_code: dict[str, AwardSource] = {
        item.code: item for item in session.scalars(select(AwardSource)).all()
    }
    for code, name in DEFAULT_AWARD_SOURCES:
        if code in by_code:
            continue
        source = AwardSource(code=code, name=name, status="active")
        session.add(source)
        by_code[code] = source
    session.flush()
    return by_code


def _seed(session: Session) -> None:
    award_by_code = _seed_award_sources(session)
    applications = [
        OAApplication(
            application_no="OA-2026-0001",
            title="研发部办公电脑采购",
            applicant="张明",
            department="研发部",
            status="APPROVED",
            procurement_status="NOT_STARTED",
            total_budget=Decimal("26000.00"),
            is_submitted=True,
            purchase_reason="研发岗位扩编，需补充笔记本电脑与显示器以满足日常开发办公。",
            budget_project_code="BUD-RD-2026",
            budget_project_name="研发办公设备预算",
            cost_center_code="CC-RD",
            requested_method="inquiry",
            urgency_level="NORMAL",
            approved_by="审批员甲",
            approval_opinion="同意采购",
            lines=[
                OAApplicationLine(
                    item_name="商务笔记本电脑",
                    specification="14英寸/32GB/1TB",
                    quantity=Decimal("2"),
                    estimated_unit_price=Decimal("9000.00"),
                ),
                OAApplicationLine(
                    item_name="27英寸显示器",
                    specification="4K IPS",
                    quantity=Decimal("2"),
                    estimated_unit_price=Decimal("4000.00"),
                ),
            ],
        ),
        OAApplication(
            application_no="OA-2026-0002",
            title="行政耗材补充",
            applicant="李华",
            department="行政部",
            status="DRAFT",
            total_budget=Decimal("3000.00"),
            is_submitted=False,
            purchase_reason="行政日常耗材库存不足，需补充复印纸等办公用品。",
            budget_project_name="行政办公耗材",
            urgency_level="NORMAL",
            lines=[
                OAApplicationLine(
                    item_name="A4复印纸",
                    specification="80g 500张/包",
                    quantity=Decimal("20"),
                    estimated_unit_price=Decimal("30.00"),
                )
            ],
        ),
        OAApplication(
            application_no="OA-2026-0003",
            title="会议室设备更新",
            applicant="王芳",
            department="综合管理部",
            status="IN_APPROVAL",
            total_budget=Decimal("12000.00"),
            is_submitted=True,
            purchase_reason="老旧会议室摄像头清晰度不足，影响跨区视频会议质量。",
            budget_project_code="BUD-FAC-2026",
            budget_project_name="设施更新预算",
            cost_center_code="CC-FAC",
            urgency_level="HIGH",
            current_approver_name="审批员乙",
            lines=[
                OAApplicationLine(
                    item_name="视频会议摄像头",
                    specification="4K USB",
                    quantity=Decimal("1"),
                    estimated_unit_price=Decimal("6500.00"),
                )
            ],
        ),
        OAApplication(
            application_no="OA-2026-0004",
            title="市场部移动终端采购",
            applicant="赵强",
            department="市场部",
            status="REJECTED",
            total_budget=Decimal("18000.00"),
            is_submitted=True,
            purchase_reason="市场外勤拜访需配备移动终端，用于演示与资料查阅。",
            budget_project_name="市场推广设备",
            urgency_level="NORMAL",
            approval_opinion="预算超标，请压缩数量后重提",
            lines=[
                OAApplicationLine(
                    item_name="平板电脑",
                    specification="11英寸 256GB",
                    quantity=Decimal("3"),
                    estimated_unit_price=Decimal("6000.00"),
                )
            ],
        ),
        OAApplication(
            application_no="OA-2026-0005",
            title="培训室投影设备补齐",
            applicant="陈敏",
            department="人力资源部",
            status="PENDING_APPROVAL",
            total_budget=Decimal("8500.00"),
            is_submitted=True,
            purchase_reason="新员工培训室缺少投影设备，影响集中授课与演示效果。",
            budget_project_code="BUD-HR-2026",
            budget_project_name="培训设施预算",
            cost_center_code="CC-HR",
            urgency_level="NORMAL",
            lines=[
                OAApplicationLine(
                    item_name="商务投影仪",
                    specification="1080P 4000流明",
                    quantity=Decimal("1"),
                    estimated_unit_price=Decimal("8500.00"),
                )
            ],
        ),
    ]
    materials = [
        ERPMaterial(
            material_code="ERP-LAPTOP-001",
            material_name="商务笔记本电脑",
            specification="14英寸/32GB/1TB",
            unit="台",
            standard_price=Decimal("8800.00"),
            status="active",
        ),
        ERPMaterial(
            material_code="ERP-MONITOR-001",
            material_name="27英寸显示器",
            specification="4K IPS",
            unit="台",
            standard_price=Decimal("3200.00"),
            status="active",
        ),
        ERPMaterial(
            material_code="ERP-PAPER-A4",
            material_name="A4复印纸",
            specification="80g 500张/包",
            unit="包",
            standard_price=Decimal("28.50"),
            status="active",
        ),
        ERPMaterial(
            material_code="ERP-CAMERA-4K",
            material_name="视频会议摄像头",
            specification="4K USB",
            unit="套",
            standard_price=Decimal("6200.00"),
            status="active",
        ),
        ERPMaterial(
            material_code="ERP-LEGACY-001",
            material_name="停用旧型号显示器",
            specification="24英寸",
            unit="台",
            standard_price=Decimal("1200.00"),
            status="inactive",
        ),
    ]
    supplier_defs = [
        ("SUP-001", "华创科技有限公司", "active"),
        ("SUP-002", "联信办公设备公司", "active"),
        ("SUP-003", "中建物资供应中心", "active"),
        ("SUP-004", "云商城自营店", "active"),
        ("SUP-LEGACY", "停用供应商", "inactive"),
    ]
    suppliers: list[ERPSupplier] = []
    for code, name, status in supplier_defs:
        supplier = ERPSupplier(supplier_code=code, supplier_name=name, status=status)
        for source_code in DEFAULT_SUPPLIER_AWARD_LINKS.get(code, ()):
            source = award_by_code.get(source_code)
            if source is not None:
                supplier.award_sources.append(source)
        suppliers.append(supplier)
    session.add_all([*applications, *materials, *suppliers])
    session.commit()


def init_database(database: Database) -> None:
    database.create_all()
    with database.session_factory() as session:
        if session.scalar(select(func.count(OAApplication.id))) == 0:
            _seed(session)


def reset_database(database: Database) -> None:
    from .migrations import (
        migrate_oa_closure,
        migrate_procurement_cloud,
        migrate_s0,
        migrate_supplier_award_sources,
        migrate_v21,
    )

    Base.metadata.drop_all(database.engine)
    migrate_s0(database.engine)
    Base.metadata.create_all(database.engine)
    migrate_v21(database.engine)
    migrate_oa_closure(database.engine)
    migrate_procurement_cloud(database.engine)
    migrate_supplier_award_sources(database.engine)
    with database.session_factory() as session:
        _seed(session)


# Short aliases are convenient for scripts and demo tooling.
init = init_database
reset = reset_database
