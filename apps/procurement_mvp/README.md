# 国企智慧采购智能助手 MVP

本地模拟 OA、采购云、ERP 三套业务系统，供人工操作与 GUI Agent / API 联调。  
不连接真实企业系统；不修改仓库 `src/` 下 GUI Agent 核心。

采购 MVP 内置 **方案 A 浏览器任务引擎**（浮窗 + LangChain 意图路由 + DomDriver）：
- 任务 `import_purchase_to_oa`：选文件夹/上传 Excel → 填 OA → 仅保存草稿
- 任务 `submit_approved_purchase`：定位已通过且未开始采购的 OA → 提交采购
- 闭环：Observe → Decide → Act → Verify → Update State；单步最多重试 2 次后 `wait_user`

当前版本按第二阶段规格书 **v2.1（传输闭环增强版）** 增量落地：

1. OA → 采购云 → ERP 传输闭环（S0）
2. ERP 采购全流程工作台
3. 新建采购申请录入（含 Excel 导入）
4. 批量申请导出与核对
5. Agent 全局浮窗

## 人工如何启动（必读）

> Windows 上请始终使用 `127.0.0.1`，不要用 `localhost`（可能解析到 IPv6 导致“无法建立连接”）。

### 首次安装（只需一次）

```powershell
cd D:\GUIAgent_project\GUI_Agent\apps\procurement_mvp
.\scripts\setup.ps1
```

### 日常启动（推荐）

**方式 A：双击启动**

```text
D:\GUIAgent_project\GUI_Agent\apps\procurement_mvp\scripts\start.cmd
```

**方式 B：PowerShell**

```powershell
cd D:\GUIAgent_project\GUI_Agent\apps\procurement_mvp
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

启动成功后终端会打印：

```text
OK  backend : http://127.0.0.1:8000/docs
OK  frontend: http://127.0.0.1:5173
```

然后浏览器打开：

| 用途 | 地址 |
|------|------|
| 前端首页 | http://127.0.0.1:5173 |
| OA 系统 | http://127.0.0.1:5173/oa |
| 采购云 | http://127.0.0.1:5173/procurement （E2E 使用 5174/8010，见 `frontend/e2e/run-e2e.mjs`） |
| ERP 工作台 | http://127.0.0.1:5173/erp/workbench |
| OpenAPI | http://127.0.0.1:8000/docs |
| 健康检查 | http://127.0.0.1:8000/health |

### 启动失败时

查看日志：

- `apps/procurement_mvp/.cache/logs/backend.log`
- `apps/procurement_mvp/.cache/logs/frontend.log`

确认本机 Node 位于 `E:\DevTools\nodejs\node-*-win-x64`（脚本会自动检测），Python 虚拟环境位于：

`D:\GUIAgent_project\GUI_Agent\apps\procurement_mvp\.venv`

停止服务：关闭启动脚本弹出的两个 PowerShell 窗口。

## 页面结构：三套独立系统

顶部是三个**大系统标签**（不是侧边栏混排）：

1. **OA 系统** — 采购申请审批
2. **采购云** — PR 承接、匹配、提交 ERP
3. **ERP** — 工作台、物料、订单、建单、批量导出

选中某个系统后，该系统的子页面从大标签**下方滑出**，点击子页切换内容区。  
Agent 为右下角全局浮窗，不占用系统主标签。

### 系统内子页

| 系统 | 子页 | 路由 |
|------|------|------|
| OA | 申请列表 / 新建 / 编辑 / 详情 | `/oa`、`/oa/applications/new`、`/oa/applications/:id/edit`、`/oa/:id` |
| OA | 审批工作台 / 审批详情 | `/oa/approvals`、`/oa/approvals/:id` |
| 采购云 | 采购申请列表 / 准备详情 | `/procurement`、`/procurement/requests`、`/procurement/requests/:prNo`（旧 `/procurement/:prNo` 兼容） |
| ERP | 工作台 | `/erp`、`/erp/workbench` |
| ERP | 物料主数据 | `/erp/materials` |
| ERP | 采购订单 | `/erp/orders`、`/erp/orders/:poNo` |
| ERP | 新建采购申请 | `/erp/requests/new` |
| ERP | 批量导出与核对 | `/erp/export` |

## 推荐人工演示路径

### OA 审批闭环（人工确认）

1. OA → **新建采购申请** → 保存草稿 → **提交审批**（状态仍为草稿，`is_submitted=true`）
2. OA → **审批工作台** → 待开始 → 打开 `OA-2026-0005` 或刚提交单 → **开始审批**
3. 审批中 → **通过**（或驳回必填原因；驳回后可修改并重新提交）
4. 仅 `APPROVED` 且采购执行未定标的单据，详情页可点 **提交采购** 进入采购云（`procurement_status=PREPARING`，审批状态仍为已通过）

### 采购云 MVP：准备确认 → 提交 ERP

1. 打开 OA → `OA-2026-0001`（已通过）→ **提交采购**（跳转 `/procurement/requests/{pr_no}`）
2. 采购云准备详情：确认采购方式、选择 ERP 供应商、填写结果来源、核对含税单价
3. **保存** → **校验** → **确认并提交 ERP**（弹窗核对 OA/PR/供应商/总额/行数）
4. 成功后 OA `procurement_status=已定标`，列表可筛「已定标」并跳转 ERP PO
5. Excel/批量导出仍在 ERP 导航；右下角 Agent 浮窗可查 `task_id` / 重置演示数据

数据乱了可在 Agent 浮窗或原诊断入口执行重置：`POST /api/v1/demo/reset`。

## 环境落盘位置（禁止写 C 盘业务缓存）

- Python venv：`apps/procurement_mvp/.venv`
- pip / npm / Playwright / 日志缓存：`apps/procurement_mvp/.cache`
- 前端依赖：`apps/procurement_mvp/frontend/node_modules`
- SQLite：`apps/procurement_mvp/backend/procurement_demo.db`

## 初始化、重置与测试

```powershell
cd D:\GUIAgent_project\GUI_Agent\apps\procurement_mvp

# 空库初始化（不覆盖已有业务数据）
.\.venv\Scripts\python.exe .\scripts\init_demo.py

# 清空并恢复固定 seed
.\.venv\Scripts\python.exe .\scripts\reset_demo.py

# S0 增量迁移（备份后 ALTER，不 drop）
.\.venv\Scripts\python.exe .\scripts\migrate_s0.py

# 后端 + 前端单测 + 构建 + Playwright E2E
.\scripts\test.ps1

# 生成四套各 100 条 Excel 测试数据 → test_data/
.\.venv\Scripts\python.exe .\scripts\generate_test_excels.py
```

## 核心 API（节选）

```text
# OA 审批闭环
POST /api/v1/oa/applications
PUT  /api/v1/oa/applications/{id}
POST /api/v1/oa/applications/{id}/submit
POST /api/v1/oa/applications/{id}/start-approval
POST /api/v1/oa/applications/{id}/approve
POST /api/v1/oa/applications/{id}/reject
POST /api/v1/oa/applications/{id}/resubmit
GET  /api/v1/oa/approvals?queue=pending_start|in_approval|done
GET  /api/v1/oa/applications/approved

# S0 传输闭环
POST /api/v1/oa/proposals/{oa_apply_no}/push-to-procurement
POST /api/v1/procurement/requests/{pr_no}/prepare-erp-submit
POST /api/v1/procurement/requests/{pr_no}/push-to-erp
POST /api/v1/integration/transfers/{transfer_id}/retry
GET  /api/v1/oa/proposals/{oa_apply_no}/lineage
GET  /api/v1/erp/orders
GET  /api/v1/erp/orders/{po_no}

# v2.1 工作台 / 导入 / 导出 / Agent
GET  /api/v1/workbench/summary
GET  /api/v1/workbench/events
POST /api/v1/procurement/imports/preview
POST /api/v1/procurement/imports/confirm
GET  /api/v1/procurement/requests/export-candidates
POST /api/v1/procurement/requests/batch-validate
POST /api/v1/procurement/requests/batch-export
GET  /api/v1/config/purchase-method-rules
GET  /api/v1/agent/tasks/active
```

成功响应统一 `{ ok, data, task_id, business_key, idempotent_replay }`；  
业务错误统一 `{ ok: false, error: { code, message, details } }`。

## 业务硬规则

- 仅 OA `APPROVED`（兼容旧 `approved`）可发起采购 / 进入正式提交准备
- OA 状态迁移必须走 command API；前端不得直接改 `approval_status`
- 审批通过须人工确认；`approve` 成功后写入 outbox / 可供 `GET .../approved` 查询
- 物料编码必须命中有效 ERP 主数据
- PR 金额由服务端按行汇总（Decimal）
- 跨系统差异留痕，禁止静默覆盖
- 提交 ERP、批量导出等高风险动作必须显式确认
- `task_id + business_key + operation` 幂等；传输失败可重试；目标已创建时只重试回写

## 目录

```text
procurement_mvp/
├── backend/     # FastAPI、SQLAlchemy、迁移、pytest
├── frontend/    # React、Ant Design、Vitest、Playwright
├── scripts/     # setup / start / init / reset / migrate / test
├── test_data/   # 生成的 Excel 夹具
└── README.md
```
