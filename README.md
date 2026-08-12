# GUI Agent · 国企智慧采购智能助手

用 **GUI Agent** 驱动企业采购全链路：在真实业务页面上完成操作，而不是靠虚构“一键建单”API 掩盖系统割裂。

本仓库包含两块能力：

1. **采购业务 MVP**（`apps/procurement_mvp`）——本地模拟 **OA → 采购云 → ERP**，演示人工操作与 Agent 联调  
2. **桌面感知与执行底座**（`perception/` / `executor/`）——截图、OCR、UI 检测与键鼠动作调度，面向更通用的 GUI 自动化

---

## 这个项目在解决什么问题

企业里常见的现实是：

- OA 负责申请与审批  
- 采购云负责寻源、定标、待建队列  
- ERP 负责正式采购订单（PO）落库  
- 系统之间 **没有完整的业务建单 API**，人或 RPA 只能在 ERP 界面里填单

本项目按 **方案 A** 建模这一现实：

> Agent / RPA 在 ERP 页面完成建单 → 回读 `po_no` → 仅通过采购云写回接口闭环；  
> **不假装**存在“采购云直接创建 ERP PO”的业务 API。

演示闭环覆盖：

```text
Excel/人工录入 → OA 审批 → 提交采购云 → 定标进入待建 PO
       → Agent 打开 ERP 建单页填单/校验/保存
       → 回读 PO 号并写回采购云
```

---

## 产品设计思路

| 原则 | 做法 |
|------|------|
| **界面即真相** | 关键控件带稳定 `data-testid`，Agent 用 DOM 驱动执行，便于校验与回归 |
| **Observe → Decide → Act → Verify** | 任务状态机驱动；单步失败有限重试，超限进入 `wait_user`，不盲目跳过 |
| **RPA 优先，VLM 可插拔** | Phase1 以规则/DOM 闭环；Phase2 增加可开关 VLM 增强（脆弱节点辅助，低置信可转人工） |
| **人机闸门** | 缺文件、多候选、校验失败、步骤超时 → 浮窗等待用户确认，不自动瞎猜 |
| **三系统分栏** | 顶栏切换 OA / 采购云 / ERP，贴近真实企业门户习惯，而非把流程糊成一个页面 |
| **可追溯** | `task_id` / `business_key` / batch / 安全事件与 Excel 跟踪快照，方便审计演示 |

Agent 当前可驱动的典型意图：

- 导入生产部采购申请到 OA（仅存草稿）  
- 处理已通过的采购申请（提交采购云）  
- 把待建 PR 创建成 ERP PO（GUI 填单 → 保存 → 写回）

---

## 技术栈

### 采购 MVP（演示应用）

| 层 | 技术 |
|----|------|
| 前端 | React 18、TypeScript、Vite、Ant Design 5、Lucide、React Router |
| 后端 | FastAPI、SQLAlchemy 2、Pydantic、SQLite（演示库） |
| Agent 运行时 | 浮窗任务引擎、LangChain Core 意图路由、浏览器 DomDriver |
| 数据/联调 | openpyxl、httpx、Playwright / Vitest（前端测试） |

### GUI 底座（通用能力）

| 模块 | 技术与职责 |
|------|------------|
| `perception/` | 截图、OpenCV 预处理、PaddleOCR、启发式 UI 检测、感知流水线 |
| `executor/` | 标准 Action 模型、鼠标/键盘控制、动作序列调度（支持 dry-run） |

---

## 仓库结构（精简）

```text
GUI_Agent/
├── apps/procurement_mvp/     # 可运行的 OA / 采购云 / ERP + GUI Agent 演示
├── perception/               # 屏幕感知（截图 / OCR / UI 检测）
├── executor/                 # 桌面动作执行（键鼠 / 动作序列）
└── README.md                 # 本页：产品定位与技术概览
```

---

## 快速体验

> Windows 请使用 `127.0.0.1`，避免 `localhost` 解析到 IPv6。

```powershell
cd apps/procurement_mvp
.\scripts\setup.ps1          # 首次安装依赖
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

| 入口 | 地址 |
|------|------|
| 前端（OA / 采购云 / ERP） | http://127.0.0.1:5173 |
| API 文档 | http://127.0.0.1:8000/docs |
| 健康检查 | http://127.0.0.1:8000/health |

打开右下角 **GUI Agent** 浮窗，可用快捷指令驱动：导入 OA → 提交采购 → 待建 PR 创建 ERP PO。

本地脚本、日志路径与页面地图等运维细节在 `apps/procurement_mvp/scripts/` 与同目录说明文件中，不影响对本页产品叙事的阅读。

---

## 当前进展（便于评审）

- [x] OA / 采购云 / ERP 本地三系统与主数据  
- [x] 传输与待建 PO 队列（方案 A）  
- [x] Agent：导入 OA、提交采购、GUI 创建 ERP PO  
- [x] Phase2：可开关 VLM 增强 + 企业风 UI（antd + Lucide Token）  
- [ ] 对接真实企业系统 / 生产级鉴权与部署（非本 MVP 范围）

---

## 说明

- MVP **不连接**真实企业 OA/采购云/ERP；数据可重置，仅用于演示与联调。  
- 业务演示应用与桌面感知/执行底座解耦：可先跑通采购闭环，再扩展到更广的 GUI 自动化场景。  
- 功能开发默认以分支交付；最新 Agent/UI 能力请查看分支 `cursor/model-tuning-framework-9115`（若尚未合入 `main`）。
