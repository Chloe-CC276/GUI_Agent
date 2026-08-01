# GUI Agent 模型调优框架

> 目标：在现有 `perception → plan → execute → verify` Agent 上，建立可迭代的
> **框架优化 / LoRA GUI 训练 / 提示词优化** 闭环。
>
> 当前状态：Agent 运行时、Prompt 体系、GUI 数据集管道已就绪；
> `transformers` / `peft` / `accelerate` / `bitsandbytes` 已声明但未接入；
> 本框架定义后续落地路径与模块边界。

---

## 1. 总体闭环

```text
┌─────────────────────────────────────────────────────────────────┐
│                     GUI Agent Tuning Loop                       │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │ A. 框架优化   │←──│ C. 提示词优化     │←──│ 评测反馈         │  │
│  │ runtime/VLM  │   │ PromptBuilder    │   │ offline+online  │  │
│  └──────┬───────┘   └────────┬─────────┘   └────────▲────────┘  │
│         │                    │                      │           │
│         ▼                    ▼                      │           │
│  ┌──────────────────────────────────────┐           │           │
│  │ B. LoRA GUI 数据集训练               │───────────┘           │
│  │ data → SFT/DPO → adapter → serve     │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

三条线并行推进，用同一套评测指标对齐：

| 线 | 改什么 | 主要产出 |
|----|--------|----------|
| A 框架优化 | 推理适配、Agent 循环、本地模型加载 | `LocalHFVLM`、adapter serve、评测入口 |
| B LoRA 训练 | GUI 轨迹 → 多模态 SFT/DPO | 训练脚本、LoRA adapter、训练配置 |
| C 提示词优化 | Planner/Verify/Repair 模板与 schema | 版本化 prompt、训练-推理对齐 |

---

## 2. 与现有代码的映射

| 现有模块 | 在调优中的角色 |
|----------|----------------|
| `src/datasets/schema.py` (`GUITaskSample` / `GUITaskStep` / `PlanningSample`) | 统一数据契约；训练样本从此派生 |
| `src/datasets/*_loader.py` + `script/preprocess_*.py` | 原始数据 → `data/processed/*.jsonl` |
| `src/agent/prompts/*` | 推理侧 prompt；训练侧必须复用同一 builder |
| `src/model/base_vlm.py` / `qwen_vlm.py` | API 推理；扩展本地 HF + LoRA 适配器 |
| `src/agent/planner.py` + `agent_chain.py` / `agent_graph.py` | 在线评测与 dry-run 对比基线 |
| `requirements.txt` (`peft` 等) | 训练依赖已预留，落地时补齐版本与 `torch` |

优先训练数据：**ScreenAgent**（有截图 + `llm_response` + `corrected_response`）。
Mind2Web 作语义动作辅助；WebArena 主要用于任务级评测（无演示轨迹）。

---

## 3. 目标目录结构（规划，尚未落地）

```text
src/tuning/                   # 规划包，实现阶段再创建
├── configs/                  # YAML/JSON 训练与评测配置
│   └── default_qlora.yaml
├── data/                     # 样本转换与过滤
│   ├── sft_converter.py      # GUITaskSample → 多模态 chat SFT
│   ├── dpo_converter.py      # llm_response vs corrected_response
│   └── filters.py            # 缺图/坏动作/超长样本过滤
├── train/                    # LoRA / QLoRA 训练
│   ├── collator.py           # 图文 batch collate
│   ├── train_lora.py         # 主训练入口
│   └── export_adapter.py     # 导出/合并 adapter
└── eval/                     # 离线 + 在线评测
    ├── offline_metrics.py    # JSON 合法率、动作类型、坐标误差
    └── online_harness.py     # dry-run / 有限步 Agent 回放

docs/
└── MODEL_TUNING_FRAMEWORK.md # 本文件（当前仅有文档）

data/                         # gitignored
├── processed/                # 已有 preprocess 输出
├── sft/                      # 转换后的训练 JSONL
├── dpo/                      # 偏好对（可选）
└── adapters/                 # LoRA 权重输出
```

---

## 4. 线 A — 框架优化

### 4.1 目标

让 Agent 既能继续用 DashScope API 基线，也能加载本地 VLM + LoRA，并保持
`BaseVLM` 接口不变，从而 planner / chain / graph 无需分叉。

### 4.2 工作包

1. **统一推理后端**
   - 新增 `src/model/local_hf_vlm.py`：HuggingFace 多模态加载
   - 支持 `peft.PeftModel.from_pretrained` 挂载 adapter
   - CLI（`src/agent/cli.py`）增加 `--backend api|local`、`--adapter-path`

2. **训练-推理格式对齐**
   - 训练用的 chat template / system prompt 必须来自 `PromptBuilder`
   - 禁止训练脚本手写第二套 planner 规则

3. **Agent 循环效率**
   - 截图 / OCR / 元素数裁剪与 `PromptConfig` 限额联动
   - 结构化 JSON 失败时优先走 `repair`，减少无效重试
   - 记录每步 `latency / token / parse_ok` 便于对比 LoRA 前后

4. **评测入口**
   - Offline：固定 screenshot + instruction → 预测 action
   - Online：`dry_run` 回放 ScreenAgent 轨迹，比步级动作匹配率

### 4.3 完成标准

- API 基线与 Local+LoRA 可切换，同一任务输出同 schema
- 评测脚本可对「无 LoRA / 有 LoRA」出对比表

---

## 5. 线 B — LoRA GUI 数据集训练

### 5.1 数据流

```text
external/{ScreenAgent,Mind2Web,...}
        │  script/preprocess_*.py
        ▼
data/processed/{source}/{split}.jsonl     # GUITaskSample
        │  src/tuning/data/sft_converter.py
        ▼
data/sft/{split}.jsonl                    # chat + image paths
        │  src/tuning/train/train_lora.py
        ▼
data/adapters/<run_id>/                   # LoRA / QLoRA weights
        │  LocalHFVLM.load_adapter
        ▼
Agent Planner / Eval
```

### 5.2 SFT 样本约定

每条训练样本对应 **一个 GUITaskStep**（或 PlanningSample）：

```json
{
  "messages": [
    {"role": "system", "content": "<PromptBuilder planner system>"},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "<instruction + context>"},
        {"type": "image", "image": "<screenshot_path>"}
      ]
    },
    {"role": "assistant", "content": "<target JSON action>"}
  ],
  "meta": {
    "task_id": "...",
    "source": "screenagent",
    "step_id": 0,
    "label_source": "corrected_response"
  }
}
```

标签优先级：

1. `corrected_response`（人工修正，首选）
2. 结构化 `action` 序列化为 schema JSON（兜底）
3. 原始 `llm_response`（仅作 DPO 负样本或弱监督，不作为主 SFT 标签）

### 5.3 过滤规则（`filters.py`）

- 截图缺失或不可读 → drop
- 动作无法解析 / 坐标越界 → drop 或修复后保留
- 超长 prompt（相对 `PromptConfig.max_prompt_chars`）→ 裁剪元素列表或 drop
- 非 executable 步（纯 semantic Mind2Web）→ 单独 semantic 实验，不混入桌面坐标 SFT

### 5.4 训练方法

| 阶段 | 方法 | 说明 |
|------|------|------|
| P0 | QLoRA SFT | 4bit 基座 + PEFT LoRA，主路径 |
| P1 | LoRA SFT | 显存充足时对比全精度 adapter |
| P2 | DPO / 偏好优化 | `corrected_response` 正 vs `llm_response` 负 |

默认超参放在 `src/tuning/configs/default_qlora.yaml`（后续实现时填充）：

- 基座：与线上接近的开源 Qwen2.5-VL / Qwen3-VL 系列（按可用权重选定）
- LoRA：`r` / `alpha` / `dropout`、目标模块（视觉投影 + LLM attention）
- 训练：epoch、lr、batch、gradient checkpointing、max image resolution

### 5.5 完成标准

- 一键：`preprocess → convert → train → eval`
- adapter 可被 Local VLM 加载，offline 动作匹配率相对 API/基座有可复现提升

---

## 6. 线 C — 提示词优化

### 6.1 原则

**训练用 prompt ≡ 推理用 prompt。**  
所有变更只走 `src/agent/prompts/`，经 `PromptBuilder` 导出后再写入 SFT。

### 6.2 工作包

1. **版本化**
   - 为 planner / verify / repair / reflection / memory 模板加 `prompt_version`
   - SFT `meta.prompt_version` 记录，避免旧数据污染新模板实验

2. **结构化约束强化**
   - 以 `schemas.py` 为唯一 JSON 契约
   - 评测拆分：`json_ok` / `schema_ok` / `action_type_acc` / `coord_mae`

3. **消融实验**
   - 有无 OCR/元素列表
   - 历史步数 `history_limit`
   - 中英模板（ScreenAgent 含中文）
   - 简短 vs 完整规则（token 成本 vs 格式遵从）

4. **与 LoRA 的配合顺序**
   - 先固定 prompt vN，训一版 LoRA
   - 再只改 prompt 做零样本对比
   - 最后「新 prompt + 重训 LoRA」验证是否叠加增益

### 6.3 完成标准

- Prompt 变更可追溯版本
- 同一测试集上可对比：仅改 prompt / 仅加 LoRA / 两者组合

---

## 7. 评测体系（三条线共用）

| 层级 | 指标 | 数据 |
|------|------|------|
| 格式 | JSON 可解析率、schema 通过率 | ScreenAgent val/test steps |
| 动作 | action type accuracy、参数命中 | 同上 |
| 坐标 | 点击/拖拽像素 MAE 或归一化误差 | executable steps |
| 轨迹 | 步级 exact / soft match | 任务级回放 |
| 在线 | 有限步任务完成率（dry-run 或沙箱） | ScreenAgent / WebArena intents |

基线对照：

1. API `qwen3-vl-plus`（现状）
2. 本地基座无 LoRA
3. 本地基座 + LoRA
4. （可选）不同 `prompt_version`

---

## 8. 实施顺序（技术依赖，非日历）

```text
Phase 0  框架设计（当前）
         └─ 仅文档：docs/MODEL_TUNING_FRAMEWORK.md

Phase 1  数据转换
         ├─ 创建 src/tuning/data（sft_converter / filters）
         ├─ 补齐 ScreenAgent export_jsonl（若仍缺失）
         └─ 产出 data/sft/{train,validation,test}.jsonl

Phase 2  推理侧本地后端
         ├─ LocalHFVLM + adapter 加载
         └─ CLI / Planner 切换 backend

Phase 3  QLoRA 训练
         ├─ train_lora.py + default_qlora.yaml
         └─ 小规模 smoke → 全量 ScreenAgent

Phase 4  评测闭环
         ├─ offline_metrics + 对比报告
         └─ online dry-run harness

Phase 5  提示词迭代 + 可选 DPO
         ├─ prompt_version 消融
         └─ corrected vs raw 偏好优化
```

每阶段合并前至少保证：单元测试覆盖转换/过滤；smoke 训练不要求 GPU 全量跑通（CI 可 mock）。

---

## 9. 风险与约束

| 风险 | 应对 |
|------|------|
| 训练-推理 prompt 漂移 | 强制共用 `PromptBuilder`；SFT 记录 `prompt_version` |
| ScreenAgent / Mind2Web 动作空间不一致 | 桌面坐标 SFT 与语义动作分轨 |
| 显存与依赖（Python 3.13 + torch/peft） | 配置中锁定经实测的 Python/torch 组合；QLoRA 优先 |
| 大数据与权重体积 | 继续 gitignore `data/`、`external/`、adapters |
| API 基线与开源基座能力差 | 评测始终带 API 对照，避免误判 LoRA 增益 |

---

## 10. 下一步（实现入口）

确认本框架后，实现阶段优先：

1. `src/tuning/data/sft_converter.py` — 从 `ProcessedDatasetLoader` 读入并写出 SFT JSONL  
2. `src/tuning/configs/default_qlora.yaml` — 填入可运行默认超参  
3. `src/model/local_hf_vlm.py` — 打通本地推理接口  

提示词优化在 Phase 1 数据转换时同步冻结一版 `prompt_version`，避免边训边改。
