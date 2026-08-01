# `src/tuning` — GUI Agent 模型调优

本目录承载 **LoRA/QLoRA 训练、样本转换、离线评测**。
完整设计见 [`docs/MODEL_TUNING_FRAMEWORK.md`](../../docs/MODEL_TUNING_FRAMEWORK.md)。

## 子模块

| 路径 | 职责 |
|------|------|
| `configs/` | 训练 / 评测 YAML |
| `data/` | `GUITaskSample` → SFT/DPO 转换与过滤 |
| `train/` | LoRA 训练与 adapter 导出 |
| `eval/` | 离线指标与在线 dry-run |

## 计划命令（落地后）

```bash
# 1) 原始数据预处理（已有）
python script/preprocess_all.py

# 2) 转为 SFT JSONL
python -m src.tuning.data.sft_converter \
  --input data/processed/screenagent/train.jsonl \
  --output data/sft/train.jsonl

# 3) QLoRA 训练
python -m src.tuning.train.train_lora \
  --config src/tuning/configs/default_qlora.yaml

# 4) 离线评测
python -m src.tuning.eval.offline_metrics \
  --split data/sft/validation.jsonl \
  --adapter data/adapters/<run_id>
```

当前阶段仅建立框架与占位实现，训练脚本尚未可运行。
