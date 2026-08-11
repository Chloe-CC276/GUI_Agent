"""Build Planner + VLM stacks for experiment variants A/B/C."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from src.agent.planner import Planner, PlannerConfig
from src.agent.prompts import PromptBuilder, PromptConfig, PromptKind, PromptLanguage
from src.evaluation.adapter_check import AdapterStatus, check_adapter
from src.evaluation.config import ExperimentConfig, VariantConfig
from src.evaluation.optimized_rules import optimized_rules


@dataclass
class VariantRuntime:
    variant: VariantConfig
    prompt_version: str
    base_model: str
    adapter_path: str | None
    adapter_loaded: bool
    adapter_status: AdapterStatus | None
    planner: Planner | None
    vlm: Any | None
    status: str  # ready | blocked
    blocked_reason: str | None = None


class ScriptedVLM:
    """Deterministic VLM for unit / dry pipeline tests (not a real model)."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls = 0
        self.model = "scripted-vlm"

    def _pick_from_prompt(self, prompt: str) -> dict[str, Any]:
        text = prompt or ""
        if "关闭" in text or "close" in text.lower():
            return {
                "decision": "act",
                "action": {
                    "type": "hotkey",
                    "parameters": {"keys": ["ctrl", "w"]},
                },
                "reason": "scripted close via Ctrl+W",
                "confidence": 0.9,
                "observation_summary": "word document visible",
                "goal_progress": "closing document",
            }
        if "搜索" in text or "search" in text.lower():
            return {
                "decision": "act",
                "action": {
                    "type": "hotkey",
                    "parameters": {"keys": ["ctrl", "l"]},
                },
                "reason": "scripted focus address bar",
                "confidence": 0.9,
                "observation_summary": "browser visible",
                "goal_progress": "focus address bar",
            }
        if "发送" in text or "消息" in text or "message" in text.lower():
            return {
                "decision": "act",
                "action": {
                    "type": "click",
                    "parameters": {"target_text": "Message"},
                },
                "reason": "scripted focus chat input",
                "confidence": 0.8,
                "observation_summary": "chat window",
                "goal_progress": "focus input",
            }
        if "打开" in text and (
            "文件" in text
            or "file" in text.lower()
            or "PDF" in text
            or ".xlsx" in text.lower()
            or "物料" in text
        ):
            target = "ERP物料主数据_100条.xlsx"
            if "ERP物料主数据_100条.xlsx" in text:
                target = "ERP物料主数据_100条.xlsx"
            return {
                "decision": "act",
                "action": {
                    "type": "double_click",
                    "parameters": {"target_text": target},
                },
                "reason": "scripted open file",
                "confidence": 0.85,
                "observation_summary": "desktop icons",
                "goal_progress": "open file",
            }
        if "浏览器" in text or "browser" in text.lower() or "Edge" in text:
            return {
                "decision": "act",
                "action": {
                    "type": "double_click",
                    "parameters": {"target_text": "Microsoft Edge"},
                },
                "reason": "scripted open browser",
                "confidence": 0.9,
                "observation_summary": "desktop",
                "goal_progress": "open browser",
            }
        return {
            "decision": "act",
            "action": {
                "type": "click",
                "parameters": {"target_text": "Microsoft Edge"},
            },
            "reason": "scripted default",
            "confidence": 0.5,
            "observation_summary": "scripted",
            "goal_progress": "scripted",
        }

    def generate_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.responses:
            return self.responses[(self.calls - 1) % len(self.responses)]
        prompt = str(kwargs.get("prompt") or (args[0] if args else ""))
        return self._pick_from_prompt(prompt)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        import json

        class _Resp:
            def __init__(self, text: str) -> None:
                self.text = text
                self.usage = None

        return _Resp(json.dumps(self.generate_json(*args, **kwargs), ensure_ascii=False))


def build_prompt_config(exp: ExperimentConfig, variant: VariantConfig) -> PromptConfig:
    lang = PromptLanguage.ZH if exp.language.lower().startswith("zh") else PromptLanguage.EN
    extra: tuple[str, ...] = ()
    if variant.prompt_version == "optimized":
        extra = optimized_rules(exp.language)
    return PromptConfig(
        language=lang,
        extra_rules=extra,
        metadata={"prompt_version": variant.prompt_version, "variant": variant.name},
    )


def build_variant_runtime(
    exp: ExperimentConfig,
    variant: VariantConfig,
    *,
    vlm_factory: Callable[[], Any] | None = None,
    scripted: bool = False,
) -> VariantRuntime:
    prompt_cfg = build_prompt_config(exp, variant)
    builder = PromptBuilder(prompt_cfg)
    adapter_status: AdapterStatus | None = None
    adapter_loaded = False
    adapter_path = variant.adapter_path
    base_model = variant.model_name or exp.cloud_model

    if variant.model_type == "local_adapter":
        adapter_status = check_adapter(adapter_path)
        base_model = adapter_status.base_model_name or exp.local_base_model
        if not adapter_status.ready:
            return VariantRuntime(
                variant=variant,
                prompt_version=variant.prompt_version,
                base_model=base_model,
                adapter_path=adapter_status.path,
                adapter_loaded=False,
                adapter_status=adapter_status,
                planner=None,
                vlm=None,
                status="blocked",
                blocked_reason=adapter_status.blocked_reason,
            )
        if not exp.allow_local_adapter:
            return VariantRuntime(
                variant=variant,
                prompt_version=variant.prompt_version,
                base_model=base_model,
                adapter_path=adapter_status.path,
                adapter_loaded=False,
                adapter_status=adapter_status,
                planner=None,
                vlm=None,
                status="blocked",
                blocked_reason=(
                    "local adapter execution disabled "
                    "(set experiment.allow_local_adapter=true after approval; GPU risk on laptop)"
                ),
            )
        if vlm_factory is not None:
            vlm = vlm_factory()
            adapter_loaded = True
        else:
            try:
                from src.model.local_peft_vlm import LocalPeftVLM

                vlm = LocalPeftVLM(
                    base_model=base_model,
                    adapter_path=adapter_status.path,
                )
                adapter_loaded = True
            except Exception as exc:  # noqa: BLE001 — surface load failures as blocked
                return VariantRuntime(
                    variant=variant,
                    prompt_version=variant.prompt_version,
                    base_model=base_model,
                    adapter_path=adapter_status.path,
                    adapter_loaded=False,
                    adapter_status=adapter_status,
                    planner=None,
                    vlm=None,
                    status="blocked",
                    blocked_reason=f"local PEFT load failed: {exc}",
                )
        return _ready_runtime(
            variant=variant,
            prompt_version=variant.prompt_version,
            base_model=base_model,
            adapter_path=adapter_status.path,
            adapter_loaded=adapter_loaded,
            adapter_status=adapter_status,
            vlm=vlm,
            prompt_builder=builder,
        )

    # original / cloud path
    if scripted or vlm_factory is None and not exp.allow_cloud_vlm:
        if scripted or vlm_factory is not None:
            vlm = vlm_factory() if vlm_factory else ScriptedVLM()
        else:
            return VariantRuntime(
                variant=variant,
                prompt_version=variant.prompt_version,
                base_model=base_model,
                adapter_path=None,
                adapter_loaded=False,
                adapter_status=None,
                planner=None,
                vlm=None,
                status="blocked",
                blocked_reason=(
                    "cloud VLM disabled (set allow_cloud_vlm=true or pass --allow-cloud-vlm); "
                    "use --scripted for non-API pipeline checks"
                ),
            )
    else:
        if vlm_factory is not None:
            vlm = vlm_factory()
        else:
            if not os.environ.get("DASHSCOPE_API_KEY"):
                return VariantRuntime(
                    variant=variant,
                    prompt_version=variant.prompt_version,
                    base_model=base_model,
                    adapter_path=None,
                    adapter_loaded=False,
                    adapter_status=None,
                    planner=None,
                    vlm=None,
                    status="blocked",
                    blocked_reason="DASHSCOPE_API_KEY not set",
                )
            from src.model.qwen_vlm import QwenVLM

            vlm = QwenVLM(
                model=variant.model_name or exp.cloud_model,
                region=exp.region,
            )

    return _ready_runtime(
        variant=variant,
        prompt_version=variant.prompt_version,
        base_model=base_model,
        adapter_path=None,
        adapter_loaded=False,
        adapter_status=None,
        vlm=vlm,
        prompt_builder=builder,
    )


def _ready_runtime(
    *,
    variant: VariantConfig,
    prompt_version: str,
    base_model: str,
    adapter_path: str | None,
    adapter_loaded: bool,
    adapter_status: AdapterStatus | None,
    vlm: Any,
    prompt_builder: PromptBuilder,
) -> VariantRuntime:
    from src.agent.cli import _planner_action_factory

    planner = Planner(
        vlm,
        # One attempt only (no planner error retry) for faster live eval feedback.
        config=PlannerConfig(max_attempts=1),
        prompt_builder=prompt_builder,
        # Use CLI factory so actions carry Executor-compatible "type".
        action_factory=_planner_action_factory,
    )
    return VariantRuntime(
        variant=variant,
        prompt_version=prompt_version,
        base_model=getattr(vlm, "model", base_model),
        adapter_path=adapter_path,
        adapter_loaded=adapter_loaded,
        adapter_status=adapter_status,
        planner=planner,
        vlm=vlm,
        status="ready",
        blocked_reason=None,
    )


def assert_original_rules_intact() -> None:
    """Guard: stock planner rules must still be the baseline constants."""

    from src.agent.prompts.planner_prompt import PLANNER_RULES_ZH

    assert any("target_text" in rule for rule in PLANNER_RULES_ZH)
    assert not any("自动保存：关" in rule for rule in PLANNER_RULES_ZH)
