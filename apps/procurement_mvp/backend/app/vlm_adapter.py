"""Phase-2 pluggable VLM adapter for ERP PO Agent.

Enable with ERP_PO_VLM_ENABLED=1 (default on for Phase 2).
Stable RPA success skips VLM unless force=True. Low confidence can gate save.
"""

from __future__ import annotations

import os
import time
from typing import Any


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class VLMAdapter:
    """Pluggable VLM enhancement used only at GUI-fragile nodes."""

    LOW_CONFIDENCE = 0.55

    def __init__(self) -> None:
        self.enabled: bool = _env_flag("ERP_PO_VLM_ENABLED", default=True)
        self.phase: str = "phase2_vlm_enhance" if self.enabled else "phase1_rpa_only"
        self.model_name: str = os.getenv("ERP_PO_VLM_MODEL", "heuristic-advisor-v1")

    def maybe_call(
        self,
        *,
        scenario: str,
        payload: dict[str, Any] | None = None,
        force: bool = False,
        rpa_ok: bool | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        if rpa_ok is True and not force:
            return self._skip(scenario, "rpa_ok_skip_vlm")
        if not self.enabled and not force:
            return self._skip(scenario, "vlm_disabled")

        started = time.perf_counter()
        result = self._heuristic(scenario, payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        confidence = float(result.get("confidence") or 0.0)
        low = confidence < self.LOW_CONFIDENCE
        block_save = bool(result.get("block_save")) or (
            low and scenario in {"pre_save_visual", "anomaly_recover", "po_readback_ocr"}
        )
        return {
            "called": True,
            "skipped": False,
            "reason": None,
            "scenario": scenario,
            "confidence": confidence,
            "suggestion": result.get("suggestion"),
            "targets": result.get("targets") or [],
            "page_state": result.get("page_state"),
            "model": self.model_name,
            "latency_ms": latency_ms,
            "low_confidence": low,
            "block_save": block_save,
        }

    def _skip(self, scenario: str, reason: str) -> dict[str, Any]:
        return {
            "called": False,
            "skipped": True,
            "reason": reason,
            "scenario": scenario,
            "confidence": None,
            "suggestion": None,
            "targets": [],
            "page_state": None,
            "model": self.model_name,
            "latency_ms": 0,
            "low_confidence": False,
            "block_save": False,
        }

    def _heuristic(self, scenario: str, payload: dict[str, Any]) -> dict[str, Any]:
        route = str(payload.get("route") or "")
        error = str(payload.get("error") or payload.get("error_code") or "")
        if scenario == "page_understand":
            if "/erp/po-create" in route:
                return {
                    "confidence": 0.92,
                    "page_state": "po_draft_form",
                    "suggestion": "ERP PO draft page detected; continue fill/verify/save.",
                }
            if "/erp/po-candidates" in route:
                return {
                    "confidence": 0.9,
                    "page_state": "po_candidate_list",
                    "suggestion": "Waiting-PO list detected; create batch then open draft.",
                }
            if "/erp/pos/" in route:
                return {
                    "confidence": 0.9,
                    "page_state": "po_detail",
                    "suggestion": "PO detail page detected.",
                }
            return {
                "confidence": 0.62,
                "page_state": "unknown",
                "suggestion": "Page unknown; prefer WAIT_USER if RPA cannot locate controls.",
            }
        if scenario == "locator_recover":
            testid = payload.get("testid") or "erp-po-create-button"
            return {
                "confidence": 0.78 if testid else 0.4,
                "suggestion": f"Retry click on stable testid={testid}",
                "targets": [{"testid": testid, "action": "click"}],
            }
        if scenario == "pre_save_visual":
            passed = bool(payload.get("rule_passed", True))
            return {
                "confidence": 0.88 if passed else 0.45,
                "suggestion": "No blocking visual error inferred." if passed else "Visual/rule risk; wait user.",
                "block_save": not passed,
            }
        if scenario == "anomaly_recover":
            return {
                "confidence": 0.5 if error else 0.7,
                "suggestion": "Close unexpected dialog / re-open draft / retry fill." if error else "Continue.",
                "targets": [{"testid": "erp-po-verify-button", "action": "click"}],
            }
        if scenario == "po_readback_ocr":
            po_no = payload.get("po_no")
            return {
                "confidence": 0.9 if po_no else 0.35,
                "suggestion": f"PO readback {po_no}" if po_no else "po_no missing; WAIT_USER",
                "block_save": not bool(po_no),
            }
        if scenario == "semantic_candidate":
            return {
                "confidence": 0.8,
                "suggestion": "Prefer exact supplier_code / material_code match.",
                "targets": [],
            }
        return {
            "confidence": 0.6,
            "suggestion": f"No specialized handler for scenario={scenario}",
        }


vlm_adapter = VLMAdapter()
