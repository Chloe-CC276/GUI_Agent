"""Colab / GPU inference HTTP server for dual-mode GUI Agent eval.

Endpoints:
  GET  /health
  POST /v1/plan
  POST /v1/verify

Run on Colab (after model load env is set):
  uvicorn src.remote_eval.colab_server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from src.agent.prompts import PromptKind
from src.agent.prompts.schemas import VERIFY_RESPONSE_SCHEMA
from src.agent.result import ResultStatus, ToolResult
from src.remote_eval.observation import (
    action_to_executor_dict,
    decode_image_b64,
    observation_from_remote,
    state_from_remote_plan,
)
from src.remote_eval.protocol import (
    PlanRequest,
    PlanResponse,
    VerifyRequest,
    VerifyResponse,
)
from src.remote_eval.runtime_cache import get_runtime

LOGGER = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Install FastAPI for Colab server: pip install fastapi uvicorn"
    ) from exc


app = FastAPI(title="GUI Agent Remote Inference", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "gui-agent-remote-inference"}


@app.post("/v1/plan")
def plan(payload: dict[str, Any]) -> dict[str, Any]:
    req = PlanRequest.from_dict(payload)
    if not req.task.strip():
        raise HTTPException(400, "task is required")
    if not req.image_b64.strip():
        raise HTTPException(400, "image_b64 is required")

    session_id = req.session_id or uuid.uuid4().hex
    t0 = time.perf_counter()
    image_path = None
    try:
        runtime = get_runtime(req.variant)
        image_path = decode_image_b64(req.image_b64)
        state = state_from_remote_plan(
            task=req.task,
            image_path=image_path,
            screen_width=req.screen_width,
            screen_height=req.screen_height,
            ocr_text=req.ocr_text,
            gui_elements=req.gui_elements,
            window_title=req.window_title,
            application_name=req.application_name,
            step_index=req.step_index,
        )
        assert runtime.planner is not None
        result = runtime.planner.plan(state)
        latency = time.perf_counter() - t0
        decision = str(
            getattr(result.decision, "value", result.decision) or ""
        ).lower()
        schema_valid = (
            result.status.value == "success"
            if hasattr(result.status, "value")
            else result.error is None
        )
        action = action_to_executor_dict(result.action)
        tokens = None
        usage = getattr(result, "usage", None)
        if usage is not None:
            tokens = getattr(usage, "total_tokens", None)
        return PlanResponse(
            ok=True,
            session_id=session_id,
            decision=decision or "unknown",
            action=action,
            reason=getattr(result, "reason", None),
            confidence=getattr(result, "confidence", None),
            schema_valid=bool(schema_valid),
            latency_seconds=latency,
            total_tokens=tokens,
            raw_output=getattr(result, "raw_output", None),
            error=None if schema_valid else str(getattr(result, "error", "") or ""),
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("plan failed")
        return PlanResponse(
            ok=False,
            session_id=session_id,
            decision="fail",
            error=f"{type(exc).__name__}: {exc}",
            latency_seconds=time.perf_counter() - t0,
            schema_valid=False,
        ).to_dict()
    finally:
        if image_path is not None:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


@app.post("/v1/verify")
def verify(payload: dict[str, Any]) -> dict[str, Any]:
    req = VerifyRequest.from_dict(payload)
    if not req.task.strip():
        raise HTTPException(400, "task is required")
    if not req.after_image_b64.strip():
        raise HTTPException(400, "after_image_b64 is required")

    session_id = req.session_id or uuid.uuid4().hex
    t0 = time.perf_counter()
    paths: list[Any] = []
    try:
        runtime = get_runtime(req.variant)
        after_path = decode_image_b64(req.after_image_b64)
        paths.append(after_path)
        before_path = None
        if req.before_image_b64:
            before_path = decode_image_b64(req.before_image_b64)
            paths.append(before_path)

        # Build state: before → after so previous_observation is set
        state = state_from_remote_plan(
            task=req.task,
            image_path=before_path or after_path,
            screen_width=req.screen_width,
            screen_height=req.screen_height,
            ocr_text=req.ocr_text,
            gui_elements=req.gui_elements,
            step_index=req.step_index,
        )
        if before_path is not None:
            after_obs = observation_from_remote(
                image_path=after_path,
                screen_width=req.screen_width,
                screen_height=req.screen_height,
                ocr_text=req.ocr_text,
                gui_elements=req.gui_elements,
            )
            state.update_observation(after_obs)

        # Attach last action for verify template
        from types import SimpleNamespace

        action_obj = SimpleNamespace(
            **(
                req.last_action
                if isinstance(req.last_action, dict)
                else {"raw": req.last_action}
            )
        )
        # Provide attributes commonly read by verify prompt
        if isinstance(req.last_action, dict):
            for k, v in req.last_action.items():
                setattr(action_obj, k, v)
        fake_planner = SimpleNamespace(
            action=req.last_action,
            decision="act",
            reason="remote_executed",
            confidence=1.0,
            is_finished=False,
            should_execute=True,
            should_retry=False,
            error=None,
            usage=None,
            status=SimpleNamespace(value="success"),
        )
        state.last_planner_result = fake_planner  # type: ignore[assignment]
        state.last_execution_result = ToolResult(
            tool_name="execute_action",
            status=ResultStatus.SUCCESS,
            message="executed_on_windows_client",
            output={"remote": True},
        )

        assert runtime.planner is not None
        builder = runtime.planner.prompt_builder
        prompt = builder.build_text(PromptKind.VERIFY, state)
        vlm = runtime.vlm
        # Pass after screenshot to VLM
        from src.model.base_vlm import ImageInput

        images = [ImageInput.from_path(str(after_path))]
        parsed: dict[str, Any] = {}
        tokens = None
        if hasattr(vlm, "generate_json"):
            out = vlm.generate_json(
                prompt,
                images=images,
                schema=VERIFY_RESPONSE_SCHEMA,
            )
            # LocalPeft / Qwen may return (dict, response) or dict
            if isinstance(out, tuple):
                parsed = dict(out[0] or {})
                resp = out[1]
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    tokens = getattr(usage, "total_tokens", None)
            elif isinstance(out, dict):
                parsed = out
            else:
                parsed = {}
        else:
            text = vlm.generate(prompt, images=images)
            import json

            parsed = json.loads(str(getattr(text, "text", text)))

        from src.agent.agent_chain import _coerce_verify_data

        data = _coerce_verify_data(parsed)
        latency = time.perf_counter() - t0
        return VerifyResponse(
            ok=True,
            session_id=session_id,
            status=str(data.get("status") or "uncertain"),
            action_effective=data.get("action_effective"),
            task_complete=data.get("task_complete"),
            recommended_next=str(data.get("recommended_next") or "continue"),
            reason=data.get("reason"),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
            evidence=list(data.get("evidence") or []),
            latency_seconds=latency,
            total_tokens=tokens,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("verify failed")
        return VerifyResponse(
            ok=False,
            session_id=session_id,
            status="uncertain",
            recommended_next="replan",
            error=f"{type(exc).__name__}: {exc}",
            latency_seconds=time.perf_counter() - t0,
        ).to_dict()
    finally:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.remote_eval.colab_server:app",
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
