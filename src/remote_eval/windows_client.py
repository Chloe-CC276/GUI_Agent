"""Windows client: real screenshot + OCR + Executor, Colab does VLM only."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.executor.executor import Executor
from src.perception.screen_capture import ScreenCapture
from src.remote_eval.protocol import PlanRequest, VerifyRequest
from src.remote_eval.safety import SafetyGuard, install_sigint_abort

LOGGER = logging.getLogger("remote_eval.windows_client")


def _encode_bgr_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _http_json(url: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _try_focus_window(title_substr: str | None) -> str | None:
    """Best-effort: bring a window whose title contains title_substr to foreground."""

    if not title_substr:
        return None
    try:
        import pygetwindow as gw  # type: ignore
    except ImportError:
        LOGGER.warning("pygetwindow not installed — skip window focus")
        return None
    matches = [w for w in gw.getAllWindows() if title_substr.lower() in (w.title or "").lower()]
    if not matches:
        LOGGER.warning("No window matching %r", title_substr)
        return None
    win = matches[0]
    try:
        if win.isMinimized:
            win.restore()
        win.activate()
        time.sleep(0.4)
        return win.title
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("activate window failed: %s", exc)
        return win.title


def _optional_perception(image: np.ndarray) -> tuple[str, list[dict[str, Any]]]:
    """Run local OCR/GUI merge when PerceptionPipeline is available; else empty."""

    try:
        from src.perception.perception_pipeline import PerceptionPipeline

        pipe = PerceptionPipeline()
        result = pipe.process_image(image)
        ocr_text = getattr(result, "ocr_text", None) or ""
        if not ocr_text and getattr(result, "ocr_items", None):
            parts = []
            for item in result.ocr_items or []:
                t = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else None)
                if t:
                    parts.append(str(t))
            ocr_text = "\n".join(parts)
        elements: list[dict[str, Any]] = []
        for el in getattr(result, "merged_elements", None) or []:
            if hasattr(el, "to_dict"):
                elements.append(dict(el.to_dict()))
            elif isinstance(el, dict):
                elements.append(dict(el))
            else:
                elements.append(
                    {
                        "element_id": getattr(el, "element_id", None),
                        "text": getattr(el, "text", None),
                        "label": getattr(el, "label", None),
                        "name": getattr(el, "name", None),
                        "bbox": getattr(el, "bbox", None),
                        "center": getattr(el, "center", None),
                        "control_type": getattr(el, "control_type", None)
                        or getattr(el, "element_type", None),
                    }
                )
        return str(ocr_text), elements
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("local perception skipped: %s", exc)
        return "", []


def run_task(
    *,
    server: str,
    task: str,
    variant: str,
    max_steps: int,
    countdown: float,
    require_confirm: bool,
    allow_real_actions: bool,
    window_title: str | None,
    post_action_wait: float,
    out_jsonl: Path | None,
) -> dict[str, Any]:
    server = server.rstrip("/")
    session_id = uuid.uuid4().hex
    guard = SafetyGuard(
        countdown_seconds=countdown,
        require_confirm=require_confirm,
    )
    install_sigint_abort(guard)

    focused = _try_focus_window(window_title)
    print(
        f"[prep] session={session_id} variant={variant}\n"
        f"  focused_window={focused!r}\n"
        f"  Ensure the TARGET app/page is visible on the primary monitor.\n"
        f"  PyAutoGUI FAILSAFE: yank mouse to a screen corner to abort.\n"
        f"  real_actions={allow_real_actions}",
        flush=True,
    )
    input("Press Enter when the correct page/window is ready...")

    capture = ScreenCapture()
    executor = Executor(dry_run=not allow_real_actions, raise_on_error=False)
    history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for step in range(max_steps):
        if guard.aborted():
            break
        _try_focus_window(window_title)
        before = capture.capture_screen()
        h, w = before.shape[:2]
        ocr_text, elements = _optional_perception(before)
        before_b64 = _encode_bgr_png(before)

        plan_req = PlanRequest(
            task=task,
            image_b64=before_b64,
            screen_width=w,
            screen_height=h,
            variant=variant,
            session_id=session_id,
            step_index=step,
            ocr_text=ocr_text,
            gui_elements=elements,
            window_title=focused or window_title,
            history=history[-5:],
        )
        print(f"\n[step {step}] POST {server}/v1/plan ...", flush=True)
        try:
            plan = _http_json(f"{server}/v1/plan", plan_req.to_dict())
        except urllib.error.URLError as exc:
            raise SystemExit(f"Cannot reach Colab server: {exc}") from exc

        decision = str(plan.get("decision") or "").lower()
        action = plan.get("action")
        print(
            f"[step {step}] decision={decision} schema={plan.get('schema_valid')} "
            f"lat={plan.get('latency_seconds')} action={action}",
            flush=True,
        )
        rec = {
            "session_id": session_id,
            "step": step,
            "plan": plan,
            "executed": False,
            "verify": None,
        }

        if decision in {"finish", "fail"} or action is None:
            records.append(rec)
            history.append({"decision": decision, "action": action})
            break

        summary = json.dumps(action, ensure_ascii=False)[:200]
        if not guard.before_action(summary):
            rec["aborted"] = True
            records.append(rec)
            break

        exec_result = executor.execute(action)
        ok = bool(getattr(exec_result, "succeeded", False) or getattr(exec_result, "ok", False))
        if hasattr(exec_result, "status"):
            st = getattr(exec_result.status, "value", exec_result.status)
            ok = ok or str(st).lower() == "success"
        rec["executed"] = True
        rec["exec_ok"] = ok
        print(f"[step {step}] executed ok={ok}", flush=True)
        time.sleep(max(0.0, post_action_wait))

        after = capture.capture_screen()
        after_b64 = _encode_bgr_png(after)
        ocr2, elements2 = _optional_perception(after)
        verify_req = VerifyRequest(
            task=task,
            after_image_b64=after_b64,
            before_image_b64=before_b64,
            last_action=action if isinstance(action, dict) else {"raw": action},
            variant=variant,
            session_id=session_id,
            screen_width=w,
            screen_height=h,
            ocr_text=ocr2,
            gui_elements=elements2,
            step_index=step,
        )
        print(f"[step {step}] POST {server}/v1/verify ...", flush=True)
        try:
            verify = _http_json(f"{server}/v1/verify", verify_req.to_dict())
        except urllib.error.URLError as exc:
            verify = {"ok": False, "error": str(exc), "recommended_next": "fail"}
        rec["verify"] = verify
        records.append(rec)
        history.append(
            {
                "decision": decision,
                "action": action,
                "verify": {
                    "status": verify.get("status"),
                    "task_complete": verify.get("task_complete"),
                    "recommended_next": verify.get("recommended_next"),
                },
            }
        )
        print(
            f"[step {step}] verify status={verify.get('status')} "
            f"complete={verify.get('task_complete')} next={verify.get('recommended_next')}",
            flush=True,
        )

        nxt = str(verify.get("recommended_next") or "continue").lower()
        if verify.get("task_complete") is True or nxt == "finish":
            break
        if nxt == "fail":
            break
        # retry/replan/continue → next loop

    summary = {
        "session_id": session_id,
        "task": task,
        "variant": variant,
        "steps": len(records),
        "records": records,
        "allow_real_actions": allow_real_actions,
    }
    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"Wrote {out_jsonl}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True, help="Colab public HTTPS base URL")
    p.add_argument("--task", required=True, help="Natural-language task instruction")
    p.add_argument(
        "--variant",
        default="adapter_v1_optimized",
        choices=["adapter_v1_original", "adapter_v1_optimized"],
    )
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--countdown", type=float, default=3.0)
    p.add_argument("--require-confirm", action="store_true")
    p.add_argument(
        "--allow-real-actions",
        action="store_true",
        help="Disable Executor dry_run (REAL mouse/keyboard)",
    )
    p.add_argument(
        "--window-title",
        default=None,
        help="Substring of target window title to focus before each step",
    )
    p.add_argument("--post-action-wait", type=float, default=0.8)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/gui_agent_evaluation/raw/remote_e2e_results.jsonl"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    if not args.allow_real_actions:
        print(
            "WARNING: running with Executor dry_run=True "
            "(pass --allow-real-actions for real clicks)",
            flush=True,
        )
    run_task(
        server=args.server,
        task=args.task,
        variant=args.variant,
        max_steps=args.max_steps,
        countdown=args.countdown,
        require_confirm=args.require_confirm,
        allow_real_actions=args.allow_real_actions,
        window_title=args.window_title,
        post_action_wait=args.post_action_wait,
        out_jsonl=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
