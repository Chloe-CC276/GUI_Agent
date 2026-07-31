"""
document_tasks
Task semantics for opening and closing desktop documents.

Opening: the document is open once the title bar shows the requested file name
next to a Word signature.

Closing: the title-bar close control is usually an unlabeled cross, so the
planner drifts onto menu labels such as 「文件」or status chips such as 「关」.
Close clicks are therefore a whitelist (✕ / × / 关闭 / Close in the title-bar
right strip of the target window). Illegal clicks are rejected; Ctrl+W / Alt+F4
are allowed only after the target window is confirmed active. Multi-window
requests use activate → close.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any

from src.common.target_validation import (
    CLICK_ACTION_TYPES,
    flatten_action_evidence,
    normalise_target_text,
)
from src.executor import win32_windows

from .observation_utils import (
    BoundingBox,
    iter_labeled_boxes,
    iter_texts,
    observation_counts,
    screen_size,
)
from .state import ObservationSource, ObservationState
from .verify_policy import latest_action_type

# Window title band: the caption strip at the very top of the screen.
_TITLE_BAR_MAX_RATIO: float = 0.06
_TITLE_BAR_MIN_PX: float = 48.0

# The close (✕) control sits in the rightmost strip of that caption.
_CLOSE_STRIP_RATIO: float = 0.12
_CLOSE_STRIP_MIN_PX: float = 72.0

# Ribbon tabs sit just below the caption strip.
_RIBBON_BAND_RATIO: float = 0.20
_MIN_RIBBON_TABS: int = 3

_OPEN_VERB_MARKERS: tuple[str, ...] = ("打开", "开启", "启动", "open", "launch")
_WORD_TASK_MARKERS: tuple[str, ...] = ("word", "文档", "docx", "doc")
_PDF_TASK_MARKERS: tuple[str, ...] = ("pdf",)

# Normalised tokens that describe the request instead of naming the file.
_TASK_STOPWORDS: frozenset[str] = frozenset(
    {
        "打开", "开启", "启动", "关闭", "关掉", "退出", "双击", "点击", "请",
        "帮我", "一下", "的", "这个", "那个", "桌面", "文件", "文档", "表格",
        "程序", "应用", "左侧", "右边", "右侧", "左边", "左", "右",
        "open", "launch", "start", "close", "quit", "exit", "double", "click",
        "doubleclick", "please", "the", "a", "an", "file", "document",
        "desktop", "left", "right", "word", "microsoft", "office", "docx", "doc",
    }
)

_WORD_TITLE_MARKERS: tuple[str, ...] = (
    "microsoftword",
    "word",
    "已保存到这台电脑",
    "保存到这台电脑",
    "自动保存",
    "savedtothispc",
    "autosave",
)

_WORD_RIBBON_TABS: frozenset[str] = frozenset(
    {"文件", "开始", "插入", "设计", "布局", "引用", "邮件", "审阅", "视图", "帮助"}
)

# Excel and PowerPoint share most ribbon tabs, so the fallback also needs a tab
# that only Word shows.
_WORD_ONLY_RIBBON_TABS: frozenset[str] = frozenset({"引用", "邮件"})

_TOKEN_PATTERN = re.compile(r"[0-9a-zA-Z]+|[\u4e00-\u9fff]+")

# ----------------------------------------------------------------------
# Close requests
# ----------------------------------------------------------------------

CLOSE_PROGRESS_KEY: str = "close_progress"
PHASE_ACTIVATE: str = "activate"
PHASE_CLOSE: str = "close"

_CLOSE_VERB_MARKERS: tuple[str, ...] = ("关闭", "关掉", "退出", "close", "quit", "exit")
_WINDOW_LEVEL_MARKERS: tuple[str, ...] = (
    "窗口", "程序", "应用", "退出", "window", "app", "quit", "exit",
)
_LEFT_MARKERS: tuple[str, ...] = ("左侧", "左边", "左半", "left")
_RIGHT_MARKERS: tuple[str, ...] = ("右侧", "右边", "右半", "right")

_DOCUMENT_CLOSE_KEYS: tuple[str, ...] = ("ctrl", "w")
_WINDOW_CLOSE_KEYS: tuple[str, ...] = ("alt", "f4")

# Cross glyphs survive no normalisation, so they are matched raw.
_CLOSE_GLYPHS: frozenset[str] = frozenset({"✕", "×", "╳", "⨯", "x", "X"})
_CLOSE_CONTROL_LABELS: frozenset[str] = frozenset(
    {"关闭", "关闭窗口", "close", "closewindow"}
)

# Autosave / toggle chips that look like "close" but are not the ✕ control.
# Keep this list tight: Word title chips such as 「已保存到这台电脑」must remain
# clickable during the activate phase.
_STATUS_DECOYS: frozenset[str] = frozenset({"关", "开", "on", "off", "自动保存", "autosave"})

# Menu and ribbon labels that only open panels.
_MENU_LABELS: frozenset[str] = frozenset(
    {
        "菜单", "开始", "插入", "设计", "布局", "引用", "邮件", "审阅", "视图",
        "帮助", "绘图", "搜索",
        "menu", "home", "insert", "design", "layout", "references", "mailings",
        "review", "view", "help", "search", "tellme",
    }
)

# Shared Word chrome that must not be used as a document identity fingerprint.
_GENERIC_TITLE_TOKENS: frozenset[str] = frozenset(
    {
        "word", "microsoft", "microsoftword", "office", "docx", "doc",
        "自动保存", "autosave", "已保存到这台电脑", "保存到这台电脑",
        "savedtothispc", "compatibilitymode", "兼容模式", "文档1", "document1",
        "untitled", "未命名",
    }
)

# Element-count drop large enough to treat a close hotkey as "screen changed".
_CLOSE_ELEMENT_DROP: int = 30


def task_instruction(state: Any) -> str:
    """Return the task instruction text carried by an AgentState."""

    task = getattr(state, "task", None)
    if isinstance(task, str):
        return task.strip()
    if isinstance(task, Mapping):
        return str(task.get("instruction") or "").strip()
    return str(getattr(task, "instruction", "") or "").strip()


def wants_word_document(instruction: Any) -> bool:
    """Report whether the task asks to open a Word document (and not a PDF)."""

    text = normalise_target_text(instruction)
    if not text:
        return False
    if any(marker in text for marker in _PDF_TASK_MARKERS):
        return False
    if not any(marker in text for marker in _OPEN_VERB_MARKERS):
        return False
    return any(marker in text for marker in _WORD_TASK_MARKERS)


def target_name_candidates(instruction: Any) -> list[str]:
    """Return normalised tokens of the instruction that can name the file."""

    candidates: list[str] = []
    for token in _TOKEN_PATTERN.findall(str(instruction or "")):
        normalised = normalise_target_text(token)
        if len(normalised) < 2 or normalised in _TASK_STOPWORDS:
            continue
        if normalised not in candidates:
            candidates.append(normalised)
    return candidates


def _title_bar_limit(height: float) -> float:
    return max(_TITLE_BAR_MIN_PX, height * _TITLE_BAR_MAX_RATIO)


def _title_bar_texts(observation: Any) -> list[str]:
    """Return normalised texts from the caption strip and the window title."""

    _width, height = screen_size(observation)
    limit = _title_bar_limit(height)

    texts: list[str] = []
    for attribute in ("window_title", "application_name"):
        normalised = normalise_target_text(getattr(observation, attribute, None))
        if normalised:
            texts.append(normalised)

    for text, bbox in iter_labeled_boxes(observation):
        _left, top, _right, bottom = bbox
        if (top + bottom) / 2.0 > limit:
            continue
        normalised = normalise_target_text(text)
        if normalised:
            texts.append(normalised)
    return texts


def _ribbon_tab_hits(observation: Any) -> list[str]:
    """Return the Word ribbon tabs visible in the band below the title bar."""

    _width, height = screen_size(observation)
    limit = height * _RIBBON_BAND_RATIO

    hits: list[str] = []
    for text, bbox in iter_labeled_boxes(observation):
        _left, top, _right, bottom = bbox
        if (top + bottom) / 2.0 > limit:
            continue
        normalised = normalise_target_text(text)
        if normalised in _WORD_RIBBON_TABS and normalised not in hits:
            hits.append(normalised)
    return hits


def detect_opened_document(
    observation: Any,
    instruction: Any,
) -> tuple[bool, list[str]]:
    """Detect the requested document open in Word: file name + Word signature."""

    if observation is None:
        return False, ["No observation available."]
    if not wants_word_document(instruction):
        return False, ["Task is not a Word document open request."]

    candidates = target_name_candidates(instruction)
    if not candidates:
        return False, ["Task does not name a file to open."]

    title_texts = _title_bar_texts(observation)
    if not title_texts:
        return False, ["No title-bar text detected."]

    name_hit = next(
        (
            (candidate, title)
            for candidate in candidates
            for title in title_texts
            if candidate in title
        ),
        None,
    )
    if name_hit is None:
        return False, [
            f"Title bar does not contain the requested name {candidates[0]!r}."
        ]

    candidate, title = name_hit
    evidence = [f"Title bar {title!r} contains the requested name {candidate!r}."]

    word_marker = next(
        (
            marker
            for title_text in title_texts
            for marker in _WORD_TITLE_MARKERS
            if marker in title_text
        ),
        None,
    )
    if word_marker is not None:
        evidence.append(f"Word signature {word_marker!r} found in the title bar.")
        return True, evidence

    ribbon_hits = _ribbon_tab_hits(observation)
    if len(ribbon_hits) >= _MIN_RIBBON_TABS and any(
        tab in _WORD_ONLY_RIBBON_TABS for tab in ribbon_hits
    ):
        evidence.append(f"Word ribbon tabs visible: {', '.join(ribbon_hits)}.")
        return True, evidence

    return False, ["Title bar has the file name but no Word signature."]


def build_open_success_verify(*, evidence: Iterable[str]) -> dict[str, Any]:
    """Build a verifier payload that completes a document-open task."""

    return {
        "status": "success",
        "action_effective": True,
        "task_complete": True,
        "evidence": list(evidence),
        "reason": (
            "The requested document is open: its name is shown in the Word "
            "title bar."
        ),
        "confidence": 0.92,
        "recommended_next": "finish",
    }


def apply_open_verify_override(
    state: Any,
    verify_data: Mapping[str, Any],
    *,
    after: Any,
) -> tuple[dict[str, Any], bool]:
    """Complete a document-open task once Word shows the requested file."""

    data = dict(verify_data)
    # Close tasks must never be completed by the open detector.
    if is_close_task(task_instruction(state)):
        return data, False

    detected, evidence = detect_opened_document(after, task_instruction(state))
    if not detected:
        return data, False

    prior = list(data.get("evidence") or [])
    data.update(build_open_success_verify(evidence=prior + evidence))
    return data, True


def is_close_task(instruction: Any) -> bool:
    """Report whether the task asks to close a document, window or app."""

    text = normalise_target_text(instruction)
    return bool(text) and any(marker in text for marker in _CLOSE_VERB_MARKERS)


def spatial_side(instruction: Any) -> str | None:
    """Return 'left' / 'right' when the task names a side of the screen."""

    text = normalise_target_text(instruction)
    has_left = any(marker in text for marker in _LEFT_MARKERS)
    has_right = any(marker in text for marker in _RIGHT_MARKERS)
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def close_shortcut_for(instruction: Any) -> list[str]:
    """Return Alt+F4 for a window-level close request, Ctrl+W otherwise."""

    text = normalise_target_text(instruction)
    if any(marker in text for marker in _WINDOW_LEVEL_MARKERS):
        return list(_WINDOW_CLOSE_KEYS)
    return list(_DOCUMENT_CLOSE_KEYS)


def _window_x_range(side: str | None, width: float) -> tuple[float, float]:
    if side == "left":
        return 0.0, width * 0.55
    if side == "right":
        return width * 0.45, width
    return 0.0, width


def _in_title_bar(bbox: BoundingBox, height: float) -> bool:
    _left, top, _right, bottom = bbox
    return (top + bottom) / 2.0 <= _title_bar_limit(height)


def _in_close_strip(
    bbox: BoundingBox,
    *,
    side: str | None,
    width: float,
) -> bool:
    left, _top, right, _bottom = bbox
    mid_x = (left + right) / 2.0
    win_left, win_right = _window_x_range(side, width)
    strip = max(_CLOSE_STRIP_MIN_PX, (win_right - win_left) * _CLOSE_STRIP_RATIO)
    return mid_x >= win_right - strip and win_left <= mid_x <= win_right


def _bbox_in_window(
    bbox: BoundingBox,
    *,
    side: str | None,
    width: float,
) -> bool:
    left, _top, right, _bottom = bbox
    mid_x = (left + right) / 2.0
    win_left, win_right = _window_x_range(side, width)
    return win_left <= mid_x <= win_right


def is_close_control_target(text: Any) -> bool:
    """Report whether a label belongs to the close-control whitelist."""

    raw = str(text or "").strip()
    if raw in _CLOSE_GLYPHS:
        return True
    return normalise_target_text(raw) in _CLOSE_CONTROL_LABELS


def is_status_close_decoy(text: Any) -> bool:
    """Report whether a label is an on/off / autosave chip, not the ✕ control."""

    raw = str(text or "").strip()
    if raw in {"关", "开"}:
        return True
    return normalise_target_text(raw) in _STATUS_DECOYS


def is_forbidden_close_target(text: Any) -> bool:
    """Report whether a label must never be treated as a close control."""

    if is_close_control_target(text):
        return False
    if is_status_close_decoy(text):
        return True
    normalised = normalise_target_text(text)
    if not normalised:
        return False
    if normalised in _MENU_LABELS:
        return True
    # Covers 文件 / 搜索文件 / 打开文件 / 最近文件.
    return "文件" in normalised


def _word_presence(
    observation: Any,
    instruction: Any,
) -> tuple[int, list[str]]:
    """Count Word title-bar evidence in the task's spatial region.

    Global ``window_title`` / ``application_name`` are ignored: after closing
    the left document the remaining Word window becomes foreground and would
    otherwise keep the count above zero.
    """

    if observation is None:
        return 0, []
    side = spatial_side(instruction)
    width, height = screen_size(observation)
    names = target_name_candidates(instruction)
    notes: list[str] = []
    count = 0

    for text, bbox in iter_labeled_boxes(observation):
        if not _in_title_bar(bbox, height):
            continue
        if not _bbox_in_window(bbox, side=side, width=width):
            continue
        normalised = normalise_target_text(text)
        if not normalised:
            continue
        if any(marker in normalised for marker in _WORD_TITLE_MARKERS):
            count += 1
            notes.append(text)
            continue
        if any(name in normalised for name in names):
            count += 1
            notes.append(text)
    return count, notes[:6]


def _identity_token(text: Any) -> str | None:
    normalised = normalise_target_text(text)
    if len(normalised) < 2:
        return None
    if normalised in _TASK_STOPWORDS or normalised in _GENERIC_TITLE_TOKENS:
        return None
    if normalised in _STATUS_DECOYS or normalised in _MENU_LABELS:
        return None
    if normalised in _CLOSE_CONTROL_LABELS:
        return None
    return normalised


def extract_identity_tokens(text: Any) -> list[str]:
    """Return distinctive tokens from a window title or OCR caption."""

    tokens: list[str] = []
    for piece in _TOKEN_PATTERN.findall(str(text or "")):
        token = _identity_token(piece)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _title_bar_region_texts(observation: Any, instruction: Any) -> list[str]:
    if observation is None:
        return []
    side = spatial_side(instruction)
    width, height = screen_size(observation)
    texts: list[str] = []
    for text, bbox in iter_labeled_boxes(observation):
        if not _in_title_bar(bbox, height):
            continue
        if not _bbox_in_window(bbox, side=side, width=width):
            continue
        if str(text or "").strip():
            texts.append(str(text).strip())
    return texts


def snapshot_target_identity(
    observation: Any,
    instruction: Any,
) -> tuple[list[str], int | None]:
    """Capture document identity tokens and the target Word HWND."""

    tokens: list[str] = []
    for name in target_name_candidates(instruction):
        if name not in tokens:
            tokens.append(name)

    hwnd: int | None = None
    if win32_windows.is_supported():
        try:
            window = win32_windows.find_word_window(
                side=spatial_side(instruction),
                title_tokens=tokens,
            )
        except win32_windows.Win32WindowError:
            window = None
        if window is not None:
            hwnd = window.hwnd
            for token in extract_identity_tokens(window.title):
                if token not in tokens:
                    tokens.append(token)

    for text in _title_bar_region_texts(observation, instruction):
        for token in extract_identity_tokens(text):
            if token not in tokens:
                tokens.append(token)

    return tokens, hwnd


def _observation_corpus(observation: Any) -> set[str]:
    """Normalised texts available for identity matching."""

    corpus: set[str] = set()
    if observation is None:
        return corpus
    for attribute in ("window_title", "application_name", "ocr_text"):
        for token in extract_identity_tokens(getattr(observation, attribute, None)):
            corpus.add(token)
        normalised = normalise_target_text(getattr(observation, attribute, None))
        if normalised:
            corpus.add(normalised)
    for text in iter_texts(observation):
        normalised = normalise_target_text(text)
        if normalised:
            corpus.add(normalised)
        for token in extract_identity_tokens(text):
            corpus.add(token)
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, Mapping):
        for title in metadata.get("window_titles") or []:
            normalised = normalise_target_text(title)
            if normalised:
                corpus.add(normalised)
            for token in extract_identity_tokens(title):
                corpus.add(token)
    return corpus


def identity_tokens_present(observation: Any, tokens: Sequence[str]) -> list[str]:
    """Return which identity tokens are still visible after an action."""

    if not tokens:
        return []
    corpus = _observation_corpus(observation)
    remaining: list[str] = []
    for token in tokens:
        normalised = normalise_target_text(token)
        if not normalised:
            continue
        if normalised in corpus or any(normalised in item for item in corpus):
            remaining.append(token)
    return remaining


def resolve_target_hwnd(instruction: Any, progress: Mapping[str, Any] | None = None) -> int | None:
    data = dict(progress or {})
    stored = data.get("target_hwnd")
    try:
        hwnd = int(stored) if stored is not None else None
    except (TypeError, ValueError):
        hwnd = None
    if hwnd is not None and win32_windows.is_window(hwnd):
        return hwnd
    if not win32_windows.is_supported():
        return None
    try:
        window = win32_windows.find_word_window(
            side=spatial_side(instruction),
            title_tokens=list(data.get("target_title_tokens") or ()),
            hwnd=hwnd,
        )
    except win32_windows.Win32WindowError:
        return None
    return window.hwnd if window is not None else None


def target_window_active(observation: Any, instruction: Any) -> bool:
    """True when the named Word window is present / focused."""

    if win32_windows.is_supported():
        hwnd = resolve_target_hwnd(instruction)
        if hwnd is not None and win32_windows.is_window(hwnd):
            return True
    count, _notes = _word_presence(observation, instruction)
    return count > 0


def close_action_changed_screen(
    before: Any,
    after: Any,
    progress: Mapping[str, Any] | None = None,
) -> bool:
    """Heuristic: the close action moved the desktop enough to matter."""

    data = dict(progress or {})

    before_hwnds = {
        int(value)
        for value in (data.get("before_word_hwnds") or [])
        if str(value).strip()
    }
    if before_hwnds:
        remaining = {hwnd for hwnd in before_hwnds if win32_windows.is_window(hwnd)}
        if remaining != before_hwnds:
            return True

    before_titles = {
        normalise_target_text(item)
        for item in (data.get("before_window_titles") or [])
        if normalise_target_text(item)
    }
    after_meta = getattr(after, "metadata", None)
    after_titles: set[str] = set()
    if isinstance(after_meta, Mapping):
        after_titles = {
            normalise_target_text(item)
            for item in (after_meta.get("window_titles") or [])
            if normalise_target_text(item)
        }
    if before_titles and after_titles and before_titles != after_titles:
        return True

    if before is None or after is None:
        return False

    before_meta = getattr(before, "metadata", None)
    after_kind = (
        str(after_meta.get("observation_kind") or "")
        if isinstance(after_meta, Mapping)
        else ""
    )
    before_kind = (
        str(before_meta.get("observation_kind") or "")
        if isinstance(before_meta, Mapping)
        else ""
    )
    # Never compare full-OCR density against a Win32 window list.
    if after_kind == "win32_windows" or before_kind == "win32_windows":
        return False

    before_elements, before_ocr = observation_counts(before)
    after_elements, after_ocr = observation_counts(after)
    if after_elements + _CLOSE_ELEMENT_DROP < before_elements:
        return True
    if after_ocr + _CLOSE_ELEMENT_DROP < before_ocr:
        return True
    return False


def snapshot_windows_for_close(state: Any) -> dict[str, Any] | None:
    """Remember the pre-close window set for change detection."""

    if not win32_windows.is_supported():
        return None
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return None
    progress = ensure_close_progress(state) or {}
    try:
        windows = win32_windows.list_top_level_windows()
        word_windows = win32_windows.list_word_windows()
    except win32_windows.Win32WindowError:
        return progress
    progress = {
        **progress,
        "before_window_titles": [window.title for window in windows],
        "before_word_hwnds": [window.hwnd for window in word_windows],
    }
    metadata[CLOSE_PROGRESS_KEY] = progress
    return progress


def is_close_action(action: Any, instruction: Any | None = None) -> bool:
    """Report whether *action* is a close hotkey or close-control click."""

    action_type = latest_action_type(action)
    if action_type == "hotkey":
        data = flatten_action_evidence(action)
        keys = [
            str(item).strip().casefold()
            for item in (data.get("keys") or [])
            if str(item).strip()
        ]
        expected = [
            str(item).casefold()
            for item in close_shortcut_for(instruction or "")
        ]
        return keys == expected
    if action_type in CLICK_ACTION_TYPES:
        data = flatten_action_evidence(action)
        target = (
            data.get("target_text")
            or data.get("matched_text")
            or data.get("text")
        )
        return is_close_control_target(target)
    return False


def close_hotkey_blocked(state: Any) -> bool:
    """True when another Ctrl+W would risk closing a different document."""

    progress = close_progress(state) or {}
    attempts = int(progress.get("close_attempts") or 0)
    return attempts >= 1 and bool(progress.get("last_close_screen_changed"))


def should_use_window_observe_after(state: Any, action: Any) -> bool:
    """Close hotkeys can be verified from Win32 window state alone."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return False
    if not win32_windows.is_supported():
        return False
    return is_close_action(action, instruction)


def build_window_close_observation(
    *,
    before: Any = None,
    instruction: Any = None,
) -> ObservationState:
    """Build a lightweight observation from top-level window titles/rects."""

    width, height = win32_windows.screen_size()
    if before is not None:
        prior_w = getattr(before, "screen_width", None)
        prior_h = getattr(before, "screen_height", None)
        try:
            if prior_w:
                width = int(prior_w)
            if prior_h:
                height = int(prior_h)
        except (TypeError, ValueError):
            pass

    windows = win32_windows.list_top_level_windows()
    items: list[dict[str, Any]] = []
    for window in windows:
        items.append(
            {
                "text": window.title,
                "bbox": [window.left, window.top, window.right, window.bottom],
                "source": "win32_window",
                "hwnd": window.hwnd,
                "class_name": window.class_name,
            }
        )

    foreground = None
    fg = win32_windows.foreground_hwnd()
    if fg is not None:
        for window in windows:
            if window.hwnd == fg:
                foreground = window
                break

    side = spatial_side(instruction)
    word_titles = win32_windows.word_window_titles(side=side)
    return ObservationState(
        screen_width=width,
        screen_height=height,
        window_title=foreground.title if foreground else None,
        application_name=foreground.class_name if foreground else None,
        ocr_text="\n".join(item["text"] for item in items) or None,
        ocr_items=list(items),
        gui_elements=list(items),
        source=ObservationSource.MANUAL,
        metadata={
            "observation_kind": "win32_windows",
            "window_titles": [window.title for window in windows],
            "word_window_titles": word_titles,
            "window_count": len(windows),
            "word_window_count": len(word_titles),
        },
    )

def visible_close_controls(
    observation: Any,
    instruction: Any,
) -> list[tuple[str, BoundingBox]]:
    """Return detected close-control labels in the target title-bar right strip."""

    if observation is None:
        return []
    width, height = screen_size(observation)
    side = spatial_side(instruction)
    hits: list[tuple[str, BoundingBox]] = []
    for text, bbox in iter_labeled_boxes(observation):
        if not is_close_control_target(text):
            continue
        if not _in_title_bar(bbox, height):
            continue
        if not _in_close_strip(bbox, side=side, width=width):
            continue
        hits.append((text, bbox))
    return hits


def close_control_visible(observation: Any, instruction: Any) -> bool:
    return bool(visible_close_controls(observation, instruction))


def close_hotkey_parameters(instruction: Any) -> dict[str, Any]:
    """Build a hotkey action payload for the close shortcut."""

    keys = close_shortcut_for(instruction)
    return {
        "keys": keys,
        "description": f"Close the target document/window with {'+'.join(keys)}.",
    }


def close_progress(state: Any) -> dict[str, Any] | None:
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    data = metadata.get(CLOSE_PROGRESS_KEY)
    return dict(data) if isinstance(data, Mapping) else None


def ensure_close_progress(state: Any) -> dict[str, Any] | None:
    """Create or refresh the activate→close stage for a close task."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return None

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return None

    current = close_progress(state) or {}
    side = spatial_side(instruction)
    tokens = list(current.get("target_title_tokens") or [])
    hwnd = current.get("target_hwnd")
    if not tokens or hwnd is None:
        snap_tokens, snap_hwnd = snapshot_target_identity(
            getattr(state, "observation", None), instruction
        )
        if not tokens:
            tokens = snap_tokens
        if hwnd is None:
            hwnd = snap_hwnd
    else:
        # Refresh HWND if the stored handle died but Word is still around.
        try:
            hwnd_i = int(hwnd) if hwnd is not None else None
        except (TypeError, ValueError):
            hwnd_i = None
        if hwnd_i is not None and not win32_windows.is_window(hwnd_i):
            _tokens, snap_hwnd = snapshot_target_identity(
                getattr(state, "observation", None), instruction
            )
            hwnd = snap_hwnd

    active = target_window_active(state.observation, instruction)
    if hwnd is not None and win32_windows.is_window(int(hwnd)):
        active = True
    control_visible = close_control_visible(state.observation, instruction)
    # A spatial cue with no Word chrome in that half means activate first.
    if side and not active and current.get("phase") != PHASE_CLOSE:
        phase = PHASE_ACTIVATE
    elif active or current.get("phase") == PHASE_CLOSE:
        phase = PHASE_CLOSE
    else:
        phase = PHASE_ACTIVATE if side else PHASE_CLOSE

    # Never rewind from close back to activate after a successful activation.
    if current.get("phase") == PHASE_CLOSE and current.get("activated"):
        phase = PHASE_CLOSE
        active = True

    attempts = int(current.get("close_attempts") or 0)
    blocked = attempts >= 1 and bool(current.get("last_close_screen_changed"))
    prefer_hotkey = (
        phase == PHASE_CLOSE and active and not control_visible and not blocked
    )
    shortcut = close_shortcut_for(instruction)
    keys = "+".join(shortcut)
    if blocked:
        next_action = (
            f"Close hotkey already changed the screen once; do not send {keys} "
            "again (risk of closing another document). Finish if the target is "
            "gone, otherwise activate the correct window without a second close."
        )
        prefer_hotkey = False
    elif phase == PHASE_ACTIVATE:
        next_action = (
            "Click the target Word title-bar text to activate that window; "
            f"do not click ✕ yet and do not send {keys}."
        )
    elif prefer_hotkey:
        next_action = (
            f"Close-control OCR is missing; send hotkey {keys} now. "
            "Do not invent or click ✕ / × / 关闭."
        )
    elif control_visible:
        next_action = (
            f"Prefer hotkey {keys}; clicking a detected title-bar close "
            "control (✕ / × / 关闭 / Close) is also allowed."
        )
    else:
        next_action = (
            f"Send hotkey {keys} once the target window is active."
        )

    progress = {
        "phase": phase,
        "side": side,
        "window_active": active,
        "close_control_visible": control_visible,
        "prefer_hotkey": prefer_hotkey,
        "shortcut": list(shortcut),
        "next_action": next_action,
        "close_attempts": attempts,
        "last_close_screen_changed": bool(
            current.get("last_close_screen_changed")
        ),
        "hotkey_blocked": blocked,
        "target_title_tokens": tokens,
    }
    if hwnd is not None:
        try:
            progress["target_hwnd"] = int(hwnd)
        except (TypeError, ValueError):
            pass
    if current.get("activated"):
        progress["activated"] = True
    for key in (
        "before_window_titles",
        "before_word_hwnds",
        "last_close_closed",
    ):
        if key in current:
            progress[key] = current[key]
    metadata[CLOSE_PROGRESS_KEY] = progress
    return progress


def maybe_rewrite_close_action(
    action_type: str,
    parameters: Mapping[str, Any],
    *,
    state: Any,
) -> tuple[str, dict[str, Any], bool]:
    """Rewrite illegal/unreachable close clicks to the close hotkey.

    When the target window is already active and the close control is not in
    OCR, clicking 关 / 文件 / an invented × cannot succeed. Fall back to
    Ctrl+W / Alt+F4 instead of failing the plan and re-observing.
    """

    params = dict(parameters)
    normalized_type = str(action_type or "").strip().lower()
    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return normalized_type, params, False

    progress = ensure_close_progress(state) or {}
    if str(progress.get("phase")) == PHASE_ACTIVATE:
        return normalized_type, params, False
    if close_hotkey_blocked(state):
        return normalized_type, params, False
    if not bool(progress.get("window_active")):
        return normalized_type, params, False

    prefer_hotkey = bool(progress.get("prefer_hotkey"))
    target = str(params.get("target_text") or params.get("text") or "")

    should_rewrite = False
    if normalized_type in CLICK_ACTION_TYPES:
        if prefer_hotkey:
            should_rewrite = True
        elif (
            is_status_close_decoy(target)
            or is_forbidden_close_target(target)
            or is_close_control_target(target)
        ):
            # Close-control labels with no matching OCR still rewrite; valid
            # detected controls keep their click path via prefer_hotkey=False
            # and geometry validation below.
            if is_close_control_target(target) and close_control_visible(
                getattr(state, "observation", None), instruction
            ):
                should_rewrite = False
            else:
                should_rewrite = True
    elif normalized_type == "hotkey":
        return normalized_type, params, False

    if not should_rewrite:
        return normalized_type, params, False

    hotkey_params = close_hotkey_parameters(instruction)
    hotkey_params["description"] = (
        f"{hotkey_params['description']} "
        f"(rewritten from {normalized_type} {target!r})"
    )
    return "hotkey", hotkey_params, True


def forced_close_hotkey(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Return a close hotkey when the planner failed but a Word window is present."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return None
    if close_hotkey_blocked(state):
        return None
    progress = ensure_close_progress(state) or {}
    observation = getattr(state, "observation", None)
    active = bool(progress.get("window_active")) or target_window_active(
        observation, instruction
    )
    # After repeated plan failures, any on-screen Word chrome is enough to
    # prefer the shortcut over another full re-observation.
    if not active:
        any_count, _notes = _word_presence(observation, "关闭 word 文档")
        active = any_count > 0
    if not active and win32_windows.is_supported():
        try:
            active = bool(win32_windows.list_word_windows())
        except win32_windows.Win32WindowError:
            active = False
    if not active:
        return None
    if str(progress.get("phase")) == PHASE_ACTIVATE:
        # Promote to close so subsequent validates accept the hotkey.
        metadata = getattr(state, "metadata", None)
        if isinstance(metadata, MutableMapping):
            progress = {
                **progress,
                "phase": PHASE_CLOSE,
                "window_active": True,
                "prefer_hotkey": not close_control_visible(observation, instruction),
                "activated": True,
                "next_action": (
                    "Send the close hotkey; do not invent a close-control click."
                ),
            }
            metadata[CLOSE_PROGRESS_KEY] = progress
    return "hotkey", close_hotkey_parameters(instruction)


def is_valid_close_click(
    text: Any,
    bbox: Any,
    observation: Any,
    instruction: Any,
) -> tuple[bool, str]:
    """Whitelist: close-control label in the title-bar right strip."""

    if is_status_close_decoy(text):
        return False, (
            f"target_text={text!r} is an autosave/status chip, not the close "
            "control."
        )
    if is_forbidden_close_target(text):
        return False, (
            f"target_text={text!r} opens a menu/panel; it is not a close control."
        )
    if not is_close_control_target(text):
        return False, (
            f"target_text={text!r} is not in the close-control whitelist "
            "(✕ / × / 关闭 / Close)."
        )
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "Close-control click requires a detected title-bar bbox."
    try:
        box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return False, "Close-control bbox is invalid."

    width, height = screen_size(observation)
    side = spatial_side(instruction)
    if not _in_title_bar(box, height):
        return False, "Close control must sit in the window title-bar band."
    if not _in_close_strip(box, side=side, width=width):
        return False, (
            "Close control must sit in the right strip of the target window "
            "title bar."
        )
    return True, "Close-control whitelist and title-bar right strip matched."


def is_valid_activate_click(
    text: Any,
    bbox: Any,
    observation: Any,
    instruction: Any,
) -> tuple[bool, str]:
    """Allow a title-bar click that focuses the named Word window."""

    if is_status_close_decoy(text) or is_forbidden_close_target(text):
        return False, (
            f"target_text={text!r} cannot activate the window; click the Word "
            "title bar instead."
        )
    if is_close_control_target(text):
        return False, (
            "Close control is not an activation target; activate the title bar "
            "first, then close."
        )
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "Activation click requires a detected title-bar bbox."
    try:
        box = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return False, "Activation bbox is invalid."

    width, height = screen_size(observation)
    side = spatial_side(instruction)
    if not _in_title_bar(box, height):
        return False, "Activation click must target the Word title-bar band."
    if not _bbox_in_window(box, side=side, width=width):
        return False, "Activation click must target the named side of the screen."

    normalised = normalise_target_text(text)
    names = target_name_candidates(instruction)
    if any(marker in normalised for marker in _WORD_TITLE_MARKERS) or any(
        name in normalised for name in names
    ):
        return True, "Title-bar activation target matched."
    return False, (
        f"target_text={text!r} is not Word title-bar evidence for activation."
    )


def validate_close_click_params(
    *,
    target_text: Any,
    bbox: Any,
    state: Any,
) -> None:
    """Raise ValueError when a close-task click violates the whitelist rules."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return

    progress = ensure_close_progress(state) or {}
    phase = str(progress.get("phase") or PHASE_CLOSE)
    observation = getattr(state, "observation", None)

    if phase == PHASE_ACTIVATE:
        ok, reason = is_valid_activate_click(
            target_text, bbox, observation, instruction
        )
        if not ok:
            raise ValueError(reason)
        return

    if bool(progress.get("prefer_hotkey")) or not close_control_visible(
        observation, instruction
    ):
        keys = "+".join(close_shortcut_for(instruction))
        raise ValueError(
            "Close-control OCR is missing in the title-bar right strip; "
            f"do not click {target_text!r}. Send hotkey {keys} instead."
        )

    ok, reason = is_valid_close_click(
        target_text, bbox, observation, instruction
    )
    if not ok:
        raise ValueError(reason)


def validate_close_hotkey(keys: Sequence[Any], state: Any) -> None:
    """Allow Ctrl+W / Alt+F4 only after the target window is active."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return

    progress = ensure_close_progress(state) or {}
    if close_hotkey_blocked(state):
        raise ValueError(
            "Close hotkey already changed the screen once; refusing a second "
            "Ctrl+W/Alt+F4 to avoid closing another document. Finish the task "
            "or activate the correct window without another close shortcut."
        )
    if str(progress.get("phase")) == PHASE_ACTIVATE or not (
        bool(progress.get("window_active"))
        or target_window_active(getattr(state, "observation", None), instruction)
    ):
        raise ValueError(
            "Close hotkey is blocked until the target Word window is active; "
            "click its title bar first."
        )

    expected = [str(item).casefold() for item in close_shortcut_for(instruction)]
    actual = [str(item).strip().casefold() for item in keys if str(item).strip()]
    if actual != expected:
        raise ValueError(
            f"Close task expects hotkey {'+'.join(expected)}, got "
            f"{'+'.join(actual) or 'empty'}."
        )


def advance_close_phase_after_verify(
    state: Any,
    action: Any,
    verify_data: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Move activate → close after a successful title-bar focus click."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return None

    data = dict(verify_data or {})
    if not (
        bool(data.get("action_effective"))
        and str(data.get("status") or "").lower() == "success"
    ):
        return None

    progress = ensure_close_progress(state) or {}
    if str(progress.get("phase")) != PHASE_ACTIVATE:
        return progress

    action_data = flatten_action_evidence(action)
    action_type = str(
        action_data.get("type") or action_data.get("action_type") or ""
    ).strip().lower()
    if action_type not in CLICK_ACTION_TYPES:
        return progress

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return progress
    progress = {
        **progress,
        "phase": PHASE_CLOSE,
        "activated": True,
        "window_active": True,
        "next_action": (
            "Click the title-bar close control (✕ / 关闭 / Close) in the "
            "right strip, or send the close hotkey."
        ),
    }
    metadata[CLOSE_PROGRESS_KEY] = progress
    return progress


def prepare_close_execution(state: Any) -> bool:
    """Activate the target Word window before a close hotkey/click."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return False
    action = getattr(state, "latest_action", None)
    if not is_close_action(action, instruction):
        return False
    snapshot_windows_for_close(state)
    if not win32_windows.is_supported():
        return False
    progress = ensure_close_progress(state) or {}
    hwnd = resolve_target_hwnd(instruction, progress)
    if hwnd is None:
        return False
    return win32_windows.activate_window(hwnd)


def record_close_attempt(
    state: Any,
    *,
    before: Any,
    after: Any,
    closed: bool,
) -> dict[str, Any] | None:
    """Remember that a close action ran and whether the desktop moved."""

    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return None
    if not is_close_action(getattr(state, "latest_action", None), instruction):
        return None

    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        return None

    progress = ensure_close_progress(state) or {}
    changed = close_action_changed_screen(before, after, progress)
    hwnd = progress.get("target_hwnd")
    if hwnd is not None and not win32_windows.is_window(hwnd):
        changed = True
    tokens = list(progress.get("target_title_tokens") or [])
    if tokens and not identity_tokens_present(after, tokens):
        changed = True

    progress = {
        **progress,
        "close_attempts": int(progress.get("close_attempts") or 0) + 1,
        "last_close_screen_changed": changed,
        "last_close_closed": bool(closed),
        "hotkey_blocked": (not closed) and changed,
    }
    if progress["hotkey_blocked"]:
        keys = "+".join(close_shortcut_for(instruction))
        progress["prefer_hotkey"] = False
        progress["next_action"] = (
            f"Close hotkey already changed the screen once; do not send {keys} "
            "again."
        )
    metadata[CLOSE_PROGRESS_KEY] = progress
    return progress


def detect_document_closed(
    before: Any,
    after: Any,
    instruction: Any,
    *,
    progress: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Detect that the target Word window/document left the named region."""

    if not is_close_task(instruction):
        return False, ["Task is not a close request."]
    if after is None:
        return False, ["No after observation."]

    data = dict(progress or {})
    evidence: list[str] = []

    # S3: HWND disappeared.
    hwnd = data.get("target_hwnd")
    try:
        hwnd_i = int(hwnd) if hwnd is not None else None
    except (TypeError, ValueError):
        hwnd_i = None
    if hwnd_i is not None and not win32_windows.is_window(hwnd_i):
        evidence.append(f"Target Word HWND {hwnd_i} is gone.")
        return True, evidence

    # S1: distinctive document tokens vanished from the whole screen.
    tokens = [str(item) for item in (data.get("target_title_tokens") or []) if str(item)]
    if tokens:
        remaining = identity_tokens_present(after, tokens)
        if not remaining:
            evidence.append(
                "Target document identity tokens disappeared "
                f"(tokens={tokens})."
            )
            return True, evidence

    # Win32: no Word window remains on the named side / with the identity.
    if win32_windows.is_supported():
        try:
            side = spatial_side(instruction)
            remaining_windows = win32_windows.filter_windows_by_side(
                win32_windows.list_word_windows(), side
            )
            if hwnd_i is not None and not remaining_windows:
                evidence.append(
                    "No Word window remains in the target screen region."
                )
                return True, evidence
            if tokens and hwnd_i is not None:
                still = []
                for window in remaining_windows:
                    title_norm = normalise_target_text(window.title)
                    if any(
                        normalise_target_text(token) in title_norm
                        for token in tokens
                        if normalise_target_text(token)
                    ):
                        still.append(window.title)
                if not still:
                    evidence.append(
                        "Target identity no longer appears in Word window titles "
                        f"(tokens={tokens})."
                    )
                    return True, evidence
        except win32_windows.Win32WindowError:
            pass

    before_count, before_notes = _word_presence(before, instruction)
    after_count, after_notes = _word_presence(after, instruction)

    if before_count > 0 and after_count == 0:
        evidence.append(
            "Word title-bar evidence disappeared from the target region "
            f"(before={before_notes})."
        )
        return True, evidence

    before_elements, _before_ocr = observation_counts(before)
    after_elements, _after_ocr = observation_counts(after)
    if (
        before_count > 0
        and after_count < before_count
        and after_elements + _CLOSE_ELEMENT_DROP < before_elements
    ):
        evidence.append(
            "Word title-bar evidence and overall UI density both dropped after "
            f"the close action (before_titles={before_notes}, "
            f"after_titles={after_notes or ['none']})."
        )
        return True, evidence

    # Safety completion: close hotkey already moved the desktop once.
    if (
        int(data.get("close_attempts") or 0) >= 1
        and bool(data.get("last_close_screen_changed"))
        and close_action_changed_screen(before, after, data)
    ):
        evidence.append(
            "Close hotkey already changed the screen substantially; treating "
            "the target document as closed to avoid a second Ctrl+W."
        )
        return True, evidence

    if before_notes and after_notes:
        before_norm = {normalise_target_text(item) for item in before_notes}
        after_norm = {normalise_target_text(item) for item in after_notes}
        after_has_word = any(
            any(marker in item for marker in _WORD_TITLE_MARKERS)
            for item in after_norm
        )
        if before_norm and not before_norm.intersection(after_norm) and not after_has_word:
            evidence.append(
                "Title-bar region no longer shows the previous Word chrome "
                f"(before={before_notes}, after={after_notes})."
            )
            return True, evidence

    return False, [
        "Target Word title-bar evidence is still present after the action."
    ]


def build_close_success_verify(*, evidence: Iterable[str]) -> dict[str, Any]:
    return {
        "status": "success",
        "action_effective": True,
        "task_complete": True,
        "evidence": list(evidence),
        "reason": (
            "The requested Word document/window is closed: its title-bar "
            "evidence is gone from the target region."
        ),
        "confidence": 0.92,
        "recommended_next": "finish",
    }


def apply_close_verify_override(
    state: Any,
    verify_data: Mapping[str, Any],
    *,
    before: Any,
    after: Any,
) -> tuple[dict[str, Any], bool]:
    """Complete a close task once the target Word chrome leaves the region."""

    data = dict(verify_data)
    instruction = task_instruction(state)
    if not is_close_task(instruction):
        return data, False

    progress = ensure_close_progress(state) or {}
    detected, evidence = detect_document_closed(
        before, after, instruction, progress=progress
    )
    record_close_attempt(state, before=before, after=after, closed=detected)
    if not detected:
        # If the screen already moved and another hotkey is blocked, finish
        # rather than letting the planner press Ctrl+W again.
        progress = close_progress(state) or progress
        if close_hotkey_blocked(state) and close_action_changed_screen(
            before, after, progress
        ):
            safety_evidence = list(data.get("evidence") or []) + [
                "Screen changed after the close hotkey; blocking a second close "
                "and completing the task."
            ]
            data.update(build_close_success_verify(evidence=safety_evidence))
            return data, True
        return data, False

    prior = list(data.get("evidence") or [])
    data.update(build_close_success_verify(evidence=prior + evidence))
    return data, True


__all__ = [
    "CLOSE_PROGRESS_KEY",
    "PHASE_ACTIVATE",
    "PHASE_CLOSE",
    "advance_close_phase_after_verify",
    "apply_close_verify_override",
    "apply_open_verify_override",
    "build_close_success_verify",
    "build_open_success_verify",
    "build_window_close_observation",
    "close_action_changed_screen",
    "close_control_visible",
    "close_hotkey_blocked",
    "close_hotkey_parameters",
    "close_progress",
    "close_shortcut_for",
    "detect_document_closed",
    "detect_opened_document",
    "ensure_close_progress",
    "extract_identity_tokens",
    "forced_close_hotkey",
    "identity_tokens_present",
    "is_close_action",
    "is_close_control_target",
    "is_close_task",
    "is_forbidden_close_target",
    "is_status_close_decoy",
    "is_valid_activate_click",
    "is_valid_close_click",
    "maybe_rewrite_close_action",
    "prepare_close_execution",
    "record_close_attempt",
    "resolve_target_hwnd",
    "should_use_window_observe_after",
    "snapshot_target_identity",
    "snapshot_windows_for_close",
    "spatial_side",
    "target_name_candidates",
    "target_window_active",
    "task_instruction",
    "validate_close_click_params",
    "validate_close_hotkey",
    "visible_close_controls",
    "wants_word_document",
]
