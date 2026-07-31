"""Extra smokes covering perception→observation and datasets package."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from src.agent.observation_utils import is_partial_observation
    from src.agent.tools import perception_to_observation

    result = SimpleNamespace(
        original_image=SimpleNamespace(shape=(200, 128, 3)),
        merged_elements=[],
        ocr_elements=[],
        elapsed_time=0.1,
        capture_region=(10, 20, 128, 200),
        metadata={
            "screen_width": 1920,
            "screen_height": 1080,
            "capture_region": [10, 20, 128, 200],
        },
    )
    obs = perception_to_observation(result)
    assert obs.screen_width == 1920, obs.screen_width
    assert obs.screen_height == 1080, obs.screen_height
    assert obs.metadata.get("capture_region") == [10, 20, 128, 200]
    assert is_partial_observation(obs)
    print("perception_to_observation region OK")

    from src.datasets import (
        Mind2WebLoader,
        ProcessedDatasetLoader,
        ScreenAgentLoader,
        WebArenaLoader,
    )

    assert callable(getattr(ScreenAgentLoader, "export_jsonl", None))
    assert callable(getattr(WebArenaLoader, "export_configs_jsonl", None))
    assert Mind2WebLoader is not None
    assert ProcessedDatasetLoader is not None
    print("datasets package OK")

    from src.agent.prompts import PromptBuilder, PromptConfig, PromptLanguage

    assert PromptBuilder and PromptConfig and PromptLanguage
    print("prompt imports OK")

    from src.executor import win32_windows

    assert win32_windows.is_supported()
    windows = win32_windows.list_top_level_windows()
    print(f"win32 windows enumerated: {len(windows)}")
    print("ALL EXTRA SMOKES PASSED")


if __name__ == "__main__":
    main()
