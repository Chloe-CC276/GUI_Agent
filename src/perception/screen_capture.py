"""
screen_capture.py

屏幕捕获和保存

"""

import logging
from pathlib import Path
from typing import Any

import cv2
import mss
import numpy as np

logger = logging.getLogger(__name__)


class ScreenCapture:
    def __init__(self, monitor_id: int = 1):
        self.sct = mss.mss()    # 初始化显示系统
        # self.monitors[0] 拼接显示器大屏
        # self.monitors[1] 主显示器
        # self.monitors[2] 副显示器
        self.monitors = self.sct.monitors   # 监测显示器数量
        self.monitor_id = self._validate_monitor_id(monitor_id)


    # 捕获整个屏幕，返回ndarray
    def capture_screen(self) -> np.ndarray:
        monitor = self.monitors[self.monitor_id]
        screenshot = self.sct.grab(monitor)

        img = np.array(screenshot)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # 色彩转换

        logger.info(
            "Screen captured: monitor_id=%s origin=(%s,%s) size=%sx%s",
            self.monitor_id,
            monitor["left"],
            monitor["top"],
            monitor["width"],
            monitor["height"],
        )
        return img
    

    # 捕获指定监视器
    def capture_monitor(self, monitor_id: int = 1) -> np.ndarray:
        monitor_id = self._validate_monitor_id(monitor_id)

        screenshot = self.sct.grab(self.monitors[monitor_id])

        img = np.array(screenshot)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img
    

    # 捕获指定区域
    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        
        monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        screenshot = self.sct.grab(monitor)

        img = np.array(screenshot)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img
    

    # 保存图片
    @staticmethod
    def save_image(
        image: np.ndarray,
        save_path: str,
    ) -> None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(str(save_path), image)

    
    # 展示图片
    @staticmethod
    def show_image(
        image: np.ndarray,
        window_name: str = "Screenshot",
    ) -> None:
        cv2.imshow(window_name, image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    
    # 获取屏幕分辨率
    def get_screen_size(self):

        monitor = self.monitors[self.monitor_id]

        return monitor["width"], monitor["height"]

    def get_monitor_geometry(self) -> dict[str, int]:
        monitor = self.monitors[self.monitor_id]
        return {
            "left": int(monitor["left"]),
            "top": int(monitor["top"]),
            "width": int(monitor["width"]),
            "height": int(monitor["height"]),
        }

    def screenshot_to_screen(
        self,
        x: int | float,
        y: int | float,
    ) -> tuple[int, int]:
        monitor = self.monitors[self.monitor_id]
        return (
            int(round(float(x) + monitor["left"])),
            int(round(float(y) + monitor["top"])),
        )

    def _validate_monitor_id(self, monitor_id: Any) -> int:
        if isinstance(monitor_id, bool) or not isinstance(monitor_id, int):
            raise TypeError("monitor_id must be an integer.")
        if monitor_id <= 0 or monitor_id >= len(self.monitors):
            raise ValueError(
                f"Monitor ID {monitor_id} does not exist; valid IDs are "
                f"1..{len(self.monitors) - 1}."
            )
        return monitor_id