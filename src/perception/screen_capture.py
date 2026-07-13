"""
screen_capture.py

屏幕捕获和保存

"""

from pathlib import Path

import cv2
import mss
import numpy as np

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()    # 初始化显示系统
        # self.monitors[0] 拼接显示器大屏
        # self.monitors[1] 主显示器
        # self.monitors[2] 副显示器
        self.monitors = self.sct.monitors   # 监测显示器数量


    # 捕获整个屏幕，返回ndarray
    def capture_screen(self) -> np.ndarray:
        screenshot = self.sct.grab(self.monitors[0])    # 截取全屏

        img = np.array(screenshot)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) # 色彩转换

        return img
    

    # 捕获指定监视器
    def capture_monitor(self, monitor_id: int = 1) -> np.ndarray:
        if monitor_id >= len(self.monitors):
            raise ValueError("Monitor ID does not exist.")

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

        monitor = self.monitors[1]

        return monitor["width"], monitor["height"]