"""
image_preprocess

Resize
Grey
Blur(Gaussian, Median)
Threshold (Binary, Adaptive, CLAHE)
Sherpen
"""

from __future__ import annotations

import cv2
import numpy as np

from perception.base_preprocess import BaseImageProcessor


MAX_PIXEL_VALUE=255

class ImageProcessor(BaseImageProcessor):
    def __init__(self):
        pass

    # Resize
    @staticmethod
    def resize(
        image: np.ndarray,
        width: int = None,
        height: int = None,
        interpolation=cv2.INTER_LINEAR,
    ) -> np.ndarray:

        h, w = image.shape[:2]  # shape(高度，宽度，通道数)

        if width is None and height is None:
            return image

        # 等比缩放
        if width is None:
            ratio = height / h
            width = int(w * ratio)

        elif height is None:
            ratio = width / w
            height = int(h * ratio)

        return cv2.resize(image, (width, height), interpolation)
    

    # Grey灰度化
    @staticmethod
    def gray(image: np.ndarray) -> np.ndarray:

        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # Gaussian Blur
    @staticmethod
    def gaussian(
        image: np.ndarray,
        kernel=(5,5),   # 滤波核：模糊矩阵尺寸
        sigma=0,    # 模糊强烈程度，离中心越远的晕染程度，sigma=0时自动计算
    ) -> np.ndarray:

        return cv2.GaussianBlur(image,kernel,sigma)
    

    # Median Blur
    @staticmethod
    def median(
        image:np.ndarray,
        kernel=3,
    ) -> np.ndarray:

        return cv2.medianBlur(image,kernel)
    

    # Binary Threshold 二值化->黑白图
    @staticmethod
    def binary(
        image: np.ndarray,
        threshold=150,
    ) -> np.ndarray:

        gray = ImageProcessor.gray(image)

        _, binary = cv2.threshold(
            gray,   # 输入灰度图
            threshold,  # 阈值分界线，小于阈值划分为纯黑，大于阈值划为纯白
            MAX_PIXEL_VALUE,    # 颜色最大值
            cv2.THRESH_BINARY,  # 二值化
        )

        return binary


    # Adaptive Threshold
    @staticmethod
    def adaptive_binary(
        image: np.ndarray,
    ) -> np.ndarray:

        gray = ImageProcessor.gray(image)
        
        ADAPTIVE_BLOCK_SIZE=11
        ADAPTIVE_C=2

        return cv2.adaptiveThreshold(
            gray,
            MAX_PIXEL_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 自适应方法
            cv2.THRESH_BINARY,  # 阈值类型
            ADAPTIVE_BLOCK_SIZE, # 局部区域大小
            ADAPTIVE_C,  #微调常量
        )


    #  CLAHE 限制对比度自适应直方图均衡化 （解决画面过暗/曝光过度）
    @staticmethod
    def clahe(
        image: np.ndarray,
    ) -> np.ndarray:

        gray = ImageProcessor.gray(image)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,   #对比度限制阈值
            tileGridSize=(8,8), # 图片分格8*8
        )

        return clahe.apply(gray)
    

    # Sharpen 锐化 （强化边缘/轮廓）
    @staticmethod
    def sharpen(
        image: np.ndarray,
    ) -> np.ndarray:

        # 锐化卷积核
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        )

        return cv2.filter2D(image, -1, kernel)
    

    # Process pipeline
    def process(
        self,
        image: np.ndarray,
        resize_width=None,
        resize_height=None,
        use_gray=True,
        use_gaussian=False,
        use_median=False,
        use_binary=False,
        use_adaptive=False,
        use_clahe=False,
        use_sharpen=False,
    ) -> np.ndarray:
        
        result = image.copy()

        # Resize
        if resize_width or resize_height:
            result = self.resize(
                result,
                resize_width,
                resize_height,
            )

        # Gray
        if use_gray:
            result = self.gray(result)

        # Gaussian
        if use_gaussian:
            result = self.gaussian(result)

        # Median
        if use_median:
            result = self.median(result)

        # Binary
        if use_binary:
            result = self.binary(result)

        # Adaptive Threshold
        if use_adaptive:
            result = self.adaptive_binary(result)

        # CLAHE
        if use_clahe:
            result = self.clahe(result)

        # Sharpen
        if use_sharpen:
            result = self.sharpen(result)

        return result