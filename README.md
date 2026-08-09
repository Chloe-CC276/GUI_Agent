# GUI_Agent

## 国企智慧采购智能助手 MVP

本地 OA、采购云、ERP 与 Agent 演示环境位于
[`apps/procurement_mvp`](apps/procurement_mvp/README.md)。该应用独立运行，
不修改现有 GUI Agent 执行链路。

## 模型调优框架

模型侧调优（框架优化 / LoRA GUI 训练 / 提示词优化）的设计文档位于本地 `docs/`（该目录已加入 `.gitignore`，不入库）。

```text
GUI_Agent/
│
├── executor/                         # 桌面动作定义与执行核心模块
│   ├── __init__.py                  # 执行模块统一导出接口
│   │
│   ├── action.py                    # 标准动作数据模型
│   │                                # 定义 ActionType、ActionStatus、MouseButton
│   │                                # 封装鼠标、键盘、等待、完成、失败等动作
│   │                                # 支持参数校验、动作状态更新、JSON/字典序列化
│   │                                # 支持 ActionSequence 多步骤动作序列
│   │
│   ├── mouse.py                     # 鼠标基础控制模块
│   │                                # 获取鼠标坐标和屏幕分辨率
│   │                                # 绝对/相对移动、单击、双击、右键、中键
│   │                                # 鼠标按下/释放、绝对/相对拖拽
│   │                                # 垂直/水平滚动、归一化坐标转换
│   │                                # 支持坐标校验、Fail-safe 和 Dry-run
│   │
│   ├── keyboard.py                  # 键盘基础控制模块
│   │                                # 单键、多次按键、按键序列和组合热键
│   │                                # ASCII文本输入及Unicode剪贴板粘贴
│   │                                # 支持复制、粘贴、全选、撤销、保存等快捷键
│   │                                # 支持按键校验、Dry-run和结构化结果输出
│   │
│   └── executor.py                  # 动作统一调度与执行器
│                                    # 接收Action、字典或JSON动作
│                                    # 按ActionType分发给MouseController或KeyboardController
│                                    # 执行WAIT、FINISH、FAIL等流程控制动作
│                                    # 支持ActionSequence顺序执行和失败中止
│                                    # 更新动作状态并记录执行历史、耗时和错误信息
│                                    # 支持Dry-run、停止请求和执行统计
│
├── perception/                       # 屏幕感知与界面解析核心模块
│   ├── __init__.py                  # 感知模块统一导出接口
│   │
│   ├── base_capture.py              # 屏幕截图模块抽象基类
│   │                                # 定义全屏截图、区域截图等统一接口
│   │
│   ├── screen_capture.py            # 屏幕截图实现
│   │                                # 获取全屏或指定区域截图
│   │                                # 获取屏幕尺寸、验证截图区域
│   │                                # 支持截图保存
│   │
│   ├── base_preprocess.py           # 图像预处理抽象基类
│   │                                # 定义图像处理接口及扩展规范
│   │
│   ├── image_preprocess.py          # OpenCV图像预处理流水线
│   │                                # 图像缩放、灰度化、高斯/中值滤波
│   │                                # 全局/自适应二值化、CLAHE增强、锐化
│   │                                # 支持多步骤流水线组合
│   │
│   ├── gui_element.py               # GUI元素统一数据结构
│   │                                # 保存文字、bbox、中心坐标、类型及置信度
│   │                                # 统一OCR、UI检测器和Executor之间的数据格式
│   │
│   ├── base_ocr.py                  # OCR引擎抽象基类
│   │                                # 定义detect等公共接口
│   │                                # 便于替换PaddleOCR、Qwen-VL、OmniParser等模型
│   │
│   ├── paddle_ocr.py                # PaddleOCR文字识别实现
│   │                                # 屏幕文字识别及边界框定位
│   │                                # 支持整图/区域OCR
│   │                                # 支持置信度过滤、文本排序、文本搜索
│   │                                # 坐标映射及GUIElement转换
│   │                                # OCR结果可视化及保存
│   │
│   ├── ui_detector.py               # OpenCV启发式UI组件检测器
│   │                                # 检测按钮、输入框、图标、复选框等组件
│   │                                # 基于轮廓、面积、长宽比等特征分类
│   │                                # OCR文字关联、重叠框过滤及可视化
│   │
│   └── perception_pipeline.py       # 感知总流水线
│                                    # 串联截图→预处理→OCR→UI检测
│                                    # 合并OCR与UI结果，输出PerceptionResult
│                                    # 支持全屏、区域及已有图像输入
│                                    # 支持文字搜索、类型筛选、最佳匹配
│                                    # 自动处理区域坐标映射
│                                    # 保存原图、处理图及可视化结果
│
├── test/                             # 自动化测试与集成测试模块
│   ├── __init__.py                  # 测试包初始化
│   │
│   ├── test_keyboard.py             # KeyboardController功能测试
│   │                                # 按键、热键、文本输入及参数校验
│   │
│   ├── test_mouse.py                # MouseController功能测试
│   │                                # 鼠标移动、点击、右键、滚动、拖拽
│   │                                # 感知模块联动测试
│   │
│   ├── test_paddle_ocr.py           # PaddleOCR功能测试
│   │                                # 整屏OCR、区域OCR、文本查找
│   │                                # OCR结果可视化保存
│   │
│   ├── test_perception_pipeline.py  # 感知流水线集成测试
│   │                                # 测试截图、预处理、OCR、UI检测及结果保存
│   │
│   └── test_perception_exe.py       # 感知与执行综合测试
│                                    # Fake组件单元测试
│                                    # GUIElement、PerceptionResult测试
│                                    # Mouse、Keyboard、Action测试
│                                    # Executor动作调度测试
│                                    # 感知→动作→执行完整链路测试
│                                    # 健壮性与异常输入测试
│                                    # 支持真实屏幕测试
│
└── __init__.py                      # 项目源码包初始化
```
