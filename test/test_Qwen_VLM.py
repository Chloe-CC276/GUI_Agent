from src.model.qwen_vlm import QwenVLM


# 文本测试
vlm = QwenVLM(
    model="qwen3-vl-8b-instruct",
)

response = vlm.generate(
    prompt="请简要介绍你的视觉理解能力。",
)

print(response.text)
print(response.usage.to_dict())

vlm.close()


# #视觉测试
# vlm = QwenVLM(
#     model="qwen3-vl-plus",
#     region="beijing",
#     default_system_prompt=(
#         "你是GUI Agent的视觉分析模块。"
#         "请准确识别界面元素、文字和可操作控件。"
#     ),
# )

# response = vlm.generate(
#     prompt=(
#         "分析该截图，列出主要按钮、输入框、文件列表和状态栏。"
#     ),
#     images=[
#         "D:/GUIAgent_project/screenshots/pp_region_original.png",
#     ],
#     temperature=0.0,
#     max_tokens=1000,
# )

# print(response.text)
# print(response.latency_seconds)
# print(response.usage.total_tokens)

# vlm.close()