from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Any


class GUIAgentTestWindow:
    """
    GUI Agent 感知与行动模拟测试窗口。

    功能：
    1. 文件夹按钮：src / test / executor
    2. 输入框：Name
    3. Submit / Clear
    4. 可滚动文件列表
    5. Canvas 内部拖拽：executor_test -> Desktop
    6. 状态栏
    7. 可选接入 PerceptionPipeline 和 Executor
    """

    def __init__(
        self,
        root: tk.Tk,
        perception_pipeline: Optional[Any] = None,
        executor: Optional[Any] = None,
    ) -> None:
        self.root = root
        self.perception_pipeline = perception_pipeline
        self.executor = executor

        self.root.title("GUI Agent Test Environment")
        self.root.geometry("900x620")
        self.root.minsize(820, 560)

        self.status_var = tk.StringVar(value="Waiting")
        self.name_var = tk.StringVar()

        self.drag_item: Optional[int] = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.drag_start_position: Optional[tuple[float, float]] = None

        self._configure_style()
        self._build_layout()

    # ================================================================
    # UI construction
    # ================================================================

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        title = ttk.Label(
            self.root,
            text="GUI Agent Test Environment",
            style="Title.TLabel",
        )
        title.grid(row=0, column=0, sticky="w", padx=18, pady=(14, 8))

        body = ttk.Frame(self.root, padding=12)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        self._build_folder_area(body)
        self._build_input_area(body)
        self._build_file_list(body)
        self._build_drag_area(body)

        status_frame = ttk.Frame(self.root, padding=(14, 8))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(
            status_frame,
            text="Status:",
            style="Status.TLabel",
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Button(
            status_frame,
            text="Run Perception",
            command=self.run_perception_test,
        ).grid(row=0, column=2, padx=4)

        ttk.Button(
            status_frame,
            text="Run Action Test",
            command=self.run_action_test,
        ).grid(row=0, column=3, padx=4)

        ttk.Button(
            status_frame,
            text="Run Full Test",
            command=self.run_full_automation_test,
        ).grid(row=0, column=4, padx=4)

    def _build_folder_area(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="Folder Area",
            style="Section.TLabelframe",
            padding=14,
        )
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        frame.columnconfigure((0, 1, 2), weight=1)

        self.src_button = ttk.Button(
            frame,
            name="src_button",
            text="src",
            command=lambda: self.open_folder("src"),
        )

        self.src_button.grid(
        row=0,
        column=0,
        padx=6,
        pady=8,
        sticky="ew",
    )

        self.test_button = ttk.Button(
        frame,
        name="test_button",
        text="test",
        command=lambda: self.open_folder("test"),
    )

        self.test_button.grid(
        row=0,
        column=1,
        padx=6,
        pady=8,
        sticky="ew",
    )

        self.executor_button = ttk.Button(
        frame,
        name="executor_button",
        text="executor",
        command=lambda: self.open_folder(
            "executor"
        ),
    )

        self.executor_button.grid(
        row=0,
        column=2,
        padx=6,
        pady=8,
        sticky="ew",
    )

    def _build_input_area(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="Input Area",
            style="Section.TLabelframe",
            padding=14,
        )
        frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Name:").grid(
            row=0,
            column=0,
            padx=(0, 8),
            pady=8,
            sticky="w",
        )

        self.name_entry = ttk.Entry(
            frame,
            name="name_entry",
            textvariable=self.name_var,
        )
        self.name_entry.grid(
            row=0,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=8,
        )

        self.submit_button = ttk.Button(
            frame,
            name="submit_button",
            text="Submit",
            command=self.submit_name,
        )
        self.submit_button.grid(
            row=1,
            column=1,
            padx=4,
            pady=8,
            sticky="ew",
        )

        self.clear_button = ttk.Button(
            frame,
            name="clear_button",
            text="Clear",
            command=self.clear_name,
        )
        self.clear_button.grid(
            row=1,
            column=2,
            padx=4,
            pady=8,
            sticky="ew",
        )

    def _build_file_list(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="Scrollable File List",
            style="Section.TLabelframe",
            padding=10,
        )
        frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.file_list = tk.Listbox(
            list_frame,
            name="file_list",
            font=("Consolas", 11),
            activestyle="dotbox",
            exportselection=False,
        )
        self.file_list.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.file_list.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)

        for index in range(1, 31):
            self.file_list.insert(tk.END, f"file_{index:02d}.txt")

        self.file_list.bind("<<ListboxSelect>>", self.on_file_selected)
        self.file_list.bind("<Double-Button-1>", self.on_file_double_click)

    def _build_drag_area(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(
            parent,
            text="Drag Area",
            style="Section.TLabelframe",
            padding=10,
        )
        frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.drag_canvas = tk.Canvas(
            frame,
            name="drag_canvas",
            background="#f3f3f3",
            highlightthickness=1,
            highlightbackground="#888888",
        )
        self.drag_canvas.grid(row=0, column=0, sticky="nsew")

        self.desktop_rect = self.drag_canvas.create_rectangle(
            250,
            180,
            430,
            290,
            fill="#dbeafe",
            outline="#2563eb",
            width=2,
            tags=("desktop_target",),
        )
        self.drag_canvas.create_text(
            340,
            215,
            text="Desktop Target",
            font=("Segoe UI", 11, "bold"),
            tags=("desktop_target",),
        )
        self.drag_canvas.create_text(
            340,
            250,
            text="[ Desktop ]",
            font=("Segoe UI", 11),
            tags=("desktop_target",),
        )

        self.executor_rect = self.drag_canvas.create_rectangle(
            35,
            35,
            190,
            90,
            fill="#e5e7eb",
            outline="#444444",
            width=2,
            tags=("draggable", "executor_test"),
        )
        self.executor_text = self.drag_canvas.create_text(
            112,
            62,
            text="[ executor_test ]",
            font=("Segoe UI", 11, "bold"),
            tags=("draggable", "executor_test"),
        )

        self.drag_canvas.tag_bind("draggable", "<ButtonPress-1>", self.start_drag)
        self.drag_canvas.tag_bind("draggable", "<B1-Motion>", self.drag_motion)
        self.drag_canvas.tag_bind("draggable", "<ButtonRelease-1>", self.end_drag)

    # ================================================================
    # Basic interaction callbacks
    # ================================================================

    def open_folder(self, folder_name: str) -> None:
        self.set_status(f"Folder clicked: {folder_name}")

    def submit_name(self) -> None:
        name = self.name_var.get().strip()

        if not name:
            self.set_status("Submit failed: name is empty")
            messagebox.showwarning("Input Required", "Please enter a name.")
            return

        self.set_status(f"Submitted name: {name}")

    def clear_name(self) -> None:
        self.name_var.set("")
        self.name_entry.focus_set()
        self.set_status("Input cleared")

    def on_file_selected(self, _event: tk.Event) -> None:
        selection = self.file_list.curselection()

        if selection:
            filename = self.file_list.get(selection[0])
            self.set_status(f"Selected file: {filename}")

    def on_file_double_click(self, _event: tk.Event) -> None:
        selection = self.file_list.curselection()

        if selection:
            filename = self.file_list.get(selection[0])
            self.set_status(f"Opened file: {filename}")

    # ================================================================
    # Canvas drag logic
    # ================================================================

    def start_drag(self, event: tk.Event) -> None:
        current = self.drag_canvas.find_withtag("current")

        if not current:
            return

        self.drag_item = current[0]
        item_bbox = self.drag_canvas.bbox("executor_test")

        if item_bbox is None:
            return

        self.drag_offset_x = event.x - item_bbox[0]
        self.drag_offset_y = event.y - item_bbox[1]
        self.drag_start_position = (item_bbox[0], item_bbox[1])
        self.set_status("Dragging executor_test")

    def drag_motion(self, event: tk.Event) -> None:
        if self.drag_item is None:
            return

        item_bbox = self.drag_canvas.bbox("executor_test")

        if item_bbox is None:
            return

        new_left = event.x - self.drag_offset_x
        new_top = event.y - self.drag_offset_y

        delta_x = new_left - item_bbox[0]
        delta_y = new_top - item_bbox[1]

        self.drag_canvas.move("executor_test", delta_x, delta_y)

    def end_drag(self, _event: tk.Event) -> None:
        if self.drag_item is None:
            return

        if self._overlaps_desktop():
            self.set_status("Drag success: executor_test dropped on Desktop")
            self.drag_canvas.itemconfigure(
                self.executor_rect,
                fill="#bbf7d0",
                outline="#15803d",
            )
        else:
            self.set_status("Drag failed: target is Desktop")
            self._reset_draggable_position()

        self.drag_item = None

    def _overlaps_desktop(self) -> bool:
        executor_bbox = self.drag_canvas.bbox("executor_test")
        desktop_bbox = self.drag_canvas.bbox("desktop_target")

        if executor_bbox is None or desktop_bbox is None:
            return False

        ex1, ey1, ex2, ey2 = executor_bbox
        dx1, dy1, dx2, dy2 = desktop_bbox

        executor_center_x = (ex1 + ex2) / 2
        executor_center_y = (ey1 + ey2) / 2

        return (
            dx1 <= executor_center_x <= dx2
            and dy1 <= executor_center_y <= dy2
        )

    def _reset_draggable_position(self) -> None:
        bbox = self.drag_canvas.bbox("executor_test")

        if bbox is None:
            return

        delta_x = 35 - bbox[0]
        delta_y = 35 - bbox[1]

        self.drag_canvas.move("executor_test", delta_x, delta_y)
        self.drag_canvas.itemconfigure(
            self.executor_rect,
            fill="#e5e7eb",
            outline="#444444",
        )

    # ================================================================
    # Perception / execution integration
    # ================================================================

    def run_perception_test(self) -> None:
        """
        使用已注入的 PerceptionPipeline 截取当前窗口并执行感知。
        """
        if self.perception_pipeline is None:
            self.set_status("PerceptionPipeline is not connected")
            messagebox.showinfo(
                "Perception Test",
                "请在 main() 中创建 PerceptionPipeline 并注入窗口。",
            )
            return

        self.root.update_idletasks()

        region = self.get_window_screen_region()
        result = self.perception_pipeline.capture_and_run(region=region)

        texts = result.get_texts()
        self.set_status(
            f"Perception complete: {result.element_count} elements, "
            f"texts={texts[:5]}"
        )

        print("Perception summary:", result.summary())
        for element in result.merged_elements:
            print(
                element.text,
                element.element_type,
                element.bbox,
                element.center,
                element.confidence,
            )

    def run_action_test(self) -> None:
        """
        使用已注入的 Executor 执行一个安全动作序列。

        默认建议 Executor(dry_run=True)，确认坐标正确后再切换真实执行。
        """
        if self.executor is None:
            self.set_status("Executor is not connected")
            messagebox.showinfo(
                "Action Test",
                "请在 main() 中创建 Executor 并注入窗口。",
            )
            return

        try:
            from src.executor.action import Action, ActionSequence
        except ImportError as error:
            self.set_status(f"Action import failed: {error}")
            return

        self.root.update_idletasks()

        submit_x, submit_y = self.get_widget_center_screen(self.submit_button)
        clear_x, clear_y = self.get_widget_center_screen(self.clear_button)
        entry_x, entry_y = self.get_widget_center_screen(self.name_entry)

        sequence = ActionSequence(
            description="Tkinter input interaction test",
            actions=[
                Action.click(x=entry_x, y=entry_y, description="Focus name input"),
                Action.type_text(text="GUI Agent", description="Type test name"),
                Action.click(x=submit_x, y=submit_y, description="Click Submit"),
                Action.wait(0.3),
                Action.click(x=clear_x, y=clear_y, description="Click Clear"),
                Action.finish("Tkinter action test complete"),
            ],
        )

        result = self.executor.execute_sequence(sequence)

        self.set_status(
            f"Action test: success={result.success}, "
            f"executed={result.executed_actions}"
        )

        print("Execution summary:", result.summary())
        for item in result.results:
            print(item.summary())

    # ================================================================
    # Coordinate helpers
    # ================================================================

    def get_window_screen_region(self) -> tuple[int, int, int, int]:
        """
        返回窗口在真实屏幕中的区域：
        (left, top, width, height)
        """
        self.root.update_idletasks()

        return (
            self.root.winfo_rootx(),
            self.root.winfo_rooty(),
            self.root.winfo_width(),
            self.root.winfo_height(),
        )

    def get_canvas_tag_center_screen(
        self,
        tag: str,
    ) -> tuple[int, int]:


        self.root.update_idletasks()

        bbox = self.drag_canvas.bbox(tag)

        if bbox is None:
            raise ValueError(
            f"Canvas tag does not exist: {tag}"
        )

        x1, y1, x2, y2 = bbox

        local_center_x = (x1 + x2) // 2
        local_center_y = (y1 + y2) // 2

        screen_x = (
            self.drag_canvas.winfo_rootx()
            + local_center_x
        )

        screen_y = (
            self.drag_canvas.winfo_rooty()
            + local_center_y
        )

        return screen_x, screen_y
    
    def get_widget_center_screen(
        self,
        widget: tk.Widget,
    ) -> tuple[int, int]:
        
        self.root.update_idletasks()
        widget.update_idletasks()

        screen_x = (
            widget.winfo_rootx()
            + widget.winfo_width() // 2
        )

        screen_y = (
            widget.winfo_rooty()
            + widget.winfo_height() // 2
        )

        return screen_x, screen_y


    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()


    def run_full_automation_test(self) -> None:

        if self.executor is None:
            self.set_status("Executor is not connected")
            messagebox.showwarning(
                "Executor Missing",
                "请先注入 Executor。",
            )
            return

        try:
            from src.executor.action import (
                Action,
                ActionSequence,
            )
        except ImportError as error:
            self.set_status(
                f"Action import failed: {error}"
            )
            return

        self.root.update_idletasks()

        src_x, src_y = self.get_widget_center_screen(
            self.src_button
        )

        test_x, test_y = self.get_widget_center_screen(
            self.test_button
        )

        executor_x, executor_y = (
            self.get_widget_center_screen(
                self.executor_button
            )
        )

        entry_x, entry_y = self.get_widget_center_screen(
            self.name_entry
        )

        submit_x, submit_y = (
            self.get_widget_center_screen(
                self.submit_button
            )
        )

        clear_x, clear_y = (
            self.get_widget_center_screen(
                self.clear_button
            )
        )

        list_x, list_y = self.get_widget_center_screen(
            self.file_list
        )

    # ------------------------------------------------------------
    # 2. 获取 Canvas 拖拽源和目标的屏幕坐标
    # ------------------------------------------------------------

        drag_source_x, drag_source_y = (
            self.get_canvas_tag_center_screen(
                "executor_test"
            )
        )

        desktop_x, desktop_y = (
            self.get_canvas_tag_center_screen(
                "desktop_target"
            )
        )

    # ------------------------------------------------------------
    # 3. 创建完整动作序列
    # ------------------------------------------------------------

        sequence = ActionSequence(
            description=(
                "Complete Tkinter GUI Agent automation test"
            ),
            actions=[
            # 文件夹按钮
                Action.click(
                    x=src_x,
                    y=src_y,
                    description="Click src folder",
                ),
                Action.wait(0.2),

                Action.click(
                    x=test_x,
                    y=test_y,
                    description="Click test folder",
                ),
                Action.wait(0.2),

                Action.click(
                    x=executor_x,
                    y=executor_y,
                    description="Click executor folder",
                ),
                Action.wait(0.2),

            # 输入框
                Action.click(
                    x=entry_x,
                    y=entry_y,
                    description="Focus Name input",
                ),

                Action.hotkey_action(
                    "ctrl",
                    "a",
                    description="Select existing text",
                ),

                Action.type_text(
                    text="GUI Agent",
                    interval=0.05,
                    description="Type GUI Agent",
                ),

            # Submit
                Action.click(
                    x=submit_x,
                    y=submit_y,
                    description="Click Submit",
                ),
                Action.wait(0.5),

            # Clear
                Action.click(
                    x=clear_x,
                    y=clear_y,
                    description="Click Clear",
                ),
                Action.wait(0.3),

            # 文件列表
                Action.click(
                    x=list_x,
                    y=list_y,
                    description="Focus file list",
                ),

                Action.scroll(
                    amount=-8,
                    x=list_x,
                    y=list_y,
                    description="Scroll file list down",
                ),
                Action.wait(0.4),

                Action.double_click(
                    x=list_x,
                    y=list_y,
                    description="Open visible list item",
                ),
                Action.wait(0.4),

            # Canvas 拖拽
                Action.move_to(
                    x=drag_source_x,
                    y=drag_source_y,
                    duration=0.2,
                    description="Move to executor_test",
                ),

                Action.drag_to(
                    x=desktop_x,
                    y=desktop_y,
                    duration=1.0,
                    description=(
                        "Drag executor_test to Desktop"
                    ),
                ),

                Action.wait(0.5),

                Action.finish(
                    "Complete Tkinter automation test finished"
                ),
            ],
        )

    # ------------------------------------------------------------
    # 4. 执行
    # ------------------------------------------------------------

        result = self.executor.execute_sequence(
            sequence
        )

        self.set_status(
            "Full automation: "
            f"success={result.success}, "
            f"executed={result.executed_actions}, "
            f"failed={result.failed_actions}"
        )

        print(
            "Full automation summary:",
            result.summary(),
        )

        for execution_result in result.results:
            print(execution_result.summary())



def main() -> None:
    root = tk.Tk()

    # ------------------------------------------------------------
    # 方式 1：只运行模拟窗口
    # ------------------------------------------------------------
    # app = GUIAgentTestWindow(root)

    # ------------------------------------------------------------
    # 方式 2：接入你的感知与执行模块
    # 取消下面代码的注释，并注释上面的 app = ...
    # ------------------------------------------------------------
    #
    from src.perception.perception_pipeline import PerceptionPipeline
    from src.executor.executor import Executor
    from src.executor.action import Action
    
    pipeline = PerceptionPipeline(
        enable_preprocessing=False,
        enable_ocr=True,
        enable_ui_detection=True,
        merge_results=True,
        include_unmatched_ocr=True,
    )
    
    executor = Executor(
        dry_run=False,          # 先验证动作，不操作真实鼠标键盘
        stop_on_failure=True,
        raise_on_error=False,
    )
    
    app = GUIAgentTestWindow(
        root,
        perception_pipeline=pipeline,
        executor=executor,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
