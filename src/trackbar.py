import argparse
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit("Pillow가 필요합니다. `python -m pip install pillow` 후 다시 실행하세요.") from exc

import process_all as pipeline

DEFAULT_INPUT_DIR = Path("data")
VALID_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGE_PANEL_SIZE = (280, 220)
CONTROL_PANEL_SIZE = (420, 440)
BASE_WIDTH = IMAGE_PANEL_SIZE[0] * 2 + CONTROL_PANEL_SIZE[0]
BASE_HEIGHT = IMAGE_PANEL_SIZE[1] * 2
BASE_RATIO = BASE_WIDTH / BASE_HEIGHT
START_WINDOW_SCALE = 1.18
START_WIDTH = int(BASE_WIDTH * START_WINDOW_SCALE)
START_HEIGHT = int(BASE_HEIGHT * START_WINDOW_SCALE)
COMPUTE_DEBOUNCE_MS = 180
BG = "#111111"
PANEL_BG = "#1b1b1b"
HEADER_BG = "#050505"
TEXT = "#f0f0f0"
SUBTEXT = "#b8b8b8"

SLIDER_SPECS = [
    ("trim_level", "트림", int(pipeline.DEFAULT_TRIM_LEVEL), 8, lambda v: f"{int(v)}"),
    ("smooth_level", "스무딩", int(pipeline.DEFAULT_SMOOTH_LEVEL), 20, lambda v: f"{int(v)}"),
    ("inner_rect_w_scale", "내부W", int(round(pipeline.INNER_RECT_W_SCALE_FROM_HOLE * 100)), 160, lambda v: f"{v / 100.0:.2f}"),
    ("inner_rect_h_scale", "내부H", int(round(pipeline.INNER_RECT_H_SCALE_FROM_HOLE * 100)), 160, lambda v: f"{v / 100.0:.2f}"),
    ("color_gate_h_margin", "Hue", int(round(pipeline.COLOR_GATE_H_MARGIN * 10)), 120, lambda v: f"{v / 10.0:.1f}"),
    ("color_gate_s_margin", "Sat", int(pipeline.COLOR_GATE_S_MARGIN), 100, lambda v: f"{int(v)}"),
    ("color_gate_v_margin", "Val", int(pipeline.COLOR_GATE_V_MARGIN), 120, lambda v: f"{int(v)}"),
    ("color_gate_high_v_pad", "밝기", int(pipeline.COLOR_GATE_HIGH_V_PAD), 60, lambda v: f"{int(v)}"),
    ("color_keep_min_ratio", "유지율", int(round(pipeline.COLOR_KEEP_MIN_RATIO * 100)), 100, lambda v: f"{v / 100.0:.2f}"),
    ("inner_defect_max_hole_ratio", "결함최대", int(round(pipeline.INNER_DEFECT_MAX_HOLE_RATIO * 100)), 50, lambda v: f"{v / 100.0:.2f}"),
    ("inner_defect_min_touch_ratio", "접촉", int(round(pipeline.INNER_DEFECT_MIN_TOUCH_RATIO * 100)), 30, lambda v: f"{v / 100.0:.2f}"),
    ("inner_defect_strong_touch_ratio", "강접촉", int(round(pipeline.INNER_DEFECT_STRONG_TOUCH_RATIO * 100)), 50, lambda v: f"{v / 100.0:.2f}"),
]

PAGE_SPECS = [
    ("기본", ["trim_level", "smooth_level", "inner_rect_w_scale", "inner_rect_h_scale"]),
    ("색상", ["color_gate_h_margin", "color_gate_s_margin", "color_gate_v_margin", "color_gate_high_v_pad"]),
    ("결함", ["color_keep_min_ratio", "inner_defect_max_hole_ratio", "inner_defect_min_touch_ratio", "inner_defect_strong_touch_ratio"]),
]

SLIDER_META = {
    key: {"label": label, "default": default, "max": max_value, "formatter": formatter}
    for key, label, default, max_value, formatter in SLIDER_SPECS
}

SLIDER_DESCRIPTIONS = {
    "trim_level": "코일 외곽을 추가로 얼마나 깎을지 조절합니다.",
    "smooth_level": "코일 경계를 얼마나 매끈하게 만들지 조절합니다.",
    "inner_rect_w_scale": "내부 빈 영역의 가로 크기 비율입니다.",
    "inner_rect_h_scale": "내부 빈 영역의 세로 크기 비율입니다.",
    "color_gate_h_margin": "코일 색상으로 인정할 Hue 허용 폭입니다.",
    "color_gate_s_margin": "코일 색상으로 인정할 채도 허용 폭입니다.",
    "color_gate_v_margin": "코일 색상으로 인정할 밝기 허용 폭입니다.",
    "color_gate_high_v_pad": "너무 밝은 비코일 부품을 더 강하게 제외합니다.",
    "color_keep_min_ratio": "색상 필터 뒤에도 유지해야 할 최소 코일 비율입니다.",
    "inner_defect_max_hole_ratio": "내부 돌출을 결함으로 볼 최대 크기입니다.",
    "inner_defect_min_touch_ratio": "결함이 코일 링에 붙어 있어야 하는 최소 비율입니다.",
    "inner_defect_strong_touch_ratio": "강하게 붙은 돌출을 살려주는 기준입니다.",
}

PIPELINE_DEFAULTS = {
    "INNER_RECT_W_SCALE_FROM_HOLE": pipeline.INNER_RECT_W_SCALE_FROM_HOLE,
    "INNER_RECT_H_SCALE_FROM_HOLE": pipeline.INNER_RECT_H_SCALE_FROM_HOLE,
    "COLOR_GATE_H_MARGIN": pipeline.COLOR_GATE_H_MARGIN,
    "COLOR_GATE_H_MARGIN_RELAXED": pipeline.COLOR_GATE_H_MARGIN_RELAXED,
    "COLOR_GATE_S_MARGIN": pipeline.COLOR_GATE_S_MARGIN,
    "COLOR_GATE_S_MARGIN_RELAXED": pipeline.COLOR_GATE_S_MARGIN_RELAXED,
    "COLOR_GATE_V_MARGIN": pipeline.COLOR_GATE_V_MARGIN,
    "COLOR_GATE_V_MARGIN_RELAXED": pipeline.COLOR_GATE_V_MARGIN_RELAXED,
    "COLOR_GATE_HIGH_V_PAD": pipeline.COLOR_GATE_HIGH_V_PAD,
    "COLOR_KEEP_MIN_RATIO": pipeline.COLOR_KEEP_MIN_RATIO,
    "COLOR_KEEP_MIN_RATIO_RELAXED": pipeline.COLOR_KEEP_MIN_RATIO_RELAXED,
    "INNER_DEFECT_MAX_HOLE_RATIO": pipeline.INNER_DEFECT_MAX_HOLE_RATIO,
    "INNER_DEFECT_MIN_TOUCH_RATIO": pipeline.INNER_DEFECT_MIN_TOUCH_RATIO,
    "INNER_DEFECT_STRONG_TOUCH_RATIO": pipeline.INNER_DEFECT_STRONG_TOUCH_RATIO,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="코일 마스킹 파이프라인 튜닝 도구")
    parser.add_argument("--image", help="single image path to open")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="directory to browse")
    return parser.parse_args()


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"입력 폴더가 없습니다: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS)


def restore_pipeline_defaults() -> None:
    for name, value in PIPELINE_DEFAULTS.items():
        setattr(pipeline, name, value)


def fit_contain(image: np.ndarray, size: tuple[int, int], pad_value: int = 24) -> np.ndarray:
    width, height = size
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    x0 = max(0, (width - new_w) // 2)
    y0 = max(0, (height - new_h) // 2)
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    border_color = (140, 140, 140)
    cv2.rectangle(canvas, (x0, y0), (x0 + new_w - 1, y0 + new_h - 1), border_color, 2)
    return canvas



def mask_from_result(result: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if result is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray > 0, 255, 0).astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def overlay_from_result(original: np.ndarray, result: np.ndarray | None) -> np.ndarray:
    if result is None:
        return original.copy()

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    overlay = (original.astype(np.float32) * 0.42).astype(np.uint8)

    if np.any(mask):
        blended = cv2.addWeighted(original, 0.45, result, 0.95, 0)
        overlay[mask] = blended[mask]
        contour_mask = np.where(mask, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 210, 120), 2, cv2.LINE_AA)
    return overlay


def fallback_result(shape: tuple[int, int]) -> np.ndarray:
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    cv2.putText(image, "No detection", (20, shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def decode_config(raw_values: dict[str, int]) -> dict[str, float | int]:
    return {
        "trim_level": int(raw_values["trim_level"]),
        "smooth_level": int(raw_values["smooth_level"]),
        "inner_rect_w_scale": raw_values["inner_rect_w_scale"] / 100.0,
        "inner_rect_h_scale": raw_values["inner_rect_h_scale"] / 100.0,
        "color_gate_h_margin": raw_values["color_gate_h_margin"] / 10.0,
        "color_gate_s_margin": int(raw_values["color_gate_s_margin"]),
        "color_gate_v_margin": int(raw_values["color_gate_v_margin"]),
        "color_gate_high_v_pad": int(raw_values["color_gate_high_v_pad"]),
        "color_keep_min_ratio": raw_values["color_keep_min_ratio"] / 100.0,
        "inner_defect_max_hole_ratio": raw_values["inner_defect_max_hole_ratio"] / 100.0,
        "inner_defect_min_touch_ratio": raw_values["inner_defect_min_touch_ratio"] / 100.0,
        "inner_defect_strong_touch_ratio": raw_values["inner_defect_strong_touch_ratio"] / 100.0,
    }


def apply_pipeline_config(config: dict[str, float | int]) -> None:
    pipeline.INNER_RECT_W_SCALE_FROM_HOLE = float(config["inner_rect_w_scale"])
    pipeline.INNER_RECT_H_SCALE_FROM_HOLE = float(config["inner_rect_h_scale"])

    hue_margin = float(config["color_gate_h_margin"])
    sat_margin = int(config["color_gate_s_margin"])
    val_margin = int(config["color_gate_v_margin"])
    keep_min_ratio = float(config["color_keep_min_ratio"])

    pipeline.COLOR_GATE_H_MARGIN = hue_margin
    pipeline.COLOR_GATE_H_MARGIN_RELAXED = min(pipeline.COLOR_GATE_H_MAX_TOL_RELAXED, hue_margin + 3.0)
    pipeline.COLOR_GATE_S_MARGIN = sat_margin
    pipeline.COLOR_GATE_S_MARGIN_RELAXED = min(255, sat_margin + 16)
    pipeline.COLOR_GATE_V_MARGIN = val_margin
    pipeline.COLOR_GATE_V_MARGIN_RELAXED = min(255, val_margin + 20)
    pipeline.COLOR_GATE_HIGH_V_PAD = int(config["color_gate_high_v_pad"])
    pipeline.COLOR_KEEP_MIN_RATIO = keep_min_ratio
    pipeline.COLOR_KEEP_MIN_RATIO_RELAXED = max(0.05, keep_min_ratio - 0.23)
    pipeline.INNER_DEFECT_MAX_HOLE_RATIO = float(config["inner_defect_max_hole_ratio"])
    pipeline.INNER_DEFECT_MIN_TOUCH_RATIO = float(config["inner_defect_min_touch_ratio"])
    pipeline.INNER_DEFECT_STRONG_TOUCH_RATIO = float(config["inner_defect_strong_touch_ratio"])


class TrackbarApp:
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths
        self.index = 0
        self.raw_values = {key: default for key, _label, default, _max_value, _formatter in SLIDER_SPECS}
        self.current_image: np.ndarray | None = None
        self.current_result: np.ndarray | None = None
        self.compute_job: str | None = None
        self.render_job: str | None = None
        self.request_lock = threading.Lock()
        self.request_event = threading.Event()
        self.stop_event = threading.Event()
        self.result_queue: queue.Queue[tuple[int, int, np.ndarray | None, str | None]] = queue.Queue()
        self.pending_request: tuple[int, int, np.ndarray, dict[str, float | int]] | None = None
        self.request_token = 0
        self.photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.panel_frames: dict[str, tk.Frame] = {}
        self.panel_layout: dict[str, dict[str, int]] = {}
        self.focused_panel: str | None = None
        self.last_window_size = (START_WIDTH, START_HEIGHT)
        self.aspect_adjusting = False

        self.root = tk.Tk()
        self.root.title("Tuner")
        self.root.configure(bg=BG)
        self.root.geometry(f"{START_WIDTH}x{START_HEIGHT}")
        self.root.minsize(BASE_WIDTH // 2, BASE_HEIGHT // 2)
        self.root.aspect(BASE_WIDTH, BASE_HEIGHT, BASE_WIDTH, BASE_HEIGHT)
        self.root.bind("<KeyPress-q>", lambda _e: self.close())
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.bind("<Configure>", self.on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.slider_vars = {
            key: tk.IntVar(master=self.root, value=value)
            for key, value in self.raw_values.items()
        }
        self.status_var = tk.StringVar(master=self.root, value="대기")
        self.value_labels: dict[str, tk.Label] = {}

        self.build_ui()

        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.load_current_image()
        self.request_compute(immediate=True)
        self.root.after(40, self.poll_worker_results)
        self.root.after(40, self.render_panels)

    def build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=1, minsize=IMAGE_PANEL_SIZE[1], uniform="main_rows")
        self.root.grid_rowconfigure(1, weight=1, minsize=IMAGE_PANEL_SIZE[1], uniform="main_rows")
        self.root.grid_columnconfigure(0, weight=1, minsize=IMAGE_PANEL_SIZE[0], uniform="main_cols")
        self.root.grid_columnconfigure(1, weight=1, minsize=IMAGE_PANEL_SIZE[0], uniform="main_cols")
        self.root.grid_columnconfigure(2, weight=0, minsize=CONTROL_PANEL_SIZE[0])

        self.original_label = self.build_image_panel("original", "원본", 0, 0)
        self.mask_label = self.build_image_panel("mask", "오버레이", 0, 1)
        self.result_label = self.build_image_panel("result", "결과", 1, 0)
        self.build_button_panel("buttons", 1, 1)
        self.build_control_panel("controls", 0, 2)

    def build_panel_shell(
        self,
        panel_key: str,
        title: str,
        row: int,
        column: int,
        rowspan: int = 1,
        columnspan: int = 1,
        clickable: bool = False,
    ) -> tk.Frame:
        frame = tk.Frame(self.root, bg=PANEL_BG, highlightbackground="#b0b0b0", highlightthickness=2)
        frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, sticky="nsew")
        self.panel_frames[panel_key] = frame
        self.panel_layout[panel_key] = {
            "row": row,
            "column": column,
            "rowspan": rowspan,
            "columnspan": columnspan,
        }
        header = tk.Frame(frame, bg=HEADER_BG, height=30)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_label = tk.Label(
            header,
            text=title,
            bg=HEADER_BG,
            fg=TEXT,
            anchor="w",
            padx=8,
            font=("Malgun Gothic", 12, "bold"),
        )
        title_label.pack(fill="both", expand=True)
        if clickable:
            for widget in (header, title_label):
                widget.bind("<Button-1>", lambda _e, key=panel_key: self.toggle_panel_focus(key))
        return frame

    def build_image_panel(self, panel_key: str, title: str, row: int, column: int) -> tk.Label:
        frame = self.build_panel_shell(panel_key, title, row, column, clickable=True)
        body = tk.Label(frame, bg=PANEL_BG, borderwidth=0, highlightthickness=0)
        body.pack(fill="both", expand=True)
        body.bind("<Configure>", self.queue_render)
        body.bind("<Button-1>", lambda _e, key=panel_key: self.toggle_panel_focus(key))
        return body

    def build_button_panel(self, panel_key: str, row: int, column: int) -> None:
        frame = self.build_panel_shell(panel_key, "버튼", row, column)
        body = tk.Frame(frame, bg=PANEL_BG, padx=18, pady=18)
        body.pack(fill="both", expand=True)
        for row_idx in range(3):
            body.grid_rowconfigure(row_idx, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        button_specs = [
            ("이전", self.prev_image, 0, 0, 1),
            ("다음", self.next_image, 0, 1, 1),
            ("기본값", self.reset_defaults, 1, 0, 1),
            ("출력", self.print_status, 1, 1, 1),
            ("종료", self.close, 2, 0, 2),
        ]
        for label, command, row_idx, col_idx, span in button_specs:
            tk.Button(
                body,
                text=label,
                command=command,
                bg="#4c4c4c",
                fg=TEXT,
                activebackground="#666666",
                activeforeground=TEXT,
                relief="solid",
                bd=1,
                font=("Malgun Gothic", 11, "bold"),
            ).grid(row=row_idx, column=col_idx, columnspan=span, sticky="nsew", padx=8, pady=8)

    def build_control_panel(self, panel_key: str, row: int, column: int) -> None:
        frame = self.build_panel_shell(panel_key, "트랙바", row, column, rowspan=2)
        body = tk.Frame(frame, bg=PANEL_BG, padx=10, pady=8)
        body.pack(fill="both", expand=True)

        self.file_label = tk.Label(
            body,
            text="",
            bg=PANEL_BG,
            fg=SUBTEXT,
            anchor="w",
            justify="left",
            font=("Malgun Gothic", 9),
        )
        self.file_label.pack(fill="x", pady=(0, 8))

        self.status_label = tk.Label(
            body,
            textvariable=self.status_var,
            bg=PANEL_BG,
            fg="#9bd1ff",
            anchor="w",
            justify="left",
            font=("Malgun Gothic", 9, "bold"),
        )
        self.status_label.pack(fill="x", pady=(0, 8))

        style = ttk.Style(self.root)
        style.theme_use("default")
        style.configure("TNotebook", background=PANEL_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(10, 6))

        notebook = ttk.Notebook(body)
        notebook.pack(fill="both", expand=True)

        for page_title, keys in PAGE_SPECS:
            tab = tk.Frame(notebook, bg=PANEL_BG)
            notebook.add(tab, text=page_title)
            for key in keys:
                self.build_slider_row(tab, key)

        tk.Label(
            body,
            text="항상 원본 해상도로 계산\n이미지 제목이나 본문 클릭: 확대 / 복귀\n창 크기는 자유 조절, 비율은 고정\nq / esc 종료",
            bg=PANEL_BG,
            fg=SUBTEXT,
            anchor="w",
            justify="left",
            font=("Malgun Gothic", 9),
        ).pack(fill="x", pady=(8, 0))

    def toggle_panel_focus(self, panel_key: str) -> None:
        if panel_key not in {"original", "mask", "result"}:
            return
        if self.focused_panel == panel_key:
            self.restore_panel_layout()
            return

        self.focused_panel = panel_key
        for name, frame in self.panel_frames.items():
            if name == "controls":
                layout = self.panel_layout[name]
                frame.grid()
                frame.grid_configure(
                    row=layout["row"],
                    column=layout["column"],
                    rowspan=layout["rowspan"],
                    columnspan=layout["columnspan"],
                    sticky="nsew",
                )
            elif name == panel_key:
                frame.grid()
                frame.grid_configure(row=0, column=0, rowspan=2, columnspan=2, sticky="nsew")
            else:
                frame.grid_remove()
        self.root.update_idletasks()
        self.queue_render()

    def restore_panel_layout(self) -> None:
        self.focused_panel = None
        for name, frame in self.panel_frames.items():
            layout = self.panel_layout[name]
            frame.grid()
            frame.grid_configure(
                row=layout["row"],
                column=layout["column"],
                rowspan=layout["rowspan"],
                columnspan=layout["columnspan"],
                sticky="nsew",
            )
        self.last_window_size = (START_WIDTH, START_HEIGHT)
        self.aspect_adjusting = True
        self.root.geometry(f"{START_WIDTH}x{START_HEIGHT}")
        self.root.after_idle(self.finish_aspect_adjust)
        self.root.update_idletasks()
        self.queue_render()

    def build_slider_row(self, parent: tk.Frame, key: str) -> None:
        meta = SLIDER_META[key]
        wrap = tk.Frame(parent, bg=PANEL_BG, pady=6)
        wrap.pack(fill="x")

        top = tk.Frame(wrap, bg=PANEL_BG)
        top.pack(fill="x")
        tk.Label(top, text=meta["label"], bg=PANEL_BG, fg=TEXT, anchor="w", font=("Malgun Gothic", 10, "bold")).pack(side="left")
        value_label = tk.Label(top, text=meta["formatter"](self.slider_vars[key].get()), bg=PANEL_BG, fg="#9bd1ff", anchor="e", font=("Consolas", 10))
        value_label.pack(side="right")
        self.value_labels[key] = value_label

        tk.Scale(
            wrap,
            from_=0,
            to=meta["max"],
            orient="horizontal",
            showvalue=False,
            resolution=1,
            variable=self.slider_vars[key],
            command=lambda value, name=key: self.on_slider_change(name, value),
            bg=PANEL_BG,
            fg=TEXT,
            highlightthickness=0,
            troughcolor="#404040",
            activebackground="#7fb5ff",
            bd=0,
        ).pack(fill="x")

        tk.Label(
            wrap,
            text=SLIDER_DESCRIPTIONS.get(key, ""),
            bg=PANEL_BG,
            fg=SUBTEXT,
            anchor="w",
            justify="left",
            wraplength=340,
            font=("Malgun Gothic", 8),
        ).pack(fill="x", pady=(2, 0))

    def queue_render(self, _event=None) -> None:
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(30, self.render_panels)

    def on_window_configure(self, event) -> None:
        if event.widget is not self.root:
            return
        width, height = event.width, event.height
        if self.aspect_adjusting:
            self.last_window_size = (width, height)
            return
        last_w, last_h = self.last_window_size
        width_changed = abs(width - last_w)
        height_changed = abs(height - last_h)
        if width_changed == 0 and height_changed == 0:
            return
        if width_changed >= height_changed:
            target_w = width
            target_h = max(1, int(round(width / BASE_RATIO)))
        else:
            target_h = height
            target_w = max(1, int(round(height * BASE_RATIO)))
        self.last_window_size = (target_w, target_h)
        if target_w != width or target_h != height:
            self.aspect_adjusting = True
            self.root.geometry(f"{target_w}x{target_h}")
            self.root.after_idle(self.finish_aspect_adjust)
        self.queue_render()

    def finish_aspect_adjust(self) -> None:
        self.aspect_adjusting = False

    def on_slider_change(self, key: str, value: str) -> None:
        raw_value = int(round(float(value)))
        self.raw_values[key] = raw_value
        self.value_labels[key].configure(text=SLIDER_META[key]["formatter"](raw_value))
        self.request_compute(immediate=False)

    def load_current_image(self) -> None:
        path = self.image_paths[self.index]
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"이미지를 읽을 수 없습니다: {path}")
        self.current_image = image
        self.current_result = None
        self.file_label.configure(text=f"{path.name}\n원본 {image.shape[1]}x{image.shape[0]}")
        self.status_var.set("계산 대기")
        self.queue_render()

    def request_compute(self, immediate: bool) -> None:
        if self.compute_job is not None:
            self.root.after_cancel(self.compute_job)
            self.compute_job = None
        if immediate:
            self.enqueue_compute()
        else:
            self.compute_job = self.root.after(COMPUTE_DEBOUNCE_MS, self.enqueue_compute)

    def enqueue_compute(self) -> None:
        if self.current_image is None or self.stop_event.is_set():
            return
        self.compute_job = None
        config = decode_config(self.raw_values)
        image = self.current_image.copy()
        self.request_token += 1
        token = self.request_token
        with self.request_lock:
            self.pending_request = (token, self.index, image, config)
        self.status_var.set("계산중...")
        self.request_event.set()

    def worker_loop(self) -> None:
        while not self.stop_event.is_set():
            self.request_event.wait(0.1)
            if self.stop_event.is_set():
                return
            if not self.request_event.is_set():
                continue
            with self.request_lock:
                request = self.pending_request
                self.pending_request = None
                self.request_event.clear()
            if request is None:
                continue
            token, image_index, image, config = request
            try:
                apply_pipeline_config(config)
                result = pipeline.apply_texture_mask(
                    img=image,
                    trim_level=int(config["trim_level"]),
                    smooth_level=int(config["smooth_level"]),
                )
                self.result_queue.put((token, image_index, result, None))
            except Exception as exc:
                self.result_queue.put((token, image_index, None, str(exc)))
            finally:
                restore_pipeline_defaults()

    def poll_worker_results(self) -> None:
        latest: tuple[int, int, np.ndarray | None, str | None] | None = None
        try:
            while True:
                latest = self.result_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            token, image_index, result, error = latest
            if token == self.request_token and image_index == self.index:
                self.current_result = result
                if error:
                    self.status_var.set("오류")
                    self.file_label.configure(text=f"{self.image_paths[self.index].name}\n{error}")
                else:
                    self.status_var.set("준비됨")
                self.render_panels()

        if not self.stop_event.is_set():
            self.root.after(40, self.poll_worker_results)

    def render_image_on_label(self, label: tk.Label, image: np.ndarray, cache_key: str) -> None:
        label.update_idletasks()
        width = max(40, label.winfo_width())
        height = max(40, label.winfo_height())
        contained = fit_contain(image, (width, height))
        rgb = cv2.cvtColor(contained, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=photo)
        label.image = photo
        self.photo_refs[cache_key] = photo

    def render_panels(self) -> None:
        self.render_job = None
        if self.current_image is None:
            return
        original = self.current_image
        result = self.current_result if self.current_result is not None else fallback_result(original.shape[:2])
        overlay = overlay_from_result(original, self.current_result)
        self.render_image_on_label(self.original_label, original, "original")
        self.render_image_on_label(self.mask_label, overlay, "mask")
        self.render_image_on_label(self.result_label, result, "result")

    def current_config(self) -> dict[str, float | int]:
        return decode_config(self.raw_values)

    def reset_defaults(self) -> None:
        for key, meta in SLIDER_META.items():
            value = int(meta["default"])
            self.raw_values[key] = value
            self.slider_vars[key].set(value)
            self.value_labels[key].configure(text=meta["formatter"](value))
        self.status_var.set("기본값 복원")
        self.request_compute(immediate=False)

    def print_status(self) -> None:
        path = self.image_paths[self.index]
        config = self.current_config()
        print(f"image={path}")
        print(
            "pipeline="
            f"trim={config['trim_level']} "
            f"smooth={config['smooth_level']} "
            f"inner_w={config['inner_rect_w_scale']:.2f} "
            f"inner_h={config['inner_rect_h_scale']:.2f} "
            f"hue_margin={config['color_gate_h_margin']:.1f} "
            f"sat_margin={config['color_gate_s_margin']} "
            f"val_margin={config['color_gate_v_margin']} "
            f"high_v_pad={config['color_gate_high_v_pad']} "
            f"keep={config['color_keep_min_ratio']:.2f} "
            f"def_max={config['inner_defect_max_hole_ratio']:.2f} "
            f"def_touch={config['inner_defect_min_touch_ratio']:.2f} "
            f"def_strong={config['inner_defect_strong_touch_ratio']:.2f}"
        )

    def change_image(self, delta: int) -> None:
        if not self.image_paths:
            return
        self.index = (self.index + delta) % len(self.image_paths)
        self.load_current_image()
        self.request_compute(immediate=True)

    def prev_image(self) -> None:
        self.change_image(-1)

    def next_image(self) -> None:
        self.change_image(1)

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.request_event.set()
        if self.compute_job is not None:
            self.root.after_cancel(self.compute_job)
            self.compute_job = None
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
            self.render_job = None
        restore_pipeline_defaults()
        self.root.after(10, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = collect_images(Path(args.input_dir))

    if not image_paths:
        raise SystemExit("열 수 있는 이미지가 없습니다.")

    app = TrackbarApp(image_paths)
    app.run()


if __name__ == "__main__":
    main()
