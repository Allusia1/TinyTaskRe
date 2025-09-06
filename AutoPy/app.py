import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
try:
    import customtkinter as ctk  # type: ignore
except Exception:
    ctk = None  # type: ignore
from collections import defaultdict

try:
    from pynput import keyboard as pynput_keyboard
    from pynput import mouse as pynput_mouse
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pynput'. Install with: pip install -r requirements.txt"
    ) from exc

# Optional heavy deps for image detection during playback
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from PIL import ImageGrab  # type: ignore
    import mss  # type: ignore
    from skimage.feature import hog  # type: ignore
    import pyautogui  # type: ignore
    import pyscreeze  # type: ignore
except Exception:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    ImageGrab = None  # type: ignore
    mss = None  # type: ignore
    hog = None  # type: ignore
    pyautogui = None  # type: ignore
    pyscreeze = None  # type: ignore


def _enable_windows_dpi_awareness() -> None:
    """Enable DPI awareness on Windows so coordinates match physical pixels."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        # Try the most modern API first
        try:
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
            if user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
                return
        except Exception:
            pass

        # Fallbacks
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            return
        except Exception:
            pass

        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        # Best-effort only
        pass


def get_virtual_screen_bounds() -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, width, height) of the virtual desktop.

    On Windows, uses GetSystemMetrics to include multi-monitor layouts.
    On other platforms, returns None (no scaling applied).
    """
    if sys.platform == "win32":
        try:
            import ctypes

            SM_XVIRTUALSCREEN = 76
            SM_YVIRTUALSCREEN = 77
            SM_CXVIRTUALSCREEN = 78
            SM_CYVIRTUALSCREEN = 79
            user32 = ctypes.windll.user32
            x = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
            y = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
            w = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
            h = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
            return (x, y, w, h)
        except Exception:
            return None
    return None


@dataclass
class RecordedEvent:
    """Represents an input event with timing metadata.

    event_type:
        'mouse_move' | 'mouse_click' | 'mouse_scroll' | 'key_press' | 'key_release'
    time_delta:
        Seconds since the previous recorded event.
    data:
        Event-specific payload.
    """

    event_type: str
    time_delta: float
    data: Dict[str, Any]


def key_to_string(key: Any) -> str:
    """Convert pynput key to reproducible string.

    For character keys, returns the character. For special keys, returns 'Key.xxx'.
    """

    try:
        # alphanumeric and symbol keys
        return key.char  # type: ignore[attr-defined]
    except Exception:
        return str(key)  # e.g. 'Key.esc', 'Key.shift'


def string_to_key(key_str: str) -> Any:
    """Convert string representation back to pynput key.

    - If looks like 'Key.xxx', map via pynput_keyboard.Key[xxx]
    - Otherwise treat as a literal character
    """

    if key_str.startswith("Key."):
        name = key_str.split(".", 1)[1]
        try:
            return getattr(pynput_keyboard.Key, name)
        except AttributeError:
            # Fallback: best-effort to return string
            return key_str
    # Single character
    return key_str


class Recorder:
    """Captures global mouse and keyboard events with timing."""

    def __init__(self) -> None:
        self._events: List[RecordedEvent] = []
        self._recording_lock = threading.Lock()
        self._is_recording = False
        self._start_time = 0.0
        self._last_time = 0.0
        self._mouse_listener: Optional[pynput_mouse.Listener] = None
        self._keyboard_listener: Optional[pynput_keyboard.Listener] = None
        self._stop_event = threading.Event()
        self._recorded_bounds: Optional[Tuple[int, int, int, int]] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def events(self) -> List[RecordedEvent]:
        # Return a copy to avoid external mutation
        with self._recording_lock:
            return list(self._events)

    @property
    def recorded_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        return self._recorded_bounds

    def clear(self) -> None:
        with self._recording_lock:
            self._events.clear()
        self._recorded_bounds = None

    def _now(self) -> float:
        return time.perf_counter()

    def _append_event(self, event_type: str, data: Dict[str, Any]) -> None:
        now = self._now()
        time_delta = now - self._last_time if self._last_time else 0.0
        self._last_time = now
        with self._recording_lock:
            self._events.append(RecordedEvent(event_type=event_type, time_delta=time_delta, data=data))

    def start(self) -> None:
        if self._is_recording:
            return
        self._is_recording = True
        self._stop_event.clear()
        self._start_time = self._now()
        self._last_time = 0.0
        # Capture the virtual screen bounds at the time of recording
        self._recorded_bounds = get_virtual_screen_bounds()

        def on_move(x: int, y: int) -> None:
            if not self._is_recording:
                return
            self._append_event("mouse_move", {"position": (x, y)})

        def on_click(x: int, y: int, button: Any, pressed: bool) -> None:
            if not self._is_recording:
                return
            self._append_event(
                "mouse_click",
                {
                    "position": (x, y),
                    "button": str(button),
                    "pressed": bool(pressed),
                },
            )

        def on_scroll(x: int, y: int, dx: int, dy: int) -> None:
            if not self._is_recording:
                return
            self._append_event("mouse_scroll", {"position": (x, y), "dx": int(dx), "dy": int(dy)})

        def on_press(key: Any) -> None:
            if not self._is_recording:
                return
            # Stop hotkey: F8 (not recorded)
            if str(key) == "Key.f8":
                self.stop()
                return
            key_str = key_to_string(key)
            self._append_event("key_press", {"key": key_str})

        def on_release(key: Any) -> None:
            if not self._is_recording:
                return
            if str(key) == "Key.f8":
                return  # already handled in on_press
            key_str = key_to_string(key)
            self._append_event("key_release", {"key": key_str})

        self._mouse_listener = pynput_mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        self._keyboard_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)

        self._mouse_listener.start()
        self._keyboard_listener.start()

    def stop(self) -> None:
        if not self._is_recording:
            return
        self._is_recording = False
        self._stop_event.set()
        try:
            if self._mouse_listener is not None:
                self._mouse_listener.stop()
                self._mouse_listener = None
        except Exception:
            pass
        try:
            if self._keyboard_listener is not None:
                self._keyboard_listener.stop()
                self._keyboard_listener = None
        except Exception:
            pass


class Player:
    """Replays recorded events either fully (play) or as a safe preview."""

    def __init__(self) -> None:
        self._mouse = pynput_mouse.Controller()
        self._keyboard = pynput_keyboard.Controller()
        self._is_playing = False
        self._stop_event = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def stop(self) -> None:
        self._stop_event.set()
        self._is_playing = False

    def play(
        self,
        events: List[RecordedEvent],
        *,
        preview_only: bool,
        recorded_bounds: Optional[Tuple[int, int, int, int]] = None,
        on_preview_note: Optional[callable] = None,
    ) -> None:
        """Play or preview the sequence.

        - If preview_only=True, only moves mouse and shows visual hints for other events.
        - If preview_only=False, performs all input actions.
        """

        if self._is_playing:
            return

        self._stop_event.clear()
        self._is_playing = True

        # Determine current bounds (for scaling), if available
        current_bounds = get_virtual_screen_bounds()

        def scale_point(pos: Tuple[int, int]) -> Tuple[int, int]:
            if recorded_bounds is None or current_bounds is None:
                return int(pos[0]), int(pos[1])
            rx, ry, rw, rh = recorded_bounds
            cx, cy, cw, ch = current_bounds
            if rw <= 0 or rh <= 0:
                return int(pos[0]), int(pos[1])
            sx = (pos[0] - rx) * (cw / rw) + cx
            sy = (pos[1] - ry) * (ch / rh) + cy
            return int(round(sx)), int(round(sy))

        # Image-based stop condition: if any image in images/ or Images/ is visible, stop
        def iter_images_dirs() -> List[Path]:
            app_dir = Path(__file__).resolve().parent
            dirs = []
            for name in ("images", "Images"):
                p = app_dir / name
                if p.exists() and p.is_dir():
                    dirs.append(p)
            return dirs

        def load_templates() -> List[Tuple[str, Any]]:
            templates: List[Tuple[str, Any]] = []
            if cv2 is None or np is None:
                return templates
            for d in iter_images_dirs():
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    for f in d.glob(ext):
                        try:
                            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                            if img is not None and img.size > 0:
                                templates.append((str(f), img))
                        except Exception:
                            continue
            return templates

        def grab_screen(bounds: Optional[Tuple[int, int, int, int]]) -> Optional[Any]:
            if ImageGrab is None:
                return None
            try:
                if bounds is not None:
                    x, y, w, h = bounds
                    bbox = (x, y, x + w, y + h)
                    shot = ImageGrab.grab(bbox=bbox)
                else:
                    # full primary screen only
                    shot = ImageGrab.grab()
                frame = np.array(shot)  # type: ignore[name-defined]
                frame = frame[:, :, ::-1]  # RGB -> BGR
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore[name-defined]
                return gray
            except Exception:
                return None

        def match_any_template(screen_gray: Any, templates: List[Tuple[str, Any]], threshold: float = 0.92) -> Optional[str]:
            try:
                for name, tmpl in templates:
                    h, w = tmpl.shape[:2]
                    H, W = screen_gray.shape[:2]
                    if h > H or w > W:
                        continue
                    res = cv2.matchTemplate(screen_gray, tmpl, cv2.TM_CCOEFF_NORMED)  # type: ignore[name-defined]
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # type: ignore[name-defined]
                    if max_val >= threshold:
                        return name
            except Exception:
                return None
            return None

        stop_templates = load_templates()

        def scanner_loop() -> None:
            if not stop_templates:
                return
            scan_bounds = current_bounds
            # Scan periodically until stopped or match found
            while not self._stop_event.is_set() and self._is_playing:
                screen = grab_screen(scan_bounds)
                if screen is not None:
                    hit = match_any_template(screen, stop_templates)
                    if hit is not None:
                        if on_preview_note is not None:
                            on_preview_note(f"Image detected: {Path(hit).name}. Stopping.")
                        self.stop()
                        break
                time.sleep(0.35)

        scanner_thread: Optional[threading.Thread] = None
        if stop_templates and cv2 is not None and ImageGrab is not None:
            scanner_thread = threading.Thread(target=scanner_loop, daemon=True)
            scanner_thread.start()

        for ev in events:
            if self._stop_event.is_set():
                break
            # Honor timing
            sleep_left = ev.time_delta
            # sleep in small chunks so we can react to stop quickly
            end_time = time.perf_counter() + max(0.0, sleep_left)
            while True:
                if self._stop_event.is_set():
                    break
                now = time.perf_counter()
                if now >= end_time:
                    break
                time.sleep(min(0.01, end_time - now))
            if self._stop_event.is_set():
                break

            # Execute event
            if ev.event_type == "mouse_move":
                pos = tuple(ev.data.get("position", (0, 0)))  # type: ignore[assignment]
                px, py = scale_point((int(pos[0]), int(pos[1])))
                try:
                    self._mouse.position = (px, py)  # type: ignore[assignment]
                except Exception:
                    pass
            elif ev.event_type == "mouse_click":
                button_str = ev.data.get("button", "Button.left")
                pressed = bool(ev.data.get("pressed", False))
                button_obj = getattr(pynput_mouse.Button, str(button_str).split(".")[-1], pynput_mouse.Button.left)
                if preview_only:
                    if on_preview_note is not None:
                        on_preview_note(f"Mouse {'down' if pressed else 'up'}: {button_obj.name}")
                else:
                    try:
                        if pressed:
                            self._mouse.press(button_obj)
                        else:
                            self._mouse.release(button_obj)
                    except Exception:
                        pass
            elif ev.event_type == "mouse_scroll":
                dx = int(ev.data.get("dx", 0))
                dy = int(ev.data.get("dy", 0))
                if preview_only:
                    if on_preview_note is not None:
                        on_preview_note(f"Scroll: dx={dx}, dy={dy}")
                else:
                    try:
                        self._mouse.scroll(dx, dy)
                    except Exception:
                        pass
            elif ev.event_type == "key_press":
                key_str = str(ev.data.get("key", ""))
                if preview_only:
                    if on_preview_note is not None:
                        on_preview_note(f"Key down: {key_str}")
                else:
                    try:
                        key_obj = string_to_key(key_str)
                        self._keyboard.press(key_obj)
                    except Exception:
                        pass
            elif ev.event_type == "key_release":
                key_str = str(ev.data.get("key", ""))
                if preview_only:
                    if on_preview_note is not None:
                        on_preview_note(f"Key up: {key_str}")
                else:
                    try:
                        key_obj = string_to_key(key_str)
                        self._keyboard.release(key_obj)
                    except Exception:
                        pass

        self._is_playing = False
        # Allow scanner thread to wind down
        if scanner_thread is not None and scanner_thread.is_alive():
            try:
                scanner_thread.join(timeout=0.2)
            except Exception:
                pass


class ImageWatcher:
    """Watches the screen and detects any template from a folder using ORB feature matching.

    Supports multi-scale templates and partial visibility via inlier ratio.
    On detection, calls on_detect(template_path).
    """

    def __init__(
        self,
        on_detect: callable,
        folder: Optional[Path] = None,
        ratio: float = 0.78,
        interval_s: float = 0.0,
        recursive: bool = True,
        min_visible_ratio: float = 0.3,
        min_template_dim: int = 96,
    ) -> None:
        self._on_detect = on_detect
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

        # ORB and matcher
        self._orb = None
        self._bf = None
        if cv2 is not None:
            try:
                self._orb = cv2.ORB_create(nfeatures=2000)  # type: ignore[attr-defined]
                self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # type: ignore[attr-defined]
            except Exception:
                self._orb = None
                self._bf = None

        # Each template variant for ORB: (path, kp, des, kp_count)
        self._templates: List[Tuple[str, Any, Any, int]] = []
        # Base grayscale images per template path (for fallback matching)
        self._base_templates: List[Tuple[str, Any]] = []
        # Edge templates per scale: (path, edges, hsv_hist, (w, h))
        self._edge_templates: List[Tuple[str, Any, Any, Tuple[int, int]]] = []
        # Simple file list for pyautogui-only fast scanning
        self._template_files: List[str] = []
        self._folder: Optional[Path] = folder
        self._ratio: float = ratio
        self._interval_s: float = interval_s
        self._recursive: bool = recursive
        self._min_visible_ratio: float = min_visible_ratio
        self._min_template_dim: int = max(32, int(min_template_dim))

    @property
    def is_running(self) -> bool:
        return self._is_running

    def update_config(
        self,
        *,
        folder: Optional[Path],
        ratio: Optional[float] = None,
        interval_s: Optional[float] = None,
        recursive: bool = True,
        min_visible_ratio: Optional[float] = None,
        min_template_dim: Optional[int] = None,
    ) -> None:
        self._folder = folder
        if ratio is not None:
            self._ratio = ratio
        if interval_s is not None:
            self._interval_s = interval_s
        self._recursive = recursive
        if min_visible_ratio is not None:
            self._min_visible_ratio = float(min_visible_ratio)
        if min_template_dim is not None:
            self._min_template_dim = max(32, int(min_template_dim))

    def _iter_image_files(self) -> List[Path]:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        result: List[Path] = []
        if self._folder is not None and self._folder.exists():
            if self._recursive:
                for ext in exts:
                    result.extend(self._folder.rglob(ext))
            else:
                for ext in exts:
                    result.extend(self._folder.glob(ext))
            return result
        # Fallback to default images/Images
        app_dir = Path(__file__).resolve().parent
        for name in ("images", "Images"):
            p = app_dir / name
            if p.exists() and p.is_dir():
                if self._recursive:
                    for ext in exts:
                        result.extend(p.rglob(ext))
                else:
                    for ext in exts:
                        result.extend(p.glob(ext))
        return result

    def _load_templates(self) -> None:
        # PyAutoGUI-only: keep only filenames for speed
        self._templates.clear()
        self._base_templates.clear()
        self._edge_templates.clear()
        self._template_files.clear()
        for f in self._iter_image_files():
            try:
                self._template_files.append(str(f))
            except Exception:
                continue

    def _grab_screen(self) -> Optional[Tuple[Any, Any, Any, Any, float, float]]:
        if ImageGrab is None or np is None or cv2 is None:
            return None
        # Try PIL first, then MSS as fallback for games/fullscreen
        frame_bgr = None
        try:
            shot = ImageGrab.grab()
            frame_bgr = np.array(shot)  # type: ignore[name-defined]
            frame_bgr = frame_bgr[:, :, ::-1]
        except Exception:
            pass
        if frame_bgr is None and mss is not None:
            try:
                with mss.mss() as sct:
                    mon = sct.monitors[0]
                    raw = sct.grab(mon)
                    frame_bgr = np.array(raw)[:, :, :3]
            except Exception:
                frame_bgr = None
        if frame_bgr is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # type: ignore[name-defined]
        # Normalize contrast for robustness
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # type: ignore[name-defined]
            gray = clahe.apply(gray)
        except Exception:
            pass
        # Downscale for fast coarse search
        scale = 0.5
        try:
            small_bgr = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # type: ignore[name-defined]
            small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # type: ignore[name-defined]
        except Exception:
            small_bgr, small_gray = frame_bgr, gray
            scale = 1.0
        ts = time.perf_counter()
        return gray, frame_bgr, small_gray, small_bgr, scale, ts

    def _detect_orb(self, screen_gray: Any) -> Optional[str]:
        if self._orb is None or self._bf is None or cv2 is None:
            return None
        try:
            kp2, des2 = self._orb.detectAndCompute(screen_gray, None)
            if des2 is None or len(kp2) < 12:
                return None
            for path, kp1, des1, kp1_count in self._templates:
                if des1 is None or kp1_count < 8:
                    continue
                matches = self._bf.knnMatch(des1, des2, k=2)  # type: ignore[arg-type]
                good = []
                for m, n in matches:
                    if m.distance < self._ratio * n.distance:
                        good.append(m)
                if len(good) < max(10, int(0.05 * kp1_count)):
                    continue
                # Homography to verify geometric consistency
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # type: ignore[name-defined]
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # type: ignore[name-defined]
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # type: ignore[name-defined]
                if H is None or mask is None:
                    continue
                inliers = int(mask.sum())
                visible_ratio = inliers / float(kp1_count)
                if inliers >= 8 and visible_ratio >= self._min_visible_ratio:
                    return path
        except Exception:
            return None
        return None

    def _nms(self, boxes: List[Tuple[int, int, int, int, float]], iou_thresh: float = 0.4) -> List[Tuple[int, int, int, int, float]]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        picked: List[Tuple[int, int, int, int, float]] = []
        def iou(a: Tuple[int, int, int, int, float], b: Tuple[int, int, int, int, float]) -> float:
            ax1, ay1, aw, ah, _ = a
            bx1, by1, bw, bh, _ = b
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh
            inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0, min(ay2, by2) - max(ay1, by1))
            inter = inter_w * inter_h
            if inter == 0:
                return 0.0
            area_a = aw * ah
            area_b = bw * bh
            union = area_a + area_b - inter
            return inter / float(max(union, 1))
        while boxes:
            best = boxes.pop(0)
            picked.append(best)
            boxes = [b for b in boxes if iou(best, b) < iou_thresh]
        return picked

    def _detect_edges_color(self, screen_edges_small: Any, screen_bgr_full: Any, scale: float) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        results: List[Tuple[str, Tuple[int, int, int, int], float]] = []
        if cv2 is None:
            return results
        try:
            Hs, Ws = screen_edges_small.shape[:2]
            for path, tmpl_edges, tmpl_hist, (tw, th) in self._edge_templates:
                if th > Hs or tw > Ws:
                    continue
                res = cv2.matchTemplate(screen_edges_small, tmpl_edges, cv2.TM_CCOEFF_NORMED)  # type: ignore[name-defined]
                # Find all locations above threshold
                edge_thresh = 0.43
                ys, xs = (res >= edge_thresh).nonzero()
                if len(xs) == 0:
                    # take the top single if nothing crosses threshold
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # type: ignore[name-defined]
                    if max_val >= edge_thresh:
                        xs = np.array([max_loc[0]])  # type: ignore[name-defined]
                        ys = np.array([max_loc[1]])  # type: ignore[name-defined]
                # Build candidate boxes
                candidates: List[Tuple[int, int, int, int, float]] = []
                for x, y in zip(xs.tolist(), ys.tolist()):
                    if y + th > Hs or x + tw > Ws:
                        continue
                    score = float(res[y, x])
                    candidates.append((x, y, tw, th, score))
                # NMS to prune overlaps
                pruned = self._nms(candidates, iou_thresh=0.45)
                for x, y, w, h, score in pruned:
                    # Optional color check; keep loose to accept style variations
                    try:
                        # scale candidate to full-res
                        fx = int(round(x / scale))
                        fy = int(round(y / scale))
                        fw = int(round(w / scale))
                        fh = int(round(h / scale))
                        patch = screen_bgr_full[fy : fy + fh, fx : fx + fw]
                        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)  # type: ignore[name-defined]
                        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])  # type: ignore[name-defined]
                        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)  # type: ignore[name-defined]
                        corr = cv2.compareHist(tmpl_hist, hist, cv2.HISTCMP_CORREL)  # type: ignore[name-defined]
                        if corr < 0.3:
                            continue
                        results.append((path, (fx, fy, fw, fh), score))
                    except Exception:
                        pass
        except Exception:
            return results
        return results

    def _fallback_template_match(self, screen_gray: Any) -> Optional[str]:
        if cv2 is None:
            return None
        try:
            H, W = screen_gray.shape[:2]
            for path, base in self._base_templates:
                h, w = base.shape[:2]
                # Build a modest set of scales
                scales = [0.5, 0.75, 1.0, 1.25, 1.5]
                min_dim = min(h, w)
                if min_dim < 96:
                    s = 96.0 / float(min_dim)
                    if s not in scales:
                        scales.append(s)
                for s in scales:
                    th, tw = int(round(h * s)), int(round(w * s))
                    if th < 24 or tw < 24 or th > H or tw > W:
                        continue
                    templ = base if abs(s - 1.0) < 1e-3 else cv2.resize(base, (tw, th), interpolation=cv2.INTER_CUBIC)  # type: ignore[name-defined]
                    res = cv2.matchTemplate(screen_gray, templ, cv2.TM_CCOEFF_NORMED)  # type: ignore[name-defined]
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # type: ignore[name-defined]
                    if max_val >= 0.90:
                        return path
        except Exception:
            return None
        return None

    def _pyautogui_match(self) -> Optional[str]:
        if pyautogui is None or pyscreeze is None:
            return None
        try:
            # Use pyautogui locateAllOnScreen for each template file at confidence
            # Use smaller region to speed up is not provided; rely on internal optimizations
            for path, base in self._base_templates:
                try:
                    # pyscreeze uses grayscale and confidence via OpenCV if available
                    locations = list(pyautogui.locateAllOnScreen(path, grayscale=True, confidence=0.75))
                    if locations:
                        return path
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _verify_present(self, path: str) -> bool:
        # Quick re-grab and check with template and edges to avoid stale reports
        snap = self._grab_screen()
        if snap is None:
            return False
        screen_gray, screen_bgr, small_gray, small_bgr, scale, ts = snap

        # 1) Use PyAutoGUI high-level locate if available (simple and robust for verification)
        try:
            if pyautogui is not None:
                box = pyautogui.locateOnScreen(path, grayscale=True, confidence=0.72)
                if box is not None:
                    return True
        except Exception:
            pass

        # 2) CV2 template matching just for this specific path (multi-scale)
        try:
            base = None
            for p, b in self._base_templates:
                if p == path:
                    base = b
                    break
            if base is not None:
                h, w = base.shape[:2]
                scales = [0.5, 0.75, 1.0, 1.25, 1.5]
                for s in scales:
                    th, tw = int(round(h * s)), int(round(w * s))
                    if th < 24 or tw < 24 or th > small_gray.shape[0] or tw > small_gray.shape[1]:
                        continue
                    templ = base if abs(s - 1.0) < 1e-3 else cv2.resize(base, (tw, th), interpolation=cv2.INTER_CUBIC)  # type: ignore[name-defined]
                    res = cv2.matchTemplate(small_gray, templ, cv2.TM_CCOEFF_NORMED)  # type: ignore[name-defined]
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # type: ignore[name-defined]
                    if max_val >= 0.86:
                        return True
        except Exception:
            pass

        # 3) Edge + color check for this specific path (downscaled)
        try:
            blurred = cv2.GaussianBlur(small_gray, (5, 5), 0)  # type: ignore[name-defined]
            screen_edges_small = cv2.Canny(blurred, 50, 150)  # type: ignore[name-defined]
            for pth, tmpl_edges, tmpl_hist, (tw, th) in self._edge_templates:
                if pth != path:
                    continue
                if th > screen_edges_small.shape[0] or tw > screen_edges_small.shape[1]:
                    continue
                res = cv2.matchTemplate(screen_edges_small, tmpl_edges, cv2.TM_CCOEFF_NORMED)  # type: ignore[name-defined]
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # type: ignore[name-defined]
                if max_val >= 0.46:
                    return True
        except Exception:
            pass

        return False

    def _loop(self) -> None:
        self._is_running = True
        try:
            self._load_templates()
            if not self._template_files:
                # Nothing to watch
                self._is_running = False
                return
            # PyAutoGUI-only scanning loop for max speed
            pyautogui.PAUSE = 0
            try:
                pyautogui.FAILSAFE = False
            except Exception:
                pass

            last_click_ts = 0.0
            debounce_s = 0.3
            cooldown_until = 0.0  # after a detection, wait 10s

            while not self._stop_event.is_set():
                # Handle cooldown after a successful detection
                now_time = time.perf_counter()
                if now_time < cooldown_until:
                    time.sleep(min(0.1, cooldown_until - now_time))
                    continue
                hit_path: Optional[str] = None
                for path in self._template_files:
                    try:
                        box = pyautogui.locateOnScreen(path, grayscale=True, confidence=self._ratio)
                        if box is None:
                            continue
                        # Strict verification on a small expanded region
                        l, t, w, h = box
                        pad = max(6, int(min(w, h) * 0.15))
                        region = (max(0, l - pad), max(0, t - pad), w + 2 * pad, h + 2 * pad)
                        strict_conf = min(0.98, max(self._ratio + 0.15, 0.90))
                        box2 = pyautogui.locateOnScreen(path, grayscale=True, confidence=strict_conf, region=region)
                        if box2 is None:
                            continue
                        # Double-check consecutive frame
                        box3 = pyautogui.locateOnScreen(path, grayscale=True, confidence=strict_conf, region=region)
                        if box3 is None:
                            continue
                        # Debounce and click center
                        now = time.perf_counter()
                        if now - last_click_ts < debounce_s:
                            hit_path = path
                            break
                        # Do not move or click; just trigger hotkeys via _on_image_detected
                        last_click_ts = now
                        hit_path = path
                        break
                    except Exception:
                        continue
                if hit_path is not None:
                    try:
                        self._on_detect(hit_path)
                    except Exception:
                        pass
                    # Enter 10s cooldown before next detection
                    cooldown_until = time.perf_counter() + 10.0
                    continue
                time.sleep(0.0005)
        finally:
            self._is_running = False

    def start(self) -> bool:
        if self._is_running:
            return True
        # Allow starting even without cv2/ImageGrab; pyautogui path requires only pyautogui
        if pyautogui is None:
            return False
        # Ensure we have templates to look for before spinning the thread
        self._load_templates()
        if not getattr(self, "_template_files", []):
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.2)
            except Exception:
                pass


class Overlay:
    """A small always-on-top bubble to show preview hints near the cursor."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._top: Optional[tk.Toplevel] = None
        self._label: Optional[ttk.Label] = None
        self._hide_after_id: Optional[str] = None
        self._create()

    def _create(self) -> None:
        top = tk.Toplevel(self.root)
        top.withdraw()
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        # Style
        frame = ttk.Frame(top, padding=(8, 6))
        frame.pack(fill=tk.BOTH, expand=True)
        label = ttk.Label(frame, text="", justify=tk.LEFT)
        label.pack()
        self._top = top
        self._label = label

    def show_note(self, text: str, duration_ms: int = 600) -> None:
        if self._top is None or self._label is None:
            return
        # Position near current mouse location
        try:
            x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        except Exception:
            x, y = 100, 100
        self._label.config(text=text)
        self._top.geometry(f"+{x + 16}+{y + 16}")
        self._top.deiconify()
        self._top.lift()
        if self._hide_after_id is not None:
            try:
                self._top.after_cancel(self._hide_after_id)
            except Exception:
                pass
        self._hide_after_id = self._top.after(duration_ms, self.hide)

    def hide(self) -> None:
        if self._top is None:
            return
        try:
            self._top.withdraw()
        except Exception:
            pass


class TinyTaskApp:
    """Minimal UI focused on watching for images and simulating actions visually."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        _enable_windows_dpi_awareness()
        self.overlay = Overlay(root)
        self.watcher = ImageWatcher(self._on_image_detected)

        self._apply_style()
        self._build_ui()
        self._refresh_status()

    def _apply_style(self) -> None:
        try:
            if ctk is not None:
                ctk.set_appearance_mode("dark")
                ctk.set_default_color_theme("green")
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.root.title("TonyTask")
        self.root.geometry("520x360")
        self.root.resizable(False, False)

        if ctk is not None:
            container = ctk.CTkFrame(self.root, corner_radius=12)
            container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        else:
            container = ttk.Frame(self.root, padding=(12, 12))
            container.pack(fill=tk.BOTH, expand=True)

        # Header removed per request

        # Row: folder picker and controls on a card
        if ctk is not None:
            controls = ctk.CTkFrame(container, corner_radius=12)
            controls.pack(fill=tk.X, pady=(0, 12), padx=2)
        else:
            controls = ttk.Frame(container, padding=(10, 8))
            controls.pack(fill=tk.X, pady=(0, 12))

        self.folder_var = tk.StringVar(value=str((Path(__file__).parent / "images").resolve()))
        if ctk is not None:
            ctk.CTkLabel(controls, text="Folder:").grid(row=0, column=0, sticky=tk.W, padx=(4, 6), pady=6)
            self.folder_entry = ctk.CTkEntry(controls, textvariable=self.folder_var, width=360)
            self.folder_entry.grid(row=0, column=1, sticky=tk.W, pady=6)
            ctk.CTkButton(controls, text="Browse", command=self._on_browse_folder).grid(row=0, column=2, padx=6)
        else:
            ttk.Label(controls, text="Folder:").grid(row=0, column=0, sticky=tk.W, padx=(4, 6), pady=6)
            self.folder_entry = ttk.Entry(controls, textvariable=self.folder_var, width=46)
            self.folder_entry.grid(row=0, column=1, sticky=tk.W, pady=6)
            ttk.Button(controls, text="Browse", command=self._on_browse_folder).grid(row=0, column=2, padx=6)

        if ctk is not None:
            ctk.CTkLabel(controls, text="Sensitivity").grid(row=1, column=0, sticky=tk.W, padx=(4, 6))
            self.sensitivity_var = tk.DoubleVar(value=0.6)
            self.sensitivity = ctk.CTkSlider(controls, from_=0.6, to=0.9, number_of_steps=30, variable=self.sensitivity_var)
            self.sensitivity.grid(row=1, column=1, sticky="ew", pady=6)
        else:
            ttk.Label(controls, text="Sensitivity").grid(row=1, column=0, sticky=tk.W, padx=(4, 6))
            self.sensitivity_var = tk.DoubleVar(value=0.6)
            self.sensitivity = ttk.Scale(controls, from_=0.6, to=0.9, orient=tk.HORIZONTAL, variable=self.sensitivity_var)
            self.sensitivity.grid(row=1, column=1, sticky=tk.EW, pady=4)

        try:
            controls.grid_columnconfigure(1, weight=1)
        except Exception:
            pass

        # Row: action buttons
        if ctk is not None:
            action_row = ctk.CTkFrame(container, fg_color="transparent")
            action_row.pack(fill=tk.X)
            self.watch_btn = ctk.CTkButton(action_row, text="Start Watch", command=self._on_toggle_watch)
            self.always_on_top_var = tk.BooleanVar(value=False)
            self.topmost_chk = ctk.CTkCheckBox(action_row, text="Always on top", variable=self.always_on_top_var, command=self._on_toggle_topmost)
            self.watch_btn.pack(side=tk.LEFT, padx=6)
            self.topmost_chk.pack(side=tk.LEFT, padx=12)
        else:
            action_row = ttk.Frame(container)
            action_row.pack(fill=tk.X)
            self.watch_btn = ttk.Button(action_row, text="Start Watch", command=self._on_toggle_watch)
            self.always_on_top_var = tk.BooleanVar(value=False)
            self.topmost_chk = ttk.Checkbutton(action_row, text="Always on top", variable=self.always_on_top_var, command=self._on_toggle_topmost)
            self.watch_btn.pack(side=tk.LEFT, padx=6)
            self.topmost_chk.pack(side=tk.LEFT, padx=6)

        # Log area
        if ctk is not None:
            log_frame = ctk.CTkFrame(container, corner_radius=12)
            log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
            ctk.CTkLabel(log_frame, text="Log", anchor="w").pack(fill=tk.X, padx=8, pady=(8, 0))
            self.log = ctk.CTkTextbox(log_frame, height=160)
            self.log.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            self.status_var = tk.StringVar(value="Idle")
            ctk.CTkLabel(container, textvariable=self.status_var, text_color="#9ca3af").pack(anchor="w", pady=(6, 0))
        else:
            log_frame = ttk.LabelFrame(container, text="Log")
            log_frame.pack(fill=tk.BOTH, expand=True)
            self.log = ScrolledText(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
            self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
            self.status_var = tk.StringVar(value="Idle")
            ttk.Label(container, textvariable=self.status_var).pack(anchor=tk.W, pady=(6, 0))


    def _set_buttons_state(self, *, recording: bool, playing: bool) -> None:
        # Only watch controls remain; nothing to toggle here beyond button text
        try:
            self.watch_btn.configure(text="Stop Watch" if self.watcher.is_running else "Start Watch")
        except Exception:
            self.watch_btn.config(text="Stop Watch" if self.watcher.is_running else "Start Watch")

    def _refresh_status(self) -> None:
        self.status_var.set("Watching..." if self.watcher.is_running else "Idle")
        self._set_buttons_state(recording=False, playing=False)
        self.root.after(200, self._refresh_status)

    def _on_toggle_topmost(self) -> None:
        try:
            self.root.attributes("-topmost", bool(self.always_on_top_var.get()))
        except Exception:
            pass

    # Recording/Playback removed

    def _on_toggle_watch(self) -> None:
        if self.watcher.is_running:
            self.watcher.stop()
            self.status_var.set("Watch stopped")
            try:
                self.watch_btn.configure(text="Start Watch")
            except Exception:
                self.watch_btn.config(text="Start Watch")
            return
        # Apply settings
        folder_path = Path(self.folder_var.get()).expanduser()
        ratio = float(self.sensitivity_var.get())
        self.watcher.update_config(
            folder=folder_path if folder_path.exists() else None,
            ratio=ratio,
            interval_s=None,
        )
        started = self.watcher.start()
        if not started:
            messagebox.showwarning("Unavailable", "Image watching requires OpenCV, NumPy, and Pillow.")
            return
        self.status_var.set("Watching...")
        try:
            self.watch_btn.configure(text="Stop Watch")
        except Exception:
            self.watch_btn.config(text="Stop Watch")

    def _on_image_detected(self, path: str) -> None:
        # Run on background thread; marshal to main thread
        def show_sequence() -> None:
            self._log(f"Detected: {Path(path).name}")
            # Give focus to target after the click performed by the watcher
            time.sleep(0.05)
            try:
                # Perform actual key combo: Ctrl + M
                pyautogui.hotkey("ctrl", "m")
            except Exception:
                pass
            self.overlay.show_note("Sent Ctrl+M")
            # Keep watching; cooldown is handled in the detection loop
            try:
                self.watch_btn.configure(text="Stop Watch")
            except Exception:
                self.watch_btn.config(text="Stop Watch")
        threading.Thread(target=show_sequence, daemon=True).start()

    def _on_browse_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.folder_var.get() or str(Path.cwd()))
        if chosen:
            self.folder_var.set(chosen)
            self._log(f"Folder set: {chosen}")

    def _log(self, text: str) -> None:
        try:
            try:
                self.log.configure(state="normal")
            except Exception:
                self.log.config(state=tk.NORMAL)
            self.log.insert("end", f"{time.strftime('%H:%M:%S')} - {text}\n")
            try:
                self.log.see("end")
            except Exception:
                pass
            try:
                self.log.configure(state="disabled")
            except Exception:
                self.log.config(state=tk.DISABLED)
        except Exception:
            pass


def main() -> None:
    if 'ctk' in globals() and ctk is not None:
        root = ctk.CTk()
        try:
            root.title("TonyTask")
            root.geometry("520x360")
            # Align root window background with dark UI
            root.configure(fg_color="#0f172a")
        except Exception:
            pass
    else:
        root = tk.Tk()
        # Use ttk style for a modern look (fallback)
        try:
            style = ttk.Style(root)
            if "vista" in style.theme_names():
                style.theme_use("vista")
            elif "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass
        try:
            root.title("TonyTask")
        except Exception:
            pass
    TinyTaskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


