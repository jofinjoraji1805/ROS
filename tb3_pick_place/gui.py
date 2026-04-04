#!/usr/bin/env python3
"""
gui.py -- PyQt5 interface for two-phase pick-and-place system.

Layout:
  Top:    Status bar (state, phase, LIDAR, task indicators)
  Bottom: Camera (left) | Map (centre) | Controls + Log (right)
"""

import cv2
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFrame, QProgressBar,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from .config import (
    ST_IDLE, ST_NAV_TO_CUBE, ST_SEARCH_OBJECT, ST_ALIGN_OBJECT,
    ST_APPROACH_OBJECT, ST_ADJUST_POSITION, ST_PICK_OBJECT,
    ST_BACKUP_PICK, ST_NAV_TO_ZONE, ST_SEARCH_DROP_ZONE,
    ST_APPROACH_DROP, ST_ADJUST_DROP, ST_PLACE_OBJECT,
    ST_BACKUP_DROP, ST_NEXT_OBJECT, ST_RETURN_HOME,
    ST_SEARCH_DOCK, ST_ALIGN_DOCK, ST_APPROACH_DOCK,
    ST_DOCK_CREEP, ST_PARKED, ST_DONE,
    MAP_SAVE_DIR,
)
from .explorer import Explorer

# ── State colour mapping ─────────────────────────────────────────────────
STATE_COLORS = {
    ST_IDLE: "#95a5a6", ST_NAV_TO_CUBE: "#1abc9c",
    ST_SEARCH_OBJECT: "#f39c12", ST_ALIGN_OBJECT: "#e67e22",
    ST_APPROACH_OBJECT: "#d35400", ST_ADJUST_POSITION: "#e74c3c",
    ST_PICK_OBJECT: "#c0392b", ST_BACKUP_PICK: "#9b59b6",
    ST_NAV_TO_ZONE: "#16a085", ST_SEARCH_DROP_ZONE: "#3498db",
    ST_APPROACH_DROP: "#2980b9", ST_ADJUST_DROP: "#2471a3",
    ST_PLACE_OBJECT: "#27ae60", ST_BACKUP_DROP: "#8e44ad",
    ST_NEXT_OBJECT: "#1abc9c", ST_RETURN_HOME: "#f1c40f",
    ST_SEARCH_DOCK: "#f39c12", ST_ALIGN_DOCK: "#e67e22",
    ST_APPROACH_DOCK: "#d35400", ST_DOCK_CREEP: "#e67e22",
    ST_PARKED: "#2ecc71", ST_DONE: "#2ecc71",
}
COLOUR_HEX = {"RED": "#e74c3c", "BLUE": "#3498db", "GREEN": "#2ecc71"}

_DOCK_STATES = (ST_RETURN_HOME, ST_SEARCH_DOCK, ST_ALIGN_DOCK,
                ST_APPROACH_DOCK, ST_DOCK_CREEP, ST_PARKED, ST_DONE)

_STATE_LABELS = {
    ST_RETURN_HOME: "Returning Home",
    ST_SEARCH_DOCK: "Searching for Charging Dock",
    ST_ALIGN_DOCK: "Aligning to Charging Dock",
    ST_APPROACH_DOCK: "Approaching Charging Dock",
    ST_DOCK_CREEP: "Docking...",
    ST_PARKED: "Charging -- Mission Complete!",
    ST_NAV_TO_CUBE: "Navigating to Cube",
    ST_SEARCH_OBJECT: "Searching for Cube",
    ST_ALIGN_OBJECT: "Aligning to Cube",
    ST_APPROACH_OBJECT: "Approaching Cube",
    ST_ADJUST_POSITION: "Fine Adjusting",
    ST_PICK_OBJECT: "Picking Cube",
    ST_BACKUP_PICK: "Backing Up",
    ST_NAV_TO_ZONE: "Navigating to Drop Zone",
    ST_SEARCH_DROP_ZONE: "Searching Drop Zone",
    ST_APPROACH_DROP: "Approaching Drop Zone",
    ST_ADJUST_DROP: "Adjusting at Drop Zone",
    ST_PLACE_OBJECT: "Placing Cube",
    ST_BACKUP_DROP: "Backing Up from Drop",
    ST_NEXT_OBJECT: "Next Cube...",
}

# ── Style constants ──────────────────────────────────────────────────────
_BG = "#080818"
_PANEL = "#0d0d24"
_BORDER = "#1e1e3a"
_ACCENT = "#e94560"

_FRAME_SS = (
    f"QFrame{{background:{_PANEL};border:1px solid {_BORDER};"
    "border-radius:10px}}")

_BTN_TPL = (
    "QPushButton{{background:{bg};color:{fg};padding:{pad};"
    "border-radius:8px;border:none;font-weight:bold;"
    "font-size:{fs}}}"
    "QPushButton:hover{{background:{hov}}}"
    "QPushButton:pressed{{background:{press}}}"
    "QPushButton:disabled{{background:#222;color:#555}}")


def _btn(bg, fg="white", pad="12px", hov=None, press=None, fs="12px"):
    return _BTN_TPL.format(
        bg=bg, fg=fg, pad=pad,
        hov=hov or bg, press=press or (hov or bg), fs=fs)


class PickPlaceGUI(QMainWindow):

    def __init__(self, controller):
        super().__init__()
        self.ctrl = controller
        self.setWindowTitle("TurtleBot3 Autonomous Pick & Place")
        self.setMinimumSize(1500, 750)
        self.setStyleSheet(f"background-color:{_BG};")
        self.setFocusPolicy(Qt.StrongFocus)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ═══════════════════════════════════════════════════════════════
        # TOP: Status Bar
        # ═══════════════════════════════════════════════════════════════
        top = QFrame()
        top.setStyleSheet(
            f"QFrame{{background:{_PANEL};border:1px solid {_BORDER};"
            "border-radius:10px}")
        top.setFixedHeight(90)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 6, 16, 6)
        tl.setSpacing(12)

        # -- State label (large)
        self.state_label = QLabel("IDLE")
        self.state_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setMinimumWidth(350)
        self.state_label.setStyleSheet(
            f"padding:10px;background:#16213e;color:#95a5a6;"
            f"border-radius:10px;border:2px solid #333")
        tl.addWidget(self.state_label, stretch=3)

        # -- Phase badge
        self.phase_label = QLabel("IDLE")
        self.phase_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setFixedWidth(120)
        self.phase_label.setStyleSheet(
            "padding:8px;background:#1a1a3e;color:#7ec8e3;"
            "border-radius:8px;border:1px solid #2980b9")
        tl.addWidget(self.phase_label)

        # -- LIDAR badge
        self.lidar_label = QLabel("LIDAR: --")
        self.lidar_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.lidar_label.setAlignment(Qt.AlignCenter)
        self.lidar_label.setFixedWidth(150)
        self.lidar_label.setStyleSheet(
            "padding:8px;background:#1a1a3e;color:#2ecc71;"
            "border-radius:8px;border:1px solid #27ae60")
        tl.addWidget(self.lidar_label)

        # -- Task indicators (R / B / G)
        ind_frame = QFrame()
        ind_frame.setStyleSheet("border:none;background:transparent")
        il = QHBoxLayout(ind_frame)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(6)
        self.indicators = {}
        for c in ["RED", "BLUE", "GREEN"]:
            hx = COLOUR_HEX[c]
            lb = QLabel(f"  {c}  ")
            lb.setFont(QFont("Segoe UI", 10, QFont.Bold))
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet(
                f"background:#1a1a2e;color:{hx};"
                f"border:2px solid #333;border-radius:8px;padding:6px")
            il.addWidget(lb)
            self.indicators[c] = lb
        tl.addWidget(ind_frame)

        # -- Detection info
        self.det_label = QLabel("--")
        self.det_label.setFont(QFont("Segoe UI", 9))
        self.det_label.setAlignment(Qt.AlignCenter)
        self.det_label.setFixedWidth(180)
        self.det_label.setStyleSheet(
            "color:#888;border:none;background:transparent")
        tl.addWidget(self.det_label)

        root.addWidget(top)

        # ═══════════════════════════════════════════════════════════════
        # BOTTOM: Camera | Map | Controls
        # ═══════════════════════════════════════════════════════════════
        bottom = QHBoxLayout()
        bottom.setSpacing(6)

        # ─── LEFT: Camera ────────────────────────────────────────────
        cam_frame = QFrame()
        cam_frame.setStyleSheet(_FRAME_SS)
        cl = QVBoxLayout(cam_frame)
        cl.setContentsMargins(6, 6, 6, 6)

        ct = QLabel("Camera (YOLO)")
        ct.setFont(QFont("Segoe UI", 10, QFont.Bold))
        ct.setStyleSheet("color:#7ec8e3;border:none")
        ct.setAlignment(Qt.AlignCenter)
        cl.addWidget(ct)

        self.cam_label = QLabel("Waiting for camera...")
        self.cam_label.setMinimumSize(520, 380)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("color:#444;border:none")
        cl.addWidget(self.cam_label, stretch=1)
        bottom.addWidget(cam_frame, stretch=4)

        # ─── CENTRE: Map ─────────────────────────────────────────────
        map_frame = QFrame()
        map_frame.setStyleSheet(_FRAME_SS)
        ml = QVBoxLayout(map_frame)
        ml.setContentsMargins(6, 6, 6, 6)

        mt = QLabel("SLAM / Navigation Map")
        mt.setFont(QFont("Segoe UI", 10, QFont.Bold))
        mt.setStyleSheet("color:#7ec8e3;border:none")
        mt.setAlignment(Qt.AlignCenter)
        ml.addWidget(mt)

        self.map_label = QLabel("Map not built yet")
        self.map_label.setMinimumSize(320, 320)
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setStyleSheet("color:#444;border:none")
        self.map_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        ml.addWidget(self.map_label, stretch=1)

        # Exploration progress bar
        self.explore_progress = QProgressBar()
        self.explore_progress.setRange(0, 100)
        self.explore_progress.setValue(0)
        self.explore_progress.setFixedHeight(22)
        self.explore_progress.setStyleSheet(
            "QProgressBar{background:#1a1a2e;border:1px solid #333;"
            "border-radius:6px;text-align:center;color:#aaa}"
            "QProgressBar::chunk{background:qlineargradient("
            "x1:0,y1:0,x2:1,y2:0,stop:0 #1abc9c,stop:1 #16a085);"
            "border-radius:5px}")
        self.explore_progress.setFormat("Exploration: %p%")
        ml.addWidget(self.explore_progress)

        bottom.addWidget(map_frame, stretch=3)

        # ─── RIGHT: Controls ─────────────────────────────────────────
        panel = QFrame()
        panel.setStyleSheet(_FRAME_SS)
        pl = QVBoxLayout(panel)
        pl.setSpacing(8)
        pl.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("Mission Control")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title.setStyleSheet(f"color:{_ACCENT};border:none")
        title.setAlignment(Qt.AlignCenter)
        pl.addWidget(title)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(2)
        sep.setStyleSheet(f"background:{_BORDER};border:none")
        pl.addWidget(sep)

        # ── Phase 1: BUILD MAP button ────────────────────────────────
        self.map_btn = QPushButton("1.  BUILD MAP")
        self.map_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.map_btn.setCursor(Qt.PointingHandCursor)
        self.map_btn.setStyleSheet(_btn(
            "#1abc9c", hov="#16a085", press="#0e8c73", fs="13px"))
        self.map_btn.clicked.connect(self._on_map)
        pl.addWidget(self.map_btn)

        # ── Phase 2: START MISSION button ────────────────────────────
        self.mission_btn = QPushButton("2.  START MISSION")
        self.mission_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.mission_btn.setCursor(Qt.PointingHandCursor)
        self.mission_btn.setStyleSheet(_btn(
            _ACCENT, hov="#c0392b", press="#a93226", fs="13px"))
        self.mission_btn.clicked.connect(self._on_mission)
        pl.addWidget(self.mission_btn)

        # ── STOP button ──────────────────────────────────────────────
        self.stop_btn = QPushButton("STOP ALL")
        self.stop_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet(_btn(
            "#e67e22", hov="#d35400", press="#ba4a00", fs="12px",
            pad="10px"))
        self.stop_btn.clicked.connect(self._on_stop)
        pl.addWidget(self.stop_btn)

        # Spacer
        pl.addSpacing(4)

        # ── Log ──────────────────────────────────────────────────────
        log_title = QLabel("Event Log")
        log_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        log_title.setStyleSheet("color:#666;border:none")
        log_title.setAlignment(Qt.AlignLeft)
        pl.addWidget(log_title)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 8))
        self.log_area.setStyleSheet(
            "background:#0a0a1e;color:#ccc;border-radius:8px;"
            "padding:6px;border:1px solid #1a1a3a")
        pl.addWidget(self.log_area, stretch=1)

        bottom.addWidget(panel, stretch=2)
        root.addLayout(bottom, stretch=1)

        # ── State ────────────────────────────────────────────────────
        self._last_status = ""
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(33)

        # Check if map already exists
        if os.path.exists(os.path.join(MAP_SAVE_DIR, "map_grid.npy")):
            self.log_area.append("Saved map found! You can skip to MISSION.")
            self.explore_progress.setValue(100)
            self.explore_progress.setFormat("Map available!")

    # ── Button handlers ──────────────────────────────────────────────

    def _on_map(self):
        if self.ctrl.phase == self.ctrl.PHASE_IDLE:
            self.ctrl.start_mapping()
            self.map_btn.setText("MAPPING...")
            self.map_btn.setEnabled(False)
            self.log_area.append("Phase 1: Building SLAM map...")

    def _on_mission(self):
        if self.ctrl.phase == self.ctrl.PHASE_IDLE:
            self.ctrl.start_mission()
            self.mission_btn.setText("RUNNING...")
            self.mission_btn.setEnabled(False)
            self.log_area.append("Phase 2: RED -> BLUE -> GREEN -> DOCK")

    def _on_stop(self):
        if self.ctrl.phase == self.ctrl.PHASE_MAPPING:
            self.ctrl.stop_mapping()
        elif self.ctrl.phase == self.ctrl.PHASE_MISSION:
            self.ctrl.stop_mission()
        self.map_btn.setText("1.  BUILD MAP")
        self.map_btn.setEnabled(True)
        self.mission_btn.setText("2.  START MISSION")
        self.mission_btn.setEnabled(True)
        self.log_area.append("STOPPED")

    # ── GUI refresh (30 fps) ─────────────────────────────────────────

    def _update(self):
        phase = self.ctrl.phase
        fsm = self.ctrl.fsm
        explorer = self.ctrl.explorer
        navigator = self.ctrl.navigator

        # ── Phase badge ──────────────────────────────────────────────
        _phase_colors = {
            self.ctrl.PHASE_IDLE: ("#7ec8e3", "#2980b9"),
            self.ctrl.PHASE_MAPPING: ("#1abc9c", "#16a085"),
            self.ctrl.PHASE_MISSION: ("#e94560", "#c0392b"),
        }
        pc, pb = _phase_colors.get(phase, ("#7ec8e3", "#2980b9"))
        self.phase_label.setText(phase)
        self.phase_label.setStyleSheet(
            f"padding:8px;background:#1a1a3e;color:{pc};"
            f"border-radius:8px;border:1px solid {pb}")

        # ── Camera feed ──────────────────────────────────────────────
        with self.ctrl.lock:
            frame = self.ctrl.annotated_frame
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.cam_label.setPixmap(
                QPixmap.fromImage(qi).scaled(
                    self.cam_label.width(), self.cam_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # ── Map display ──────────────────────────────────────────────
        ox, oy, oyaw = self.ctrl._get_odom()

        if phase == self.ctrl.PHASE_MISSION and navigator.map_loaded:
            map_img = navigator.get_nav_map_image(ox, oy, oyaw)
        else:
            map_img = self.ctrl.lidar.get_map_image(ox, oy, oyaw)

        if map_img is not None:
            mr = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
            mh, mw, mch = mr.shape
            mqi = QImage(mr.data, mw, mh, mch * mw, QImage.Format_RGB888)
            self.map_label.setPixmap(
                QPixmap.fromImage(mqi).scaled(
                    self.map_label.width(), self.map_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # ── LIDAR distance ───────────────────────────────────────────
        front_d = self.ctrl.lidar.get_front_distance()
        if front_d < 0.30:
            lc, lb = "#e74c3c", "#c0392b"
        elif front_d < 0.50:
            lc, lb = "#f39c12", "#e67e22"
        else:
            lc, lb = "#2ecc71", "#27ae60"
        self.lidar_label.setText(f"LIDAR: {front_d:.2f}m")
        self.lidar_label.setStyleSheet(
            f"padding:8px;background:#1a1a3e;color:{lc};"
            f"border-radius:8px;border:1px solid {lb}")

        # ── Exploration progress ─────────────────────────────────────
        if phase == self.ctrl.PHASE_MAPPING:
            pct = int(explorer.progress * 100)
            self.explore_progress.setValue(pct)
            self.explore_progress.setFormat(f"Exploring: {pct}%")
        elif explorer.state == Explorer.ST_DONE:
            self.explore_progress.setValue(100)
            self.explore_progress.setFormat("Map saved!")
            self.map_btn.setText("1.  BUILD MAP")
            self.map_btn.setEnabled(True)

        # ── State display ────────────────────────────────────────────
        if phase == self.ctrl.PHASE_MAPPING:
            st_text = explorer.state.replace("_", " ")
            sc = "#1abc9c"
            status = explorer.status_text
        elif phase == self.ctrl.PHASE_MISSION:
            st_text = _STATE_LABELS.get(
                fsm.state, fsm.state.replace("_", " "))
            sc = STATE_COLORS.get(fsm.state, "#ecf0f1")
            status = fsm.status_text
        else:
            st_text = "IDLE"
            sc = "#95a5a6"
            status = "Ready"

        self.state_label.setText(st_text)
        self.state_label.setStyleSheet(
            f"padding:10px;background:#16213e;color:{sc};"
            f"border-radius:10px;font-size:16px;font-weight:bold;"
            f"border:2px solid {sc}")

        self.det_label.setText(
            f"{fsm.detected_label}\n{fsm.detected_distance}")

        # ── Colour indicators ────────────────────────────────────────
        ti = fsm.task_idx
        nt = len(fsm.tasks)
        cp = min(ti, nt)
        for i, c in enumerate(["RED", "BLUE", "GREEN"]):
            hx = COLOUR_HEX[c]
            lb = self.indicators[c]
            if i < cp:
                lb.setStyleSheet(
                    "background:#143d14;color:#2ecc71;"
                    "border:2px solid #2ecc71;border-radius:8px;padding:6px")
            elif i == ti and fsm.state not in _DOCK_STATES + (ST_IDLE,):
                lb.setStyleSheet(
                    f"background:#1a1a3e;color:{hx};"
                    f"border:2px solid {hx};border-radius:8px;"
                    "padding:6px;font-weight:bold")
            else:
                lb.setStyleSheet(
                    f"background:#1a1a2e;color:{hx};"
                    "border:2px solid #333;border-radius:8px;padding:6px")

        # ── Log ──────────────────────────────────────────────────────
        if status != self._last_status:
            self._last_status = status
            self.log_area.append(status)
            sb = self.log_area.verticalScrollBar()
            sb.setValue(sb.maximum())

        # ── Mission complete ─────────────────────────────────────────
        if fsm.state in (ST_DONE, ST_PARKED) and phase == self.ctrl.PHASE_MISSION:
            self.mission_btn.setText("MISSION COMPLETE!")
            self.mission_btn.setStyleSheet(_btn(
                "#27ae60", hov="#27ae60", fs="13px"))
            self.mission_btn.setEnabled(False)

    def closeEvent(self, event):
        self.ctrl.motion.stop()
        self.ctrl.fsm.running = False
        self.ctrl.explorer.running = False
        self.ctrl.set_manual_mode(False)
        event.accept()
