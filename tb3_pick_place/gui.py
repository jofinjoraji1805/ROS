#!/usr/bin/env python3
"""
gui.py -- PyQt5 interface for two-phase pick-and-place system.

Layout (3 columns):
  Left:   Live camera feed with YOLO overlays
  Centre: LIDAR/SLAM map with robot position and planned paths
  Right:  Phase controls, state, logs, WASD teleop
"""

import cv2
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTextEdit, QFrame, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent

from .config import (
    ST_IDLE, ST_NAV_TO_CUBE, ST_SEARCH_OBJECT, ST_ALIGN_OBJECT,
    ST_APPROACH_OBJECT, ST_ADJUST_POSITION, ST_PICK_OBJECT,
    ST_BACKUP_PICK, ST_NAV_TO_ZONE, ST_SEARCH_DROP_ZONE,
    ST_APPROACH_DROP, ST_ADJUST_DROP, ST_PLACE_OBJECT,
    ST_BACKUP_DROP, ST_NEXT_OBJECT, ST_RETURN_HOME, ST_DONE,
    TELEOP_LINEAR, TELEOP_ANGULAR, MAP_SAVE_DIR,
)
from .explorer import Explorer

STATE_COLORS = {
    ST_IDLE: "#95a5a6", ST_NAV_TO_CUBE: "#1abc9c",
    ST_SEARCH_OBJECT: "#f39c12", ST_ALIGN_OBJECT: "#e67e22",
    ST_APPROACH_OBJECT: "#d35400", ST_ADJUST_POSITION: "#e74c3c",
    ST_PICK_OBJECT: "#c0392b", ST_BACKUP_PICK: "#9b59b6",
    ST_NAV_TO_ZONE: "#16a085", ST_SEARCH_DROP_ZONE: "#3498db",
    ST_APPROACH_DROP: "#2980b9", ST_ADJUST_DROP: "#2471a3",
    ST_PLACE_OBJECT: "#27ae60", ST_BACKUP_DROP: "#8e44ad",
    ST_NEXT_OBJECT: "#1abc9c", ST_RETURN_HOME: "#f1c40f",
    ST_DONE: "#2ecc71",
}
COLOUR_HEX = {"RED": "#e74c3c", "BLUE": "#3498db", "GREEN": "#2ecc71"}

_DARK_BG = "background-color: #0a0a1a;"
_FRAME = "QFrame{background:#111;border:2px solid #333;border-radius:10px}"
_BTN = (
    "QPushButton{{background:{bg};color:{fg};padding:{pad};"
    "border-radius:8px;border:none;font-weight:bold}}"
    "QPushButton:hover{{background:{hov}}}"
    "QPushButton:disabled{{background:#333;color:#666}}")

def _btn_ss(bg, fg="white", pad="10px", hov=None):
    return _BTN.format(bg=bg, fg=fg, pad=pad, hov=hov or bg)


class PickPlaceGUI(QMainWindow):

    def __init__(self, controller):
        super().__init__()
        self.ctrl = controller
        self.setWindowTitle("YOLO Pick & Place -- SLAM + Navigation")
        self.setMinimumSize(1450, 680)
        self.setStyleSheet(_DARK_BG)
        self.setFocusPolicy(Qt.StrongFocus)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ═══════════════ LEFT: Camera ═════════════════════════════════
        cam_frame = QFrame(); cam_frame.setStyleSheet(_FRAME)
        cl = QVBoxLayout(cam_frame); cl.setContentsMargins(4, 4, 4, 4)

        ct = QLabel("Camera (YOLO)")
        ct.setFont(QFont("Segoe UI", 10, QFont.Bold))
        ct.setStyleSheet("color:#aaa;border:none")
        ct.setAlignment(Qt.AlignCenter)
        cl.addWidget(ct)

        self.cam_label = QLabel("Waiting for camera...")
        self.cam_label.setMinimumSize(480, 360)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("color:#555;border:none")
        cl.addWidget(self.cam_label)
        root.addWidget(cam_frame, stretch=3)

        # ═══════════════ CENTRE: Map ═════════════════════════════════
        map_frame = QFrame(); map_frame.setStyleSheet(_FRAME)
        ml = QVBoxLayout(map_frame); ml.setContentsMargins(4, 4, 4, 4)

        mt = QLabel("SLAM / Navigation Map")
        mt.setFont(QFont("Segoe UI", 10, QFont.Bold))
        mt.setStyleSheet("color:#aaa;border:none")
        mt.setAlignment(Qt.AlignCenter)
        ml.addWidget(mt)

        self.map_label = QLabel("Map not built yet")
        self.map_label.setMinimumSize(350, 350)
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setStyleSheet("color:#555;border:none")
        ml.addWidget(self.map_label)

        self.lidar_label = QLabel("LIDAR: --")
        self.lidar_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.lidar_label.setAlignment(Qt.AlignCenter)
        self.lidar_label.setStyleSheet("color:#2ecc71;border:none;padding:4px")
        ml.addWidget(self.lidar_label)

        # Exploration progress bar
        self.explore_progress = QProgressBar()
        self.explore_progress.setRange(0, 100)
        self.explore_progress.setValue(0)
        self.explore_progress.setStyleSheet(
            "QProgressBar{background:#1a1a2e;border:1px solid #333;"
            "border-radius:6px;text-align:center;color:#aaa;height:20px}"
            "QProgressBar::chunk{background:#1abc9c;border-radius:5px}")
        self.explore_progress.setFormat("Exploration: %p%")
        ml.addWidget(self.explore_progress)

        root.addWidget(map_frame, stretch=2)

        # ═══════════════ RIGHT: Controls ═════════════════════════════
        panel = QFrame(); panel.setStyleSheet(_FRAME)
        pl = QVBoxLayout(panel)
        pl.setSpacing(5); pl.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("Mission Control")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color:#e94560;border:none")
        title.setAlignment(Qt.AlignCenter)
        pl.addWidget(title)

        # Phase label
        self.phase_label = QLabel("Phase: IDLE")
        self.phase_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setStyleSheet(
            "color:#7ec8e3;border:none;padding:4px")
        pl.addWidget(self.phase_label)

        # State label
        self.state_label = QLabel("IDLE")
        self.state_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.state_label.setWordWrap(True)
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setStyleSheet(
            "padding:8px;background:#16213e;color:#f39c12;"
            "border-radius:8px;border:1px solid #1a1a2e")
        pl.addWidget(self.state_label)

        # Detection info
        self.det_label = QLabel("Detected: --")
        self.det_label.setFont(QFont("Segoe UI", 9))
        self.det_label.setAlignment(Qt.AlignCenter)
        self.det_label.setStyleSheet("color:#bbb;border:none")
        pl.addWidget(self.det_label)

        # Colour indicators
        il = QHBoxLayout()
        self.indicators = {}
        for c in ["RED", "BLUE", "GREEN"]:
            hx = COLOUR_HEX[c]
            lb = QLabel(f" {c} ")
            lb.setFont(QFont("Segoe UI", 9, QFont.Bold))
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet(
                f"background:#222;color:{hx};"
                "border:2px solid #333;border-radius:6px;padding:3px")
            il.addWidget(lb); self.indicators[c] = lb
        pl.addLayout(il)

        # ── Phase 1: MAP button ──────────────────────────────────────
        self.map_btn = QPushButton("1. BUILD MAP")
        self.map_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.map_btn.setCursor(Qt.PointingHandCursor)
        self.map_btn.setStyleSheet(_btn_ss("#1abc9c", hov="#16a085"))
        self.map_btn.clicked.connect(self._on_map)
        pl.addWidget(self.map_btn)

        # ── Phase 2: MISSION button ──────────────────────────────────
        self.mission_btn = QPushButton("2. START MISSION")
        self.mission_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.mission_btn.setCursor(Qt.PointingHandCursor)
        self.mission_btn.setStyleSheet(_btn_ss("#e94560", hov="#c0392b"))
        self.mission_btn.clicked.connect(self._on_mission)
        pl.addWidget(self.mission_btn)

        # ── STOP button ──────────────────────────────────────────────
        self.stop_btn = QPushButton("STOP ALL")
        self.stop_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet(_btn_ss("#e67e22", hov="#d35400"))
        self.stop_btn.clicked.connect(self._on_stop)
        pl.addWidget(self.stop_btn)

        # ── Manual toggle ────────────────────────────────────────────
        self.manual_btn = QPushButton("MANUAL MODE")
        self.manual_btn.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.manual_btn.setCheckable(True)
        self.manual_btn.setStyleSheet(
            "QPushButton{background:#8e44ad;color:white;padding:8px;"
            "border-radius:8px;border:none}"
            "QPushButton:hover{background:#9b59b6}"
            "QPushButton:checked{background:#27ae60;"
            "border:2px solid #2ecc71}")
        self.manual_btn.clicked.connect(self._on_manual)
        pl.addWidget(self.manual_btn)

        # ── WASD teleop ──────────────────────────────────────────────
        tf = QFrame()
        tf.setStyleSheet(
            "QFrame{background:#0d0d2b;border:1px solid #333;"
            "border-radius:8px}")
        tg = QGridLayout(tf)
        tg.setSpacing(3); tg.setContentsMargins(4, 4, 4, 4)

        tl = QLabel("Teleop (W/A/S/D)")
        tl.setFont(QFont("Segoe UI", 8))
        tl.setStyleSheet("color:#7ec8e3;border:none")
        tl.setAlignment(Qt.AlignCenter)
        tg.addWidget(tl, 0, 0, 1, 3)

        _TS = ("QPushButton{background:#1a1a3e;color:#7ec8e3;padding:12px;"
               "border-radius:6px;border:2px solid #2980b9;font-size:14px;"
               "font-weight:bold}"
               "QPushButton:hover{background:#2980b9;color:white}"
               "QPushButton:pressed{background:#1abc9c}")
        _TX = ("QPushButton{background:#c0392b;color:white;padding:12px;"
               "border-radius:6px;border:2px solid #e74c3c;font-size:12px;"
               "font-weight:bold}"
               "QPushButton:hover{background:#e74c3c}")

        bw = QPushButton("W"); bw.setStyleSheet(_TS)
        bw.pressed.connect(lambda: self._teleop(TELEOP_LINEAR, 0))
        bw.released.connect(lambda: self._teleop(0, 0))
        tg.addWidget(bw, 1, 1)

        ba = QPushButton("A"); ba.setStyleSheet(_TS)
        ba.pressed.connect(lambda: self._teleop(0, TELEOP_ANGULAR))
        ba.released.connect(lambda: self._teleop(0, 0))
        tg.addWidget(ba, 2, 0)

        bx = QPushButton("X"); bx.setStyleSheet(_TX)
        bx.clicked.connect(lambda: self._teleop(0, 0))
        tg.addWidget(bx, 2, 1)

        bd = QPushButton("D"); bd.setStyleSheet(_TS)
        bd.pressed.connect(lambda: self._teleop(0, -TELEOP_ANGULAR))
        bd.released.connect(lambda: self._teleop(0, 0))
        tg.addWidget(bd, 2, 2)

        bs = QPushButton("S"); bs.setStyleSheet(_TS)
        bs.pressed.connect(lambda: self._teleop(-TELEOP_LINEAR, 0))
        bs.released.connect(lambda: self._teleop(0, 0))
        tg.addWidget(bs, 3, 1)

        pl.addWidget(tf)

        # ── Log ──────────────────────────────────────────────────────
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 8))
        self.log_area.setStyleSheet(
            "background:#0f3460;color:#e2e2e2;border-radius:6px;"
            "padding:4px;border:1px solid #1a1a2e")
        self.log_area.setMaximumHeight(140)
        pl.addWidget(self.log_area, stretch=1)

        root.addWidget(panel, stretch=2)

        # ── Refresh ──────────────────────────────────────────────────
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
            self.manual_btn.setChecked(False)
            self.ctrl.start_mission()
            self.mission_btn.setText("RUNNING...")
            self.mission_btn.setEnabled(False)
            self.log_area.append("Phase 2: RED -> BLUE -> GREEN")

    def _on_stop(self):
        if self.ctrl.phase == self.ctrl.PHASE_MAPPING:
            self.ctrl.stop_mapping()
        elif self.ctrl.phase == self.ctrl.PHASE_MISSION:
            self.ctrl.stop_mission()
        elif self.ctrl.phase == self.ctrl.PHASE_MANUAL:
            self.ctrl.set_manual_mode(False)
            self.manual_btn.setChecked(False)
        self.map_btn.setText("1. BUILD MAP")
        self.map_btn.setEnabled(True)
        self.mission_btn.setText("2. START MISSION")
        self.mission_btn.setEnabled(True)
        self.log_area.append("STOPPED")

    def _on_manual(self, checked):
        self.ctrl.set_manual_mode(checked)
        if checked:
            self._on_stop()  # stop any running phase first
            self.ctrl.set_manual_mode(True)  # re-enable after stop
            self.manual_btn.setChecked(True)
            self.log_area.append("MANUAL MODE -- use W/A/S/D")
        else:
            self.log_area.append("MANUAL OFF")

    def _teleop(self, lx, az):
        if self.ctrl.phase == self.ctrl.PHASE_MANUAL:
            self.ctrl.manual_cmd(float(lx), float(az))

    # ── Keyboard ─────────────────────────────────────────────────────

    def keyPressEvent(self, e: QKeyEvent):
        if self.ctrl.phase != self.ctrl.PHASE_MANUAL:
            return super().keyPressEvent(e)
        k = e.key()
        if k == Qt.Key_W: self._teleop(TELEOP_LINEAR, 0)
        elif k == Qt.Key_S: self._teleop(-TELEOP_LINEAR, 0)
        elif k == Qt.Key_A: self._teleop(0, TELEOP_ANGULAR)
        elif k == Qt.Key_D: self._teleop(0, -TELEOP_ANGULAR)
        elif k in (Qt.Key_X, Qt.Key_Space): self._teleop(0, 0)
        else: super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QKeyEvent):
        if self.ctrl.phase != self.ctrl.PHASE_MANUAL:
            return super().keyReleaseEvent(e)
        if e.key() in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D):
            self._teleop(0, 0)
        else:
            super().keyReleaseEvent(e)

    # ── GUI refresh (30 fps) ─────────────────────────────────────────

    def _update(self):
        phase = self.ctrl.phase
        fsm = self.ctrl.fsm
        explorer = self.ctrl.explorer
        navigator = self.ctrl.navigator

        # ── Phase label ──────────────────────────────────────────────
        self.phase_label.setText(f"Phase: {phase}")

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
            # Show navigation map with path
            map_img = navigator.get_nav_map_image(ox, oy, oyaw)
        else:
            # Show live LIDAR map
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
            lc = "#e74c3c"
        elif front_d < 0.50:
            lc = "#f39c12"
        else:
            lc = "#2ecc71"
        self.lidar_label.setText(f"LIDAR Front: {front_d:.2f}m")
        self.lidar_label.setStyleSheet(
            f"color:{lc};border:none;padding:4px;font-weight:bold")

        # ── Exploration progress ─────────────────────────────────────
        if phase == self.ctrl.PHASE_MAPPING:
            pct = int(explorer.progress * 100)
            self.explore_progress.setValue(pct)
            self.explore_progress.setFormat(f"Exploring: {pct}%")
        elif explorer.state == Explorer.ST_DONE:
            self.explore_progress.setValue(100)
            self.explore_progress.setFormat("Map saved!")
            # Re-enable buttons
            self.map_btn.setText("1. BUILD MAP")
            self.map_btn.setEnabled(True)

        # ── State display ────────────────────────────────────────────
        if phase == self.ctrl.PHASE_MAPPING:
            st_text = explorer.state.replace("_", " ")
            sc = "#1abc9c"
            status = explorer.status_text
        elif phase == self.ctrl.PHASE_MISSION:
            st_text = fsm.state.replace("_", " ")
            sc = STATE_COLORS.get(fsm.state, "#ecf0f1")
            status = fsm.status_text
        elif phase == self.ctrl.PHASE_MANUAL:
            st_text = "MANUAL"
            sc = "#9b59b6"
            status = "Manual teleop active"
        else:
            st_text = "IDLE"
            sc = "#95a5a6"
            status = "Ready"

        self.state_label.setText(st_text)
        self.state_label.setStyleSheet(
            f"padding:8px;background:#16213e;color:{sc};"
            f"border-radius:8px;font-size:13px;font-weight:bold;"
            f"border:1px solid {sc}")

        self.det_label.setText(
            f"Detected: {fsm.detected_label} | "
            f"Dist: {fsm.detected_distance}")

        # ── Colour indicators ────────────────────────────────────────
        ti = fsm.task_idx
        nt = len(fsm.tasks)
        cp = min(ti, nt)
        for i, c in enumerate(["RED", "BLUE", "GREEN"]):
            hx = COLOUR_HEX[c]
            lb = self.indicators[c]
            if i < cp:
                lb.setStyleSheet(
                    "background:#1a3a1a;color:#2ecc71;"
                    "border:2px solid #2ecc71;border-radius:6px;padding:3px")
            elif i == ti and fsm.state not in (ST_DONE, ST_RETURN_HOME, ST_IDLE):
                lb.setStyleSheet(
                    f"background:#1a1a3e;color:{hx};"
                    f"border:2px solid {hx};border-radius:6px;"
                    "padding:3px;font-weight:bold")
            else:
                lb.setStyleSheet(
                    f"background:#222;color:{hx};"
                    "border:2px solid #333;border-radius:6px;padding:3px")

        # ── Log ──────────────────────────────────────────────────────
        if status != self._last_status:
            self._last_status = status
            self.log_area.append(status)
            sb = self.log_area.verticalScrollBar()
            sb.setValue(sb.maximum())

        # Mission complete
        if fsm.state == ST_DONE and phase == self.ctrl.PHASE_MISSION:
            self.mission_btn.setText("COMPLETE!")
            self.mission_btn.setStyleSheet(_btn_ss("#27ae60"))
            self.mission_btn.setEnabled(False)

    def closeEvent(self, event):
        self.ctrl.motion.stop()
        self.ctrl.fsm.running = False
        self.ctrl.explorer.running = False
        self.ctrl.set_manual_mode(False)
        event.accept()
