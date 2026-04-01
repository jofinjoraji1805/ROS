#!/usr/bin/env python3
"""
nav_pick_place.py — SLAM + Nav2 + YOLO autonomous pick-and-place

Phase 1: SLAM Mapping
  - Launch cartographer, drive with WASD teleop, save map
Phase 2: Nav2 Mission (RED → BLUE → GREEN)
  - Nav2 → pick standoff → YOLO servo → physical grab →
  - Nav2 → drop standoff → YOLO servo → physical release
  - Return home

Usage:
  python3 nav_pick_place.py     # Gazebo sim must already be running
"""

import os
import sys
import math
import time
import threading
import subprocess
import signal as py_signal
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np

# Fix Qt plugin conflict
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)

from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration

import tf2_ros

from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped

from gazebo_msgs.srv import GetEntityState
from linkattacher_msgs.srv import AttachLink, DetachLink
from cv_bridge import CvBridge

try:
    from nav2_msgs.action import NavigateToPose
    HAS_NAV2 = True
except ImportError:
    HAS_NAV2 = False

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QTextEdit, QGroupBox, QGridLayout,
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QTextCursor


# =============================================================================
# Configuration
# =============================================================================
YOLO_MODEL_PATH = "/home/jofin/colcon_ws/src/tb3_pick_place/yolomodel/best.pt"
YOLO_CONF = 0.35

CAMERA_TOPIC = "/pi_camera/image_raw"
CMD_VEL_TOPIC = "/cmd_vel"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

ROBOT_ENTITY = "turtlebot3_manipulation_system"
BASE_ENTITY = f"{ROBOT_ENTITY}::base_footprint"

# Gazebo model names for cubes (used for distance measurement)
CUBE_MODELS = {"RED": "red_cube", "BLUE": "blue_cube", "GREEN": "green_cube"}

# World positions (fallback / ground truth)
CUBE_WORLD_POS = {
    "RED":   (-0.91, -1.50, 0.175),
    "BLUE":  (-0.91,  0.00, 0.175),
    "GREEN": (-0.91,  1.50, 0.175),
}
DROP_ZONE_POS = {
    "RED":   (2.0,  1.50),
    "BLUE":  (2.0, -1.50),
    "GREEN": (2.0,  0.00),
}

# Nav2 standoff waypoints (x, y, yaw) — robot faces the table/zone
# Tables at x=-1.0, cubes at x≈-0.91, approach from +x direction — closer standoff
PICK_STANDOFF = {
    "RED":   (-0.45, -1.50, math.pi),
    "BLUE":  (-0.45,  0.00, math.pi),
    "GREEN": (-0.45,  1.50, math.pi),
}
# Drop zones at x=2.0, approach from -x direction — closer standoff
DROP_STANDOFF = {
    "RED":   (1.60,  1.50, 0.0),
    "BLUE":  (1.60, -1.50, 0.0),
    "GREEN": (1.60,  0.00, 0.0),
}
HOME_POS = (0.50, -2.00, math.pi / 2)

# SLAM / Nav2
MAP_DIR = "/home/jofin/colcon_ws/maps"
MAP_NAME = "pick_place_map"
MAP_YAML = os.path.join(MAP_DIR, MAP_NAME + ".yaml")
TELEOP_LIN_SPEED = 0.15
TELEOP_ANG_SPEED = 0.5

# Visual servo parameters
SCAN_ROTATE_SPEED = 0.12       # rad/s scan rotation
APPROACH_FWD      = 0.18       # m/s approach speed (fast)
MAX_ANG_VEL       = 0.25       # rad/s angular clamp (fast centering)
ALIGN_ROTATE_KP   = 1.0        # proportional gain for pure rotation (increased)
ALIGN_START_TOL   = 0.015      # must be THIS centred before driving (tighter)
ALIGN_STOP_TOL    = 0.035      # stop driving to re-align if error exceeds (tighter)
CLOSE_FWD         = 0.06       # m/s close-range speed
CUBE_LOST_TICKS   = 5          # frames lost → cube under camera
ADJUST_FWD        = 0.03       # m/s final creep
ADJUST_DURATION   = 4.0        # seconds — max 0.08m to fine-tune
ZONE_CLOSE_AREA   = 0.012      # zone fills this much → final approach (smaller = closer)
ZONE_FINAL_TIME   = 8.0        # seconds — long creep to get very close to drop zone
ZONE_FINAL_VEL    = 0.05       # m/s final creep (~0.40m of travel)

# Camera horizontal FOV (approx for Raspberry Pi camera in Gazebo)
CAMERA_HFOV_RAD   = 1.085      # ~62 degrees

# Alignment & approach parameters (strict)
MICRO_CENTER_PIXEL_TOL = 5     # ±5 pixels — tight centering for better alignment
MICRO_CENTER_STEP_RAD  = math.radians(0.5)  # 0.5° per correction tick
APPROACH_STEP_DIST     = 0.03  # 3cm per forward step (with re-verify each)

# PROXIMITY THRESHOLDS — bbox width / image width ratios
# These are the PRIMARY proximity sensor — arm MUST NOT extend unless met
CUBE_FILL_MIN      = 0.06  # GATE: arm must NOT extend if below 6%
CUBE_FILL_TARGET   = 0.15  # TARGET: stop driving when cube fills 15%+ of frame
CUBE_FILL_DANGER   = 0.25  # DANGER: too close, stop immediately
ZONE_FILL_MIN      = 0.25  # GATE: must NOT drop if below 25%
ZONE_FILL_TARGET   = 0.35  # TARGET: stop at 35% — in front of the box, arm reaches in

# Distance thresholds (Gazebo ground truth backup)
CUBE_STOP_DIST     = 0.29  # arm base 29cm from cube — pads reach cube center
ZONE_STOP_DIST     = 0.25  # arm base 25cm from zone — stop IN FRONT, arm drops into box

# Gripper speed control
GRIPPER_CLOSE_STEP  = 0.002    # 2mm increments
GRIPPER_CLOSE_PAUSE = 0.10     # 100ms pause between steps
GRIPPER_SLOW_EFFORT = 5.0      # firm closing effort
GRIPPER_CONTACT_POS = -0.010   # fully closed (22mm gap — physics stops at 25mm cube)

# Arm speed control
ARM_SLOW_DURATION = 1.0        # fast arm movements

# Arm poses [j1, j2, j3, j4] — IK-computed for side-grasp
ARM_HOME      = [0.0, -1.0,   0.30,  0.70]    # tucked, camera clear
ARM_READY     = [0.0, -0.5,   0.00,  0.00]    # above table
ARM_PRE_PICK  = [0.0,  0.0,   0.00,  0.00]    # above cube, L-shape
ARM_PICK      = [0.0,  0.523, 0.410, -0.895]   # Tuned pick pose — gripper at cube height, correct reach
ARM_LIFT      = [0.0, -0.3,   0.00,  0.00]    # lift cube
ARM_CARRY     = [0.0, -1.0,   0.30,  0.70]    # same as HOME for safe travel

# Drop: TOP-DOWN approach — arm goes UP, OVER zone, then DOWN
# Uses 30-degree downward tilt for more vertical release
ARM_DROP_HIGH = [0.0, -0.529,  0.016,  1.037]  # r=0.18 h=0.20 tilt=30° — high above zone
ARM_DROP_OVER = [0.0, -0.494,  0.418,  0.600]  # r=0.20 h=0.15 tilt=30° — over zone center
ARM_DROP_LOW  = [0.0, -0.436,  0.788,  0.173]  # r=0.20 h=0.10 tilt=30° — lowering into zone
ARM_DROP_REL  = [0.0, -0.137,  1.030, -0.369]  # r=0.20 h=0.05 tilt=30° — release height

GRIPPER_OPEN   =  0.019     # 80mm gap — wide open for approach
GRIPPER_GRAB   = -0.009     # 24mm gap — squeezes 25mm cube (friction holds)
GRIPPER_CLOSED = -0.010     # 22mm gap — fully closed
GRIPPER_EFFORT =  5.0       # firm enough to hold, gentle enough not to launch

# OpenMANIPULATOR-X IK geometry
BASE_TO_J2_R = 0.012
BASE_TO_J2_H = 0.0765
J2_J3_X = 0.024;  J2_J3_Z = 0.128
LA = math.sqrt(J2_J3_X**2 + J2_J3_Z**2)
BETA = math.atan2(J2_J3_X, J2_J3_Z)
LB = 0.124
GRIPPER_OFFSET = 0.12
SIDE_GRASP_TILT = 0.0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

BASE_TF_CANDIDATES = ["base_footprint", "base_link"]
ARM_BASE_TF_CANDIDATES = [
    "arm_base_link", "manipulator_base_link",
    "open_manipulator_base_link", "link1", "open_manipulator_link1",
]


# =============================================================================
# Utilities
# =============================================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def norm_angle(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def ik_side(target_r, target_h):
    """Analytical IK for side grasp — returns (q2, q3, q4) or None."""
    alpha = SIDE_GRASP_TILT
    wr = target_r - GRIPPER_OFFSET * math.cos(alpha)
    wh = target_h + GRIPPER_OFFSET * math.sin(alpha)
    dr = wr - BASE_TO_J2_R
    dh = wh - BASE_TO_J2_H
    d = math.sqrt(dr**2 + dh**2)
    if d > LA + LB - 0.001 or d < abs(LA - LB) + 0.001:
        return None
    psi = math.atan2(dh, dr)
    cos_d = clamp((LA**2 + d**2 - LB**2) / (2 * LA * d), -1.0, 1.0)
    delta = math.acos(cos_d)
    for sign in [1, -1]:
        angle_A = psi + sign * delta
        q2 = math.pi / 2.0 - BETA - angle_A
        if q2 < -1.5 or q2 > 1.5:
            continue
        j3r = LA * math.sin(q2 + BETA)
        j3h = LA * math.cos(q2 + BETA)
        q3 = -math.atan2(dh - j3h, dr - j3r) - q2
        q4 = alpha - q2 - q3
        if -1.5 <= q3 <= 1.4 and -1.7 <= q4 <= 1.97:
            return (q2, q3, q4)
    return None


# =============================================================================
# Data types
# =============================================================================
@dataclass
class Detection:
    cls_id: int
    x1: float; y1: float; x2: float; y2: float
    conf: float

@dataclass
class ColorTask:
    label: str
    cube_cls: int
    zone_cls: int
    pick_standoff: Tuple[float, float, float]
    drop_standoff: Tuple[float, float, float]


# =============================================================================
# Signal bridge (thread-safe ROS → Qt)
# =============================================================================
class SignalBridge(QObject):
    log_signal = pyqtSignal(str, str)
    state_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    color_chip_signal = pyqtSignal(str, str)
    mission_done_signal = pyqtSignal()


# =============================================================================
# Perception Module (YOLO)
# =============================================================================
class PerceptionModule:
    def __init__(self, model_path, conf=0.35):
        self.model = YOLO(model_path)
        self.conf = conf
        self.names = self.model.names
        _n2i = {v: k for k, v in self.names.items()}
        self.CLS_RED_CUBE   = _n2i["red_cube"]
        self.CLS_RED_ZONE   = _n2i["red_drop_zone"]
        self.CLS_BLUE_CUBE  = _n2i["blue_cube"]
        self.CLS_BLUE_ZONE  = _n2i["blue_drop_zone"]
        self.CLS_GREEN_CUBE = _n2i["green_cube"]
        self.CLS_GREEN_ZONE = _n2i["green_drop_zone"]
        self.CUBE_IDS = {self.CLS_RED_CUBE, self.CLS_BLUE_CUBE, self.CLS_GREEN_CUBE}
        _cm = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 100, 0)}
        self.draw_colors = {}
        for cid, name in self.names.items():
            for key, bgr in _cm.items():
                if key in name:
                    self.draw_colors[cid] = bgr
                    break

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                dets.append(Detection(
                    int(box.cls[0]), x1, y1, x2, y2, float(box.conf[0])))
        return dets

    def find_target(self, dets, target_cls, img_w, img_h):
        """Find best detection of target_cls. Returns (cx_norm, cy_norm, area_norm) or None."""
        img_area = img_w * img_h
        is_cube = target_cls in self.CUBE_IDS
        best, best_conf = None, 0.0
        for d in dets:
            if d.cls_id != target_cls or d.conf <= best_conf:
                continue
            area = ((d.x2 - d.x1) * (d.y2 - d.y1)) / img_area
            if is_cube and area > 0.15:
                continue
            if not is_cube and area < 0.001:
                continue
            best = d
            best_conf = d.conf
        if best is None:
            return None
        cx = ((best.x1 + best.x2) / 2.0) / img_w
        cy = ((best.y1 + best.y2) / 2.0) / img_h
        area = ((best.x2 - best.x1) * (best.y2 - best.y1)) / img_area
        return cx, cy, area

    def annotate(self, frame, dets, hud=""):
        vis = frame.copy()
        for d in dets:
            c = self.draw_colors.get(d.cls_id, (255, 255, 255))
            cv2.rectangle(vis, (int(d.x1), int(d.y1)),
                          (int(d.x2), int(d.y2)), c, 2)
            lbl = f"{self.names[d.cls_id]} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (int(d.x1), int(d.y1) - th - 6),
                          (int(d.x1) + tw, int(d.y1)), c, -1)
            cv2.putText(vis, lbl, (int(d.x1), int(d.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Crosshair
        cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
        cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (255, 255, 0), 1)
        cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (255, 255, 0), 1)
        if hud:
            cv2.rectangle(vis, (0, 0), (CAMERA_WIDTH, 30), (30, 30, 30), -1)
            cv2.putText(vis, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        return vis

    def build_tasks(self):
        return [
            ColorTask("RED", self.CLS_RED_CUBE, self.CLS_RED_ZONE,
                      PICK_STANDOFF["RED"], DROP_STANDOFF["RED"]),
            ColorTask("BLUE", self.CLS_BLUE_CUBE, self.CLS_BLUE_ZONE,
                      PICK_STANDOFF["BLUE"], DROP_STANDOFF["BLUE"]),
            ColorTask("GREEN", self.CLS_GREEN_CUBE, self.CLS_GREEN_ZONE,
                      PICK_STANDOFF["GREEN"], DROP_STANDOFF["GREEN"]),
        ]


# =============================================================================
# ROS Node — NavPickPlace
# =============================================================================
class NavPickPlaceNode(Node):

    def __init__(self, bridge: SignalBridge):
        super().__init__("nav_pick_place")
        self.sig = bridge
        self.bridge_cv = CvBridge()

        # State
        self._state = "IDLE"
        self._status = "Initializing..."
        self._running = False
        self._paused = False
        self._estop = False
        self._carrying = False
        self._attached_cube = ''
        self._current_task: Optional[ColorTask] = None
        self.completed_colors: list = []
        self.nav2_available = False
        self._last_cube_cx = 0.5   # last known cube center-x (0.5 = perfectly centred)

        # YOLO
        self._log("Loading YOLO model...", "info")
        self.perception = PerceptionModule(YOLO_MODEL_PATH, YOLO_CONF)
        self._log(f"YOLO loaded — classes: {list(self.perception.names.values())}", "ok")
        self.tasks = self.perception.build_tasks()

        # Camera data (thread-safe)
        self._lock = threading.Lock()
        self.latest_frame = None
        self.display_frame = None
        self._detections: List[Detection] = []

        # ROS interfaces
        self._arm_ac = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self._gripper_ac = ActionClient(self, GripperCommand, GRIPPER_ACTION)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(Image, CAMERA_TOPIC, self._img_cb, 10)

        # Initial pose publisher for AMCL
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)

        # Gazebo services
        self.get_state_cli = self.create_client(GetEntityState, "/get_entity_state")
        self.attach_cli = self.create_client(AttachLink, "/ATTACHLINK")
        self.detach_cli = self.create_client(DetachLink, "/DETACHLINK")

        # Nav2 action client
        self._nav2_ac = None
        if HAS_NAV2:
            self._nav2_ac = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.base_tf = None
        self.arm_tf = None

        # Wait for controllers
        self._set_status("Waiting for controllers...")
        self._wait_infra()
        self._resolve_tf_frames()

        # Home the arm at startup — Gazebo spawns with joints at 0
        self._set_status("Homing arm...")
        self._log("Homing arm to carry position...", "info")
        self.move_arm(ARM_HOME, 4.0)
        self._spin_for(1.0)
        gpos = self.get_gripper_world_pos()
        if gpos:
            self._log(f"Arm homed: gripper at z={gpos[2]:.3f}m", "ok")

        self._set_status("Ready — map first, then start mission")
        self._log("All systems ready.", "ok")

    # -- Logging / state -------------------------------------------------------
    def _log(self, msg, level="info"):
        self.get_logger().info(msg)
        self.sig.log_signal.emit(msg, level)

    def _set_state(self, s):
        self._state = s
        self.sig.state_signal.emit(s)

    def _set_status(self, s):
        self._status = s
        self.sig.status_signal.emit(s)

    def _check_abort(self):
        return self._estop or not self._running

    def _wait_if_paused(self):
        while self._paused and self._running and not self._estop:
            self._spin_for(0.2)

    # -- Infrastructure --------------------------------------------------------
    def _spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            try:
                rclpy.spin_once(self, timeout_sec=0.05)
            except (ValueError, RuntimeError):
                time.sleep(0.02)  # fallback if executor busy

    def _wait_infra(self):
        self._log("Waiting for arm + gripper controllers...", "info")
        self._arm_ac.wait_for_server(timeout_sec=60.0)
        self._gripper_ac.wait_for_server(timeout_sec=30.0)
        if self.get_state_cli.wait_for_service(timeout_sec=10.0):
            self._log("Gazebo GetEntityState available.", "info")
        self._spin_for(0.5)
        self._log("Controllers connected.", "ok")

    def _resolve_tf_frames(self):
        self._spin_for(1.0)
        for base in BASE_TF_CANDIDATES:
            for arm in ARM_BASE_TF_CANDIDATES:
                try:
                    t = self.tf_buffer.lookup_transform(
                        base, arm, rclpy.time.Time(),
                        timeout=RclpyDuration(seconds=0.8))
                    tr = t.transform.translation
                    self.base_tf = base
                    self.arm_tf = arm
                    self._log(f"TF: base='{base}' arm='{arm}' "
                              f"offset=[{tr.x:+.3f},{tr.y:+.3f},{tr.z:+.3f}]", "info")
                    return
                except Exception:
                    continue
        self._log("Warning: TF base→arm not resolved", "warn")

    # -- Camera callback -------------------------------------------------------
    def _img_cb(self, msg):
        try:
            frame = self.bridge_cv.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        dets = self.perception.detect(frame)
        hud = f"{self._state}  |  {self._status}"
        vis = self.perception.annotate(frame, dets, hud)
        with self._lock:
            self.latest_frame = frame
            self.display_frame = vis
            self._detections = dets

    def get_display_frame(self):
        with self._lock:
            if self.display_frame is not None:
                return self.display_frame.copy()
        return None

    def _find_target(self, target_cls):
        with self._lock:
            dets = list(self._detections)
            frame = self.latest_frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return self.perception.find_target(dets, target_cls, w, h)

    def _find_target_raw(self, target_cls):
        """Return raw Detection object for target_cls (for bbox measurements)."""
        with self._lock:
            dets = list(self._detections)
        best, best_conf = None, 0.0
        for d in dets:
            if d.cls_id != target_cls or d.conf <= best_conf:
                continue
            best = d
            best_conf = d.conf
        return best

    # -- Motion primitives -----------------------------------------------------
    def publish_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)

    def stop_base(self):
        z = Twist()
        for _ in range(5):
            self.cmd_pub.publish(z)
            self._spin_for(0.02)

    def move_arm(self, positions, duration_sec=3.0):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.velocities = [0.0] * 4
        sec = int(duration_sec)
        pt.time_from_start = Duration(
            sec=sec, nanosec=int((duration_sec - sec) * 1e9))
        traj.points = [pt]
        goal.trajectory = traj
        fut = self._arm_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result()
        if not gh or not gh.accepted:
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=duration_sec + 10.0)
        return True

    def move_arm_multi(self, waypoints, durations):
        """Send multi-point trajectory — waypoints is list of [j1,j2,j3,j4],
        durations is cumulative time for each waypoint."""
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        for pos, dur in zip(waypoints, durations):
            pt = JointTrajectoryPoint()
            pt.positions = [float(p) for p in pos]
            pt.velocities = [0.0] * 4
            sec = int(dur)
            pt.time_from_start = Duration(
                sec=sec, nanosec=int((dur - sec) * 1e9))
            traj.points.append(pt)
        goal.trajectory = traj
        fut = self._arm_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result()
        if not gh or not gh.accepted:
            return False
        rf = gh.get_result_async()
        total = durations[-1] if durations else 5.0
        rclpy.spin_until_future_complete(self, rf, timeout_sec=total + 10.0)
        return True

    def move_gripper(self, position, max_effort=GRIPPER_EFFORT, timeout=5.0):
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        fut = self._gripper_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        gh = fut.result()
        if not gh or not gh.accepted:
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=timeout)
        return True

    # -- Gazebo helpers --------------------------------------------------------
    def get_entity_state(self, name, timeout=1.0):
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = "world"
        fut = self.get_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        res = fut.result()
        if res and res.success:
            return res.state
        return None

    def attach_cube(self, cube_model_name):
        """Attach cube to gripper using Gazebo link attacher plugin."""
        req = AttachLink.Request()
        req.model1_name = ROBOT_ENTITY
        req.link1_name = "gripper_left_link"
        req.model2_name = cube_model_name
        req.link2_name = "link"
        fut = self.attach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        res = fut.result()
        if res and res.success:
            self.get_logger().info(f"ATTACHED {cube_model_name} to gripper")
            return True
        self.get_logger().warn(f"Failed to attach {cube_model_name}")
        return False

    def detach_cube(self, cube_model_name):
        """Detach cube from gripper."""
        req = DetachLink.Request()
        req.model1_name = ROBOT_ENTITY
        req.link1_name = "gripper_left_link"
        req.model2_name = cube_model_name
        req.link2_name = "link"
        fut = self.detach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        res = fut.result()
        if res and res.success:
            self.get_logger().info(f"DETACHED {cube_model_name} from gripper")
            return True
        self.get_logger().warn(f"Failed to detach {cube_model_name}")
        return False

    def get_robot_pose_2d(self):
        """Get robot (x, y, yaw) from Gazebo ground truth or TF fallback."""
        for ent in [ROBOT_ENTITY, BASE_ENTITY]:
            st = self.get_entity_state(ent)
            if st is None:
                continue
            p = st.pose.position
            q = st.pose.orientation
            if abs(p.x) > 20 or abs(p.y) > 20:
                continue
            return (p.x, p.y, yaw_from_quat([q.x, q.y, q.z, q.w]))
        # Fallback: TF
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom", "base_footprint", rclpy.time.Time(),
                timeout=RclpyDuration(seconds=1.0))
            p = tf.transform.translation
            q = tf.transform.rotation
            return (p.x, p.y, yaw_from_quat([q.x, q.y, q.z, q.w]))
        except Exception:
            return None

    def get_odom_pose(self):
        """Get robot pose in odom frame (for AMCL initial pose)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom", "base_footprint", rclpy.time.Time(),
                timeout=RclpyDuration(seconds=1.0))
            p = tf.transform.translation
            q = tf.transform.rotation
            return (p.x, p.y, yaw_from_quat([q.x, q.y, q.z, q.w]))
        except Exception:
            return None

    def get_gripper_world_pos(self):
        """Get gripper (end effector) world position via TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom", "link5", rclpy.time.Time(),
                timeout=RclpyDuration(seconds=1.0))
            p = tf.transform.translation
            return (p.x, p.y, p.z)
        except Exception:
            return None

    # -- Navigation helpers ----------------------------------------------------
    def turn_to_angle(self, target_yaw, tol_deg=2.0, slow=False):
        max_w = 0.15 if slow else 0.3
        for _ in range(200):
            if self._check_abort():
                break
            self._wait_if_paused()
            pose = self.get_robot_pose_2d()
            if pose is None:
                self._spin_for(0.1)
                continue
            err = norm_angle(target_yaw - pose[2])
            if abs(err) < math.radians(tol_deg):
                break
            w = clamp(0.5 * err, -max_w, max_w)
            if 0 < abs(w) < 0.06:
                w = 0.06 * (1 if w > 0 else -1)
            fwd = 0.01 if slow else 0.0  # tiny forward bias when carrying
            self.publish_cmd(fwd, w)
            self._spin_for(0.1)
        self.stop_base()

    def drive_forward(self, dist, speed=0.05, timeout=20.0):
        pose = self.get_robot_pose_2d()
        if pose is None:
            return
        sx, sy = pose[0], pose[1]
        fwd = 1.0 if dist > 0 else -1.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._check_abort():
                break
            self._wait_if_paused()
            pose = self.get_robot_pose_2d()
            if pose is None:
                self._spin_for(0.1)
                continue
            if math.hypot(pose[0] - sx, pose[1] - sy) >= abs(dist) - 0.01:
                break
            self.publish_cmd(fwd * abs(speed), 0.0)
            self._spin_for(0.1)
        self.stop_base()

    def navigate_to_point(self, tx, ty, stop_dist=0.20, speed=0.08):
        """Simple point-to-point navigation (fallback when Nav2 unavailable)."""
        pose = self.get_robot_pose_2d()
        if pose is None:
            return
        slow = self._carrying
        self.turn_to_angle(math.atan2(ty - pose[1], tx - pose[0]), slow=slow)
        pose = self.get_robot_pose_2d()
        if pose is None:
            return
        d = math.hypot(tx - pose[0], ty - pose[1])
        if d - stop_dist > 0.02:
            s = 0.04 if slow else speed
            self.drive_forward(d - stop_dist, speed=s)

    # -- Nav2 navigation -------------------------------------------------------
    def publish_initial_pose(self, x, y, yaw):
        """Publish initial pose for AMCL localization."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)
        cov = [0.0] * 36
        cov[0] = 0.25   # x variance
        cov[7] = 0.25   # y variance
        cov[35] = 0.06  # yaw variance
        msg.pose.covariance = cov
        for _ in range(3):
            self.initial_pose_pub.publish(msg)
            self._spin_for(0.3)
        self._log(f"Published initial pose: ({x:.2f}, {y:.2f}, "
                  f"yaw={math.degrees(yaw):.0f}deg)", "info")

    def navigate_with_nav2(self, tx, ty, goal_yaw=None, timeout=120.0):
        """Nav2 path-planned navigation. Returns True when close enough."""
        if not self.nav2_available or self._nav2_ac is None:
            return False

        self._set_status(f"Nav2: going to ({tx:.2f}, {ty:.2f})...")
        self._log(f"Nav2: sending goal ({tx:.2f}, {ty:.2f})", "info")

        pose = self.get_robot_pose_2d()
        if pose is None:
            return False
        if goal_yaw is None:
            goal_yaw = math.atan2(ty - pose[1], tx - pose[0])
        qz, qw = math.sin(goal_yaw / 2), math.cos(goal_yaw / 2)

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(tx)
        goal.pose.pose.position.y = float(ty)
        goal.pose.pose.orientation.z = float(qz)
        goal.pose.pose.orientation.w = float(qw)

        fut = self._nav2_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result()
        if not gh or not gh.accepted:
            self._log("Nav2: goal rejected!", "warn")
            return False

        self._log("Nav2: goal accepted, navigating...", "info")
        result_future = gh.get_result_async()

        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                try:
                    gh.cancel_goal_async()
                except Exception:
                    pass
                self.stop_base()
                return False
            self._wait_if_paused()
            self._spin_for(0.3)

            if result_future.done():
                self._log("Nav2: goal reached!", "ok")
                return True

            cur = self.get_robot_pose_2d()
            if cur:
                d = math.hypot(tx - cur[0], ty - cur[1])
                self._set_status(f"Nav2: {d:.2f}m remaining")
                if d < 0.30:
                    try:
                        gh.cancel_goal_async()
                    except Exception:
                        pass
                    self._spin_for(0.5)
                    self.stop_base()
                    self._log(f"Nav2: close enough ({d:.2f}m)", "ok")
                    return True

        try:
            gh.cancel_goal_async()
        except Exception:
            pass
        self.stop_base()
        self._log("Nav2: timeout!", "warn")
        return False

    def check_nav2_ready(self):
        if self._nav2_ac is None:
            return False
        return self._nav2_ac.wait_for_server(timeout_sec=2.0)

    # -- Visual servo phases ---------------------------------------------------
    def phase_scan(self, target_cls, timeout=40.0):
        """Rotate slowly to find target. Returns True if found."""
        # Check if already visible
        r = self._find_target(target_cls)
        if r is not None:
            return True

        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return False
            self._wait_if_paused()
            r = self._find_target(target_cls)
            if r is not None:
                self.stop_base()
                return True
            self.publish_cmd(0.0, SCAN_ROTATE_SPEED)
            self._spin_for(0.1)
        self.stop_base()
        return False

    def phase_approach_cube(self, cube_cls, timeout=60.0):
        """Stop-rotate-drive approach to cube. Returns True when at table.
        Tracks last known lateral offset in self._last_cube_cx."""
        driving = False
        lost = 0
        t0 = time.time()
        self._last_cube_cx = 0.5  # reset

        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return False
            self._wait_if_paused()
            r = self._find_target(cube_cls)

            if r is None:
                lost += 1
                if lost > CUBE_LOST_TICKS:
                    self.stop_base()
                    return True  # cube under camera → at table
                if driving:
                    self.publish_cmd(CLOSE_FWD, 0.0)
                self._spin_for(0.1)
                continue
            lost = 0
            cx, cy, area = r
            self._last_cube_cx = cx  # track last known position
            err = cx - 0.5

            # Close enough → switch to precision close-range drive
            # area=0.012 ≈ cube at ~0.35m — start careful centering drive
            if area >= 0.012:
                self.stop_base()
                driving = False
                self._log(f"Close range — precision centering (area={area:.4f})", "info")
                inner_t0 = time.time()
                inner_lost = 0
                while time.time() - inner_t0 < 30.0 and rclpy.ok():
                    if self._check_abort():
                        self.stop_base()
                        return False
                    ri = self._find_target(cube_cls)
                    if ri is None:
                        inner_lost += 1
                        if inner_lost > CUBE_LOST_TICKS:
                            self.stop_base()
                            return True
                        self.publish_cmd(CLOSE_FWD, 0.0)
                        self._spin_for(0.1)
                        continue
                    inner_lost = 0
                    self._last_cube_cx = ri[0]
                    ie = ri[0] - 0.5
                    # Strict centering at close range
                    if abs(ie) > 0.025:
                        ang = clamp(-ALIGN_ROTATE_KP * ie, -MAX_ANG_VEL, MAX_ANG_VEL)
                        self.publish_cmd(0.0, ang)
                    else:
                        self.publish_cmd(CLOSE_FWD, 0.0)
                    self._spin_for(0.1)
                self.stop_base()
                return True

            # Stop-rotate-drive pattern
            if driving:
                if abs(err) > ALIGN_STOP_TOL:
                    self.stop_base()
                    driving = False
                else:
                    self.publish_cmd(APPROACH_FWD, 0.0)
            else:
                if abs(err) <= ALIGN_START_TOL:
                    driving = True
                    self.publish_cmd(APPROACH_FWD, 0.0)
                else:
                    ang = clamp(-ALIGN_ROTATE_KP * err, -MAX_ANG_VEL, MAX_ANG_VEL)
                    self.publish_cmd(0.0, ang)
            self._spin_for(0.1)

        self.stop_base()
        return False

    def phase_micro_center(self, target_cls, pixel_tol=None, timeout=8.0):
        """Proportional centering — rotate in place until target is centered.
        Uses proportional speed: fast when far off, slow when close."""
        if pixel_tol is None:
            pixel_tol = MICRO_CENTER_PIXEL_TOL
        self.stop_base()
        self._spin_for(0.3)

        t0 = time.time()
        stable_count = 0
        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return
            r = self._find_target(target_cls)
            if r is None:
                self._spin_for(0.1)
                continue
            cx, cy, area = r
            self._last_cube_cx = cx
            pixel_err = (cx - 0.5) * CAMERA_WIDTH

            if abs(pixel_err) <= pixel_tol:
                stable_count += 1
                if stable_count >= 3:
                    self.stop_base()
                    self._log(f"Micro-center: aligned ({pixel_err:+.1f}px, ±{pixel_tol}px)", "ok")
                    return
                self.publish_cmd(0.0, 0.0)
            else:
                stable_count = 0
                # Proportional rotation — fast when far off, slow when close
                err_norm = (cx - 0.5)  # -0.5 to +0.5
                ang = clamp(-ALIGN_ROTATE_KP * err_norm, -MAX_ANG_VEL, MAX_ANG_VEL)
                # Min speed to overcome friction
                if 0 < abs(ang) < 0.04:
                    ang = 0.04 * (1 if ang > 0 else -1)
                self.publish_cmd(0.0, ang)
            self._spin_for(0.1)
        self.stop_base()
        self._log("Micro-center: timeout (proceeding)", "warn")

    def _center_on_target(self, target_cls, pixel_tol=8, timeout=10.0):
        """Pure rotation centering — NEVER drives forward.
        Accounts for rotational inertia: stops early, waits, re-checks.
        Returns True if centered within pixel_tol after full stop."""
        self.stop_base()
        self._spin_for(0.3)
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return False
            r = self._find_target(target_cls)
            if r is None:
                self._spin_for(0.1)
                continue
            cx, cy, area = r
            self._last_cube_cx = cx
            pixel_err = (cx - 0.5) * CAMERA_WIDTH

            if abs(pixel_err) <= pixel_tol:
                # Stop and wait for inertia to settle
                self.stop_base()
                self._spin_for(0.5)
                # Re-check after settling
                r2 = self._find_target(target_cls)
                if r2 is not None:
                    settled_err = (r2[0] - 0.5) * CAMERA_WIDTH
                    self._last_cube_cx = r2[0]
                    if abs(settled_err) <= pixel_tol:
                        self._log(f"Centered: err={settled_err:+.1f}px (±{pixel_tol}px)", "ok")
                        return True
                    # Overshot due to inertia — correct again
                    pixel_err = settled_err
                else:
                    return True  # lost = under camera = close enough

            # Proportional rotation — very gentle
            ang_speed = clamp(-0.5 * (pixel_err / CAMERA_WIDTH * 2), -0.08, 0.08)
            # Min speed to overcome friction
            if 0 < abs(ang_speed) < 0.025:
                ang_speed = 0.025 * (1 if ang_speed > 0 else -1)
            self.publish_cmd(0.0, ang_speed)  # ROTATION ONLY
            self._spin_for(0.15)
        self.stop_base()
        self._spin_for(0.3)
        # Check final position even on timeout
        r = self._find_target(target_cls)
        if r is not None:
            self._last_cube_cx = r[0]
            final_err = (r[0] - 0.5) * CAMERA_WIDTH
            self._log(f"Center timeout: err={final_err:+.1f}px", "warn")
            return abs(final_err) <= pixel_tol
        return False

    def _get_ground_truth_dist(self, target_model):
        """Get distance from arm base to target model via Gazebo."""
        if not target_model:
            return None
        st = self.get_entity_state(target_model)
        pose = self.get_robot_pose_2d()
        if st and pose:
            yaw = pose[2]
            arm_x = pose[0] - 0.092 * math.cos(yaw)
            arm_y = pose[1] - 0.092 * math.sin(yaw)
            return math.hypot(st.pose.position.x - arm_x,
                              st.pose.position.y - arm_y)
        return None

    def phase_verified_approach(self, cube_cls, target_model=None,
                                 target_pos=None, timeout=90.0):
        """Two-phase approach:
        Phase 1 (far): Proportional steer+drive until dist < 0.55m
        Phase 2 (close): Strict center-then-drive-straight until at table.
        Returns (fill_ratio, cube_px_width)."""
        self.stop_base()
        self._spin_for(0.3)
        cube_px_width = 0
        fill_ratio = 0.0
        t0 = time.time()
        tick = 0

        # ═══ PHASE 1: PROPORTIONAL APPROACH (far range) ═══
        # Drive + steer simultaneously — fast and reliable for long distances
        self._log("Phase 1: proportional approach", "info")
        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return fill_ratio, cube_px_width

            dist = self._get_ground_truth_dist(target_model)
            if dist is not None and dist <= 0.55:
                self.stop_base()
                self._log(f"Phase 1 done: dist={dist:.3f}m — switching to precision", "ok")
                break
            if dist is not None and dist <= CUBE_STOP_DIST:
                self.stop_base()
                self._log(f"Already close: dist={dist:.3f}m", "ok")
                break

            r = self._find_target(cube_cls)
            det = self._find_target_raw(cube_cls)
            if r is not None and det is not None:
                cx, cy, area = r
                self._last_cube_cx = cx
                cube_px_width = det.x2 - det.x1
                fill_ratio = cube_px_width / CAMERA_WIDTH
                err_norm = cx - 0.5
                ang = clamp(-ALIGN_ROTATE_KP * err_norm, -MAX_ANG_VEL, MAX_ANG_VEL)
                center_q = 1.0 - min(1.0, abs(err_norm) * 2.0 / 0.5)
                fwd = APPROACH_FWD * max(0.3, center_q)
                self.publish_cmd(fwd, ang)
            else:
                # Cube lost — drive straight slowly
                self.publish_cmd(APPROACH_FWD * 0.5, 0.0)

            tick += 1
            if tick % 10 == 0 and dist:
                self._log(f"Phase1: dist={dist:.3f}m fill={fill_ratio:.0%}", "info")
            self._spin_for(0.1)

        self.stop_base()
        self._spin_for(0.3)

        # ═══ PHASE 2: STRICT CENTER-THEN-DRIVE (close range) ═══
        # Center first, then drive straight in 2cm steps
        self._log("Phase 2: strict center-then-drive", "info")
        step_count = 0
        t1 = time.time()

        while time.time() - t1 < 40.0 and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return fill_ratio, cube_px_width

            # Check stop condition
            dist = self._get_ground_truth_dist(target_model)
            if dist is not None and dist <= CUBE_STOP_DIST:
                self.stop_base()
                self._log(f"At table! dist={dist:.3f}m", "ok")
                break

            # CENTER (rotate only)
            self._center_on_target(cube_cls, pixel_tol=5, timeout=5.0)

            # Re-check distance after centering
            dist = self._get_ground_truth_dist(target_model)
            if dist is not None and dist <= CUBE_STOP_DIST:
                self.stop_base()
                self._log(f"At table! dist={dist:.3f}m", "ok")
                break

            # DRIVE STRAIGHT 2cm
            drive_time = 0.02 / APPROACH_FWD
            self.publish_cmd(APPROACH_FWD, 0.0)
            self._spin_for(drive_time)
            self.stop_base()
            self._spin_for(0.2)

            step_count += 1
            if step_count % 3 == 0:
                d = self._get_ground_truth_dist(target_model)
                self._log(f"Step {step_count}: dist={d:.3f}m" if d else
                          f"Step {step_count}", "info")

        self.stop_base()
        self._spin_for(0.3)

        # Final center correction
        self._center_on_target(cube_cls, pixel_tol=5, timeout=4.0)

        det = self._find_target_raw(cube_cls)
        if det:
            cube_px_width = det.x2 - det.x1
            fill_ratio = cube_px_width / CAMERA_WIDTH
        self._log(f"Approach done: fill={fill_ratio:.0%}, cube_width={cube_px_width:.0f}px", "ok")
        return fill_ratio, cube_px_width

    def phase_verified_zone_approach(self, zone_cls, timeout=90.0):
        """Drive toward drop zone continuously with proportional steering.
        Returns fill_ratio."""
        self.stop_base()
        self._spin_for(0.3)
        fill_ratio = 0.0
        lost_count = 0

        t0 = time.time()
        tick = 0
        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return fill_ratio

            r = self._find_target(zone_cls)
            det = self._find_target_raw(zone_cls)
            if r is None or det is None:
                lost_count += 1
                if lost_count > 20:
                    self.stop_base()
                    break
                self.publish_cmd(0.03, 0.0)
                self._spin_for(0.1)
                continue

            lost_count = 0
            cx, cy, area = r
            pixel_err = (cx - 0.5) * CAMERA_WIDTH
            zone_px_width = det.x2 - det.x1
            fill_ratio = zone_px_width / CAMERA_WIDTH

            if tick % 5 == 0:
                self._log(f"Zone: fill={fill_ratio:.0%} err={pixel_err:+.1f}px", "info")
            tick += 1

            if fill_ratio >= ZONE_FILL_TARGET:
                self.stop_base()
                self._log(f"CLOSE to zone! fill={fill_ratio:.0%}", "ok")
                break

            # Proportional steering + forward
            err_norm = cx - 0.5
            ang = clamp(-ALIGN_ROTATE_KP * err_norm, -MAX_ANG_VEL, MAX_ANG_VEL)
            center_q = 1.0 - min(1.0, abs(pixel_err) / 80.0)
            fwd = APPROACH_FWD * max(0.2, center_q)
            self.publish_cmd(fwd, ang)
            self._spin_for(0.1)

        self.stop_base()
        self._spin_for(0.3)

        # No blind approach — already close enough from visual servo
        self.stop_base()
        self._spin_for(0.3)

        det = self._find_target_raw(zone_cls)
        if det:
            fill_ratio = (det.x2 - det.x1) / CAMERA_WIDTH
        self._log(f"Zone approach done: fill={fill_ratio:.0%}", "ok")
        return fill_ratio

    def phase_drive_to_target(self, target_model=None, target_pos=None,
                               cube_cls=None, stop_dist=0.22,
                               speed=ADJUST_FWD, timeout=30.0):
        """Drive forward until arm base is within stop_dist of the target.
        Uses Gazebo ground truth for distance, YOLO for centering.
        stop_dist=0.22 means arm reaches ~0.20m and stops ~0.02m short of cube.
        Falls back to time-based if ground truth is unavailable."""
        # Try to get target position from Gazebo
        target_xy = None
        if target_model:
            st = self.get_entity_state(target_model)
            if st:
                target_xy = (st.pose.position.x, st.pose.position.y)
                self._log(f"Target '{target_model}' at ({target_xy[0]:.2f}, {target_xy[1]:.2f})", "info")
        if target_xy is None and target_pos:
            target_xy = (target_pos[0], target_pos[1])
            self._log(f"Using fixed target position ({target_xy[0]:.2f}, {target_xy[1]:.2f})", "info")

        if target_xy is None:
            # Fallback: time-based drive
            self._log("No target position — using timed approach (6s)", "warn")
            t0 = time.time()
            while time.time() - t0 < 6.0 and rclpy.ok():
                if self._check_abort():
                    break
                self.publish_cmd(speed, 0.0)
                self._spin_for(0.1)
            self.stop_base()
            self._spin_for(0.5)
            return

        # Distance-based approach: drive until arm base is stop_dist from target
        # Arm base (link1) is 0.092m behind base_footprint
        ARM_BASE_OFFSET = 0.092
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                break
            pose = self.get_robot_pose_2d()
            if pose is None:
                self._spin_for(0.1)
                continue
            # Arm base position
            yaw = pose[2]
            arm_x = pose[0] - ARM_BASE_OFFSET * math.cos(yaw)
            arm_y = pose[1] - ARM_BASE_OFFSET * math.sin(yaw)
            dist = math.hypot(target_xy[0] - arm_x, target_xy[1] - arm_y)

            if dist <= stop_dist:
                self.stop_base()
                self._log(f"At target! Arm base {dist:.3f}m from target (stop={stop_dist:.2f}m)", "ok")
                break

            # Maintain centering if YOLO target visible
            ang = 0.0
            if cube_cls is not None:
                r = self._find_target(cube_cls)
                if r is not None:
                    self._last_cube_cx = r[0]
                    err = r[0] - 0.5
                    if abs(err) > 0.02:
                        ang = clamp(-0.6 * err, -0.08, 0.08)

            # Slow down as we get close
            v = speed if dist > stop_dist + 0.10 else speed * 0.6
            self.publish_cmd(v, ang)
            self._spin_for(0.1)

        self.stop_base()
        self._spin_for(0.5)
        self._log(f"Drive complete. Last cube cx={self._last_cube_cx:.3f}", "ok")

    def phase_approach_zone(self, zone_cls, timeout=60.0):
        """Stop-rotate-drive approach to drop zone. Returns True when at zone."""
        driving = False
        lost = 0
        in_final = False
        final_start = 0.0
        t0 = time.time()

        while time.time() - t0 < timeout and rclpy.ok():
            if self._check_abort():
                self.stop_base()
                return False
            self._wait_if_paused()

            if in_final:
                fe = time.time() - final_start
                if fe >= ZONE_FINAL_TIME:
                    self.stop_base()
                    return True
                r = self._find_target(zone_cls)
                if r:
                    err_z = r[0] - 0.5
                    if abs(err_z) > ALIGN_STOP_TOL:
                        ang = clamp(-ALIGN_ROTATE_KP * err_z,
                                    -MAX_ANG_VEL, MAX_ANG_VEL)
                        self.publish_cmd(0.0, ang)
                    else:
                        self.publish_cmd(ZONE_FINAL_VEL, 0.0)
                else:
                    self.publish_cmd(ZONE_FINAL_VEL, 0.0)
                self._spin_for(0.1)
                continue

            r = self._find_target(zone_cls)
            if r is None:
                lost += 1
                if lost > 20:
                    self.stop_base()
                    return False
                self._spin_for(0.1)
                continue
            lost = 0
            cx, cy, area = r
            err = cx - 0.5

            if area >= ZONE_CLOSE_AREA:
                self.stop_base()
                in_final = True
                final_start = time.time()
                driving = False
                continue

            if driving:
                if abs(err) > ALIGN_STOP_TOL:
                    self.stop_base()
                    driving = False
                else:
                    self.publish_cmd(APPROACH_FWD, 0.0)
            else:
                if abs(err) <= ALIGN_START_TOL:
                    driving = True
                    self.publish_cmd(APPROACH_FWD, 0.0)
                else:
                    ang = clamp(-ALIGN_ROTATE_KP * err,
                                -MAX_ANG_VEL, MAX_ANG_VEL)
                    self.publish_cmd(0.0, ang)
            self._spin_for(0.1)

        self.stop_base()
        return False

    # -- Pick sequence (REAL physical grasp — horizontal reach, then lower, grip)
    def phase_pick(self, task: ColorTask):
        self._set_state("PICKING")
        lbl = task.label
        cube_model = CUBE_MODELS.get(lbl, "")

        # ── STEP 0: STOP base + get ground truth ──
        self.stop_base()
        self._spin_for(0.5)

        # Get cube position from Gazebo ground truth
        cube_st = self.get_entity_state(cube_model) if cube_model else None
        cube_z = cube_st.pose.position.z if cube_st else 0.175
        cube_x = cube_st.pose.position.x if cube_st else -0.91
        cube_y = cube_st.pose.position.y if cube_st else -1.50

        # ── STEP 0b: CHECK ALIGNMENT — back up and re-center if not perpendicular ──
        self._set_status(f"[{lbl}] Checking alignment...")
        pose = self.get_robot_pose_2d()
        if pose and cube_st:
            rx, ry, ryaw = pose
            dx = cube_x - rx
            dy = cube_y - ry
            target_yaw = math.atan2(dy, dx)
            heading_err = norm_angle(target_yaw - ryaw)
            lateral = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
            fwd_dist = dx * math.cos(ryaw) + dy * math.sin(ryaw)

            self._log(f"[{lbl}] Alignment: heading_err={math.degrees(heading_err):.1f}° "
                      f"lateral={lateral:+.3f}m fwd={fwd_dist:.3f}m", "info")

            # If lateral offset > 2cm or heading > 3°, back up and re-align
            if abs(lateral) > 0.02 or abs(heading_err) > math.radians(3.0):
                self._log(f"[{lbl}] NOT centered — backing up to re-align", "warn")
                # Back up 15cm
                self.drive_forward(-0.15, speed=0.04, timeout=8.0)
                self._spin_for(0.3)
                # Turn to face cube exactly
                pose = self.get_robot_pose_2d()
                if pose:
                    rx, ry, ryaw = pose
                    target_yaw = math.atan2(cube_y - ry, cube_x - rx)
                self.turn_to_angle(target_yaw, tol_deg=0.5, slow=True)
                self._spin_for(0.3)
                # Drive forward to correct distance
                pose = self.get_robot_pose_2d()
                if pose:
                    rx, ry, ryaw = pose
                    arm_x = rx - 0.092 * math.cos(ryaw)
                    arm_y = ry - 0.092 * math.sin(ryaw)
                    cur_dist = math.hypot(cube_x - arm_x, cube_y - arm_y)
                    drive_dist = cur_dist - 0.33  # target 33cm from cube
                    if drive_dist > 0.02:
                        self.drive_forward(min(drive_dist, 0.20), speed=0.04, timeout=10.0)
                        self._spin_for(0.3)
                # Final heading correction
                pose = self.get_robot_pose_2d()
                if pose:
                    rx, ry, ryaw = pose
                    target_yaw = math.atan2(cube_y - ry, cube_x - rx)
                    if abs(norm_angle(target_yaw - ryaw)) > math.radians(1.0):
                        self.turn_to_angle(target_yaw, tol_deg=0.5, slow=True)
                        self._spin_for(0.3)
                # Re-check alignment
                pose = self.get_robot_pose_2d()
                if pose:
                    rx, ry, ryaw = pose
                    dx = cube_x - rx
                    dy = cube_y - ry
                    lateral = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
                    heading_err = norm_angle(math.atan2(dy, dx) - ryaw)
                    self._log(f"[{lbl}] Re-aligned: heading_err={math.degrees(heading_err):.1f}° "
                              f"lateral={lateral:+.3f}m", "ok")
            elif abs(heading_err) > math.radians(1.0):
                # Small heading correction only
                self.turn_to_angle(target_yaw, tol_deg=0.5, slow=True)
                self._spin_for(0.3)
                pose2 = self.get_robot_pose_2d()
                if pose2:
                    err2 = norm_angle(target_yaw - pose2[2])
                    self._log(f"[{lbl}] After align: err={math.degrees(err2):.1f}°", "ok")

        # ── STEP 1: Compute J1 lateral correction (small residual) ──
        j1_offset = 0.0
        pose = self.get_robot_pose_2d()
        if pose and cube_st:
            rx, ry, ryaw = pose
            dx = cube_x - rx
            dy = cube_y - ry
            lateral = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
            if abs(lateral) > 0.005:
                fwd_dist = dx * math.cos(ryaw) + dy * math.sin(ryaw)
                if fwd_dist > 0.1:
                    j1_offset = math.atan2(lateral, fwd_dist)
                    j1_offset = max(-0.15, min(0.15, j1_offset))
                    self._log(f"[{lbl}] Residual lateral={lateral:+.3f}m → J1={math.degrees(j1_offset):+.1f}°", "info")

        # ── STEP 2: Open gripper ──
        self._set_status(f"[{lbl}] Opening gripper...")
        self.move_gripper(GRIPPER_OPEN, max_effort=5.0, timeout=2.0)

        # ── STEP 3: Extend arm to PICK pose (tuned joints) ──
        # ARM_PICK = [0, 0.523, 0.410, -0.895] — manually tuned for correct reach + height
        self._set_status(f"[{lbl}] Extending arm to pick pose...")
        pick_joints = [j1_offset, ARM_PICK[1], ARM_PICK[2], ARM_PICK[3]]

        # Smooth transition: intermediate poses
        self.move_arm([j1_offset, -0.3, 0.10, 0.20], 2.5)
        self._spin_for(0.3)
        # Halfway
        half = [j1_offset,
                ARM_PICK[1] * 0.5, ARM_PICK[2] * 0.5, ARM_PICK[3] * 0.5]
        self.move_arm(half, 2.5)
        self._spin_for(0.3)
        # Full pick pose
        self.move_arm(pick_joints, 2.5)
        self._spin_for(0.5)

        # Verify position
        pad_offset = 0.097
        gpos = self.get_gripper_world_pos()
        if gpos:
            gx, gy, gz = gpos
            g2c = math.hypot(cube_x - gx, cube_y - gy)
            self._log(f"[{lbl}] PICK POSE: link5=({gx:.3f},{gy:.3f},{gz:.3f}) "
                      f"g2c={g2c:.3f}m pad={g2c-pad_offset:+.3f}m "
                      f"gz-cz={gz-cube_z:+.3f}m", "info")

        # ── STEP 6: Close gripper around cube ──
        self._set_status(f"[{lbl}] Gripping cube...")
        # Ensure fully open first
        self.move_gripper(GRIPPER_OPEN, max_effort=5.0, timeout=2.0)
        self._spin_for(0.3)
        # Close in steps: wide → near cube → grab
        self.move_gripper(0.005, max_effort=3.0, timeout=2.0)   # 52mm gap
        self._spin_for(0.2)
        self.move_gripper(-0.002, max_effort=3.0, timeout=2.0)  # 38mm gap
        self._spin_for(0.2)
        self.move_gripper(-0.005, max_effort=4.0, timeout=2.0)  # 32mm gap — near cube
        self._spin_for(0.2)
        self.move_gripper(-0.007, max_effort=4.0, timeout=2.0)  # 28mm gap — touching
        self._spin_for(0.3)
        self.move_gripper(GRIPPER_GRAB, max_effort=GRIPPER_EFFORT, timeout=2.0)  # 24mm — squeeze
        self._spin_for(0.5)

        self._carrying = True
        self._log(f"[{lbl}] Gripper closed — carrying", "ok")

        # ── STEP 7: Lift + Tuck ──
        self._set_status(f"[{lbl}] Lifting...")
        self.stop_base()

        # Lift by interpolating from pick pose back up
        p = ARM_PICK
        for frac in [0.7, 0.4, 0.0]:
            self.move_arm([j1_offset, p[1]*frac, p[2]*frac, p[3]*frac], 2.5)
            self._spin_for(0.3)

        # Tuck to carry
        self._set_status(f"[{lbl}] Tucking arm...")
        self.move_arm([j1_offset, -0.3, 0.10, 0.20], 2.5)
        self._spin_for(0.2)
        self.move_arm([j1_offset, -0.5, 0.15, 0.35], 2.5)
        self._spin_for(0.2)
        self.move_arm(ARM_CARRY, 2.5)
        self._spin_for(0.3)

        if cube_model and cube_st:
            cl = self.get_entity_state(cube_model)
            if cl:
                self._log(f"[{lbl}] Cube z: {cl.pose.position.z:.3f}m (was {cube_st.pose.position.z:.3f}m)", "info")

        self._log(f"[{lbl}] Pick complete!", "ok")

    # -- Place sequence (TOP-DOWN drop — arm goes up, over, then down) ---------
    def phase_place(self, task: ColorTask):
        self._set_state("PLACING")
        lbl = task.label

        self.stop_base()
        self._spin_for(0.2)

        # STEP 1: Raise arm HIGH
        self._set_status(f"[{lbl}] Raising arm...")
        self.move_arm(ARM_DROP_HIGH, 1.5)
        self._spin_for(0.2)

        # STEP 2: Extend OVER zone
        self._set_status(f"[{lbl}] Extending over zone...")
        self.move_arm(ARM_DROP_OVER, 1.5)
        self._spin_for(0.2)

        # STEP 3: Lower into zone
        self._set_status(f"[{lbl}] Lowering...")
        self.move_arm(ARM_DROP_LOW, 1.5)
        self._spin_for(0.2)

        # STEP 4: Final descent
        self._set_status(f"[{lbl}] Release height...")
        self.move_arm(ARM_DROP_REL, 1.5)
        self._spin_for(0.2)

        # STEP 5: Open gripper to release (actual grip — no detach needed)
        self._set_status(f"[{lbl}] Releasing cube...")
        self.move_gripper(GRIPPER_OPEN, max_effort=5.0, timeout=2.0)
        self._carrying = False
        self._spin_for(0.3)

        # STEP 6: Retract arm
        self._set_status(f"[{lbl}] Retracting...")
        self.move_arm(ARM_DROP_HIGH, 1.2)
        self._spin_for(0.2)
        self.move_arm(ARM_HOME, 1.5)
        self._spin_for(0.2)

        self._log(f"[{lbl}] Cube placed at {lbl} drop zone!", "ok")

    # =========================================================================
    # Main mission loop
    # =========================================================================
    def run(self):
        self._running = True
        self._paused = False
        self._estop = False
        self.completed_colors = []

        self._log("=== MISSION START ===", "ok")
        self._set_status("Preparing arm...")
        self.move_arm(ARM_HOME, 2.0)
        self.move_gripper(GRIPPER_OPEN, max_effort=5.0, timeout=2.0)
        self._spin_for(1.0)

        # Publish initial pose for AMCL
        odom_pose = self.get_odom_pose()
        if odom_pose:
            self.publish_initial_pose(odom_pose[0], odom_pose[1], odom_pose[2])
        else:
            pose = self.get_robot_pose_2d()
            if pose:
                self.publish_initial_pose(pose[0], pose[1], pose[2])
        self._spin_for(2.0)

        for task in self.tasks:
            if self._check_abort():
                break
            self._current_task = task
            self.sig.color_chip_signal.emit(task.label, "active")
            lbl = task.label
            self._log(f"--- [{lbl}] Starting pickup ---", "info")

            # ---- 1. Navigate to pick standoff ----
            self._set_state("NAV TO PICK")
            sx, sy, syaw = task.pick_standoff
            if self.nav2_available:
                self._set_status(f"[{lbl}] Nav2 → pick table...")
                if not self.navigate_with_nav2(sx, sy, goal_yaw=syaw):
                    self._log(f"[{lbl}] Nav2 failed, using direct nav", "warn")
                    self.navigate_to_point(sx, sy, stop_dist=0.15)
            else:
                self._set_status(f"[{lbl}] Driving to pick table...")
                self.navigate_to_point(sx, sy, stop_dist=0.15)
            self.turn_to_angle(syaw)
            if self._check_abort():
                break

            # ---- 2. YOLO visual servo to cube ----
            self._set_state("SERVO TO CUBE")
            self._set_status(f"[{lbl}] Scanning for cube...")
            found = self.phase_scan(task.cube_cls, timeout=30.0)
            if not found:
                self._log(f"[{lbl}] Cube not found, moving closer...", "warn")
                self.drive_forward(0.15, speed=0.05)
                found = self.phase_scan(task.cube_cls, timeout=20.0)

            if found:
                # Step A: Strict center-then-drive approach
                self._set_status(f"[{lbl}] Center-first approach...")
                fill, cpw = self.phase_verified_approach(
                    cube_cls=task.cube_cls,
                    target_model=CUBE_MODELS[lbl],
                    target_pos=CUBE_WORLD_POS[lbl])

                # Step B: GROUND TRUTH heading alignment before pick
                self._set_status(f"[{lbl}] GT heading alignment...")
                self.stop_base()
                self._spin_for(0.5)
                cube_st = self.get_entity_state(CUBE_MODELS[lbl])
                pose = self.get_robot_pose_2d()
                if cube_st and pose:
                    rx, ry, ryaw = pose
                    cx, cy = cube_st.pose.position.x, cube_st.pose.position.y
                    target_yaw = math.atan2(cy - ry, cx - rx)
                    heading_err = abs(norm_angle(target_yaw - ryaw))
                    self._log(f"[{lbl}] Pre-pick heading err: {math.degrees(heading_err):.1f}°", "info")
                    if heading_err > math.radians(1.5):
                        self.turn_to_angle(target_yaw, tol_deg=1.0, slow=True)
                        self._log(f"[{lbl}] Heading corrected", "ok")
                else:
                    self._log(f"[{lbl}] No GT — using YOLO alignment", "info")
                    r = self._find_target(task.cube_cls)
                    if r is not None:
                        pixel_err = (r[0] - 0.5) * CAMERA_WIDTH
                        if abs(pixel_err) > 5:
                            self._center_on_target(task.cube_cls, pixel_tol=5, timeout=6.0)
            else:
                self._log(f"[{lbl}] Using ground truth fallback — driving very close", "warn")
                cp = CUBE_WORLD_POS[lbl]
                self.navigate_to_point(cp[0] + 0.12, cp[1], stop_dist=0.05)
                self.turn_to_angle(math.pi)
            if self._check_abort():
                break

            # ---- 3. Pick cube ----
            self.phase_pick(task)
            if self._check_abort():
                break

            # ---- 4. Reverse from table ----
            self._set_state("REVERSING")
            self._set_status(f"[{lbl}] Backing up from table...")
            self.drive_forward(-0.25, speed=0.04)
            self._spin_for(0.5)
            if self._check_abort():
                break

            # ---- 5. Navigate to drop standoff ----
            self._set_state("NAV TO DROP")
            dx, dy, dyaw = task.drop_standoff
            if self.nav2_available:
                self._set_status(f"[{lbl}] Nav2 → drop zone...")
                if not self.navigate_with_nav2(dx, dy, goal_yaw=dyaw):
                    self._log(f"[{lbl}] Nav2 failed, using direct nav", "warn")
                    self.navigate_to_point(dx, dy, stop_dist=0.15)
            else:
                self._set_status(f"[{lbl}] Driving to drop zone...")
                self.navigate_to_point(dx, dy, stop_dist=0.15)
            self.turn_to_angle(dyaw, slow=self._carrying)
            if self._check_abort():
                break

            # ---- 6. YOLO visual servo to drop zone ----
            self._set_state("SERVO TO ZONE")
            self._set_status(f"[{lbl}] Scanning for drop zone...")
            found = self.phase_scan(task.zone_cls, timeout=30.0)
            if found:
                # Step A: Initial alignment to zone
                self._set_status(f"[{lbl}] Aligning to drop zone (±3px)...")
                self.phase_micro_center(task.zone_cls)

                # Step B: Drive VERY close — zone bbox > 50% of frame
                self._set_status(f"[{lbl}] Driving CLOSE to zone (bbox>{ZONE_FILL_TARGET:.0%})...")
                zone_fill = self.phase_verified_zone_approach(task.zone_cls)

                # Step C: Final centering on zone
                self._set_status(f"[{lbl}] Centering on drop zone (±3px)...")
                self.phase_micro_center(task.zone_cls)

                # Step D: GATE CHECK — keep driving until truly close
                self.stop_base()
                self._spin_for(0.5)
                for _gate in range(10):  # max 10 attempts
                    det = self._find_target_raw(task.zone_cls)
                    if det is not None:
                        zone_fill = (det.x2 - det.x1) / CAMERA_WIDTH
                        self._log(f"[{lbl}] Zone gate: fill={zone_fill:.0%}", "info")
                        if zone_fill >= ZONE_FILL_MIN:
                            break
                        self._log(f"[{lbl}] Zone TOO FAR ({zone_fill:.0%} < {ZONE_FILL_MIN:.0%}) — driving closer", "warn")
                        self.drive_forward(0.10, speed=0.03)
                        self.phase_micro_center(task.zone_cls)
                        self.stop_base()
                        self._spin_for(0.3)
                    else:
                        self.drive_forward(0.08, speed=0.03)
                        self._spin_for(0.3)
            else:
                self._log(f"[{lbl}] Drop zone not found, using GT fallback — driving very close", "warn")
                dp = DROP_ZONE_POS[lbl]
                # Drive to within 5cm of the zone center
                self.navigate_to_point(dp[0] - 0.05, dp[1], stop_dist=0.03)
                self.turn_to_angle(0.0, slow=self._carrying)
            if self._check_abort():
                break

            # ---- 7. Place cube ----
            self.phase_place(task)

            # ---- 8. Reverse from drop zone ----
            self._set_state("REVERSING")
            self._set_status(f"[{lbl}] Backing up from zone...")
            self.drive_forward(-0.25, speed=0.04)
            self._spin_for(0.5)

            # Mark complete
            self.completed_colors.append(lbl)
            self.sig.color_chip_signal.emit(lbl, "done")
            self.sig.progress_signal.emit(len(self.completed_colors))
            self._log(f"[{lbl}] COMPLETE! ({len(self.completed_colors)}/3)", "ok")
            self._spin_for(1.0)

        # Return home
        if not self._estop and self._running:
            self._set_state("RETURNING HOME")
            hx, hy, hyaw = HOME_POS
            if self.nav2_available:
                self._set_status("Nav2 → home...")
                self.navigate_with_nav2(hx, hy, goal_yaw=hyaw)
            else:
                self._set_status("Returning home...")
                self.navigate_to_point(hx, hy, stop_dist=0.15)
            self.turn_to_angle(hyaw)

        # Safety cleanup — open gripper if still carrying
        if self._carrying:
            self.move_gripper(GRIPPER_OPEN, max_effort=5.0, timeout=2.0)
            self._carrying = False
        self.move_arm(ARM_HOME, 2.5)
        self._spin_for(1.0)

        self._set_state("DONE")
        if self._estop:
            self._set_status("Emergency stopped!")
            self._log("=== EMERGENCY STOP ===", "err")
        elif len(self.completed_colors) >= 3:
            self._set_status("All 3 cubes delivered!")
            self._log("=== MISSION COMPLETE ===", "ok")
        else:
            self._set_status("Mission stopped.")
            self._log("=== MISSION STOPPED ===", "warn")
        self._running = False


# =============================================================================
# PyQt5 GUI
# =============================================================================
STATE_COLORS = {
    "IDLE": "#888888",
    "NAV TO PICK": "#1E90FF",
    "SERVO TO CUBE": "#FF8C00",
    "PICKING": "#FF4500",
    "REVERSING": "#CD853F",
    "NAV TO DROP": "#9370DB",
    "SERVO TO ZONE": "#6A5ACD",
    "PLACING": "#32CD32",
    "RETURNING HOME": "#1E90FF",
    "PAUSED": "#FFA500",
    "DONE": "#00FF00",
}
LOG_COLORS = {
    "info": "#cdd6f4", "ok": "#a6e3a1",
    "warn": "#f9e2af", "err": "#f38ba8",
}


class MainWindow(QWidget):
    def __init__(self, node: NavPickPlaceNode, bridge: SignalBridge):
        super().__init__()
        self.node = node
        self.bridge = bridge
        self._ros_thread = None
        self._carto_proc = None
        self._nav2_proc = None
        self._mapping_active = False
        self._teleop_lin = 0.0
        self._teleop_ang = 0.0

        self.setWindowTitle("Nav Pick & Place  |  SLAM + Nav2 + YOLO")
        self.setMinimumSize(960, 880)
        self.setStyleSheet("""
            QWidget { background-color: #1e1e2e; color: #cdd6f4; }
            QGroupBox {
                border: 1px solid #45475a; border-radius: 6px;
                margin-top: 8px; padding-top: 14px;
                font-weight: bold; color: #89b4fa;
            }
            QGroupBox::title { subcontrol-position: top left; padding: 0 6px; }
        """)

        bs = """
            QPushButton {{
                background-color: {bg}; color: {fg};
                border: none; border-radius: 6px;
                padding: 8px 16px; font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #45475a; color: #6c7086; }}
        """

        # -- Camera --
        self.camera_label = QLabel("Waiting for camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet(
            "border: 2px solid #45475a; background: #181825;")

        # ---- SLAM Controls ----
        self.btn_start_map = QPushButton("Start Mapping")
        self.btn_start_map.setStyleSheet(
            bs.format(bg="#74c7ec", fg="#1e1e2e", hover="#89dceb"))
        self.btn_start_map.clicked.connect(self._on_start_mapping)

        self.btn_save_map = QPushButton("Save Map")
        self.btn_save_map.setStyleSheet(
            bs.format(bg="#a6e3a1", fg="#1e1e2e", hover="#94e2d5"))
        self.btn_save_map.clicked.connect(self._on_save_map)
        self.btn_save_map.setEnabled(False)

        self.btn_stop_map = QPushButton("Stop Mapping")
        self.btn_stop_map.setStyleSheet(
            bs.format(bg="#f38ba8", fg="#1e1e2e", hover="#eba0ac"))
        self.btn_stop_map.clicked.connect(self._on_stop_mapping)
        self.btn_stop_map.setEnabled(False)

        self.map_status = QLabel("Map: not saved")
        self.map_status.setFont(QFont("Monospace", 9))
        self.map_status.setStyleSheet("color: #6c7086;")
        if os.path.exists(MAP_YAML):
            self.map_status.setText("Map: saved")
            self.map_status.setStyleSheet("color: #a6e3a1;")

        self.teleop_hint = QLabel(
            "Teleop: W/A/S/D keys (active during mapping)")
        self.teleop_hint.setFont(QFont("Monospace", 9))
        self.teleop_hint.setStyleSheet("color: #585b70;")

        map_row = QHBoxLayout()
        map_row.addWidget(self.btn_start_map)
        map_row.addWidget(self.btn_save_map)
        map_row.addWidget(self.btn_stop_map)
        map_row.addWidget(self.map_status)
        map_row.addStretch()

        # ---- Nav2 Controls ----
        self.btn_start_nav2 = QPushButton("Start Nav2")
        self.btn_start_nav2.setStyleSheet(
            bs.format(bg="#cba6f7", fg="#1e1e2e", hover="#b4befe"))
        self.btn_start_nav2.clicked.connect(self._on_start_nav2)
        if not os.path.exists(MAP_YAML):
            self.btn_start_nav2.setEnabled(False)

        self.btn_stop_nav2 = QPushButton("Stop Nav2")
        self.btn_stop_nav2.setStyleSheet(
            bs.format(bg="#f38ba8", fg="#1e1e2e", hover="#eba0ac"))
        self.btn_stop_nav2.clicked.connect(self._on_stop_nav2)
        self.btn_stop_nav2.setEnabled(False)

        self.nav2_status = QLabel("Nav2: offline")
        self.nav2_status.setFont(QFont("Monospace", 9))
        self.nav2_status.setStyleSheet("color: #6c7086;")

        nav2_row = QHBoxLayout()
        nav2_row.addWidget(self.btn_start_nav2)
        nav2_row.addWidget(self.btn_stop_nav2)
        nav2_row.addWidget(self.nav2_status)
        nav2_row.addStretch()

        slam_group = QGroupBox("SLAM & Navigation")
        slam_lay = QVBoxLayout()
        slam_lay.addLayout(map_row)
        slam_lay.addWidget(self.teleop_hint)
        slam_lay.addLayout(nav2_row)
        slam_group.setLayout(slam_lay)

        # ---- Mission Controls ----
        self.btn_start = QPushButton("Start Mission")
        self.btn_start.setStyleSheet(
            bs.format(bg="#a6e3a1", fg="#1e1e2e", hover="#94e2d5"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setStyleSheet(
            bs.format(bg="#f9e2af", fg="#1e1e2e", hover="#fab387"))
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_pause.setEnabled(False)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(
            bs.format(bg="#f38ba8", fg="#1e1e2e", hover="#eba0ac"))
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)

        self.btn_estop = QPushButton("E-STOP")
        self.btn_estop.setStyleSheet("""
            QPushButton {
                background-color: #e64553; color: white;
                border: 2px solid #d20f39; border-radius: 6px;
                padding: 8px 18px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #d20f39; }
        """)
        self.btn_estop.clicked.connect(self._on_estop)

        mission_row = QHBoxLayout()
        mission_row.addWidget(self.btn_start)
        mission_row.addWidget(self.btn_pause)
        mission_row.addWidget(self.btn_stop)
        mission_row.addWidget(self.btn_estop)

        # ---- Status ----
        self.state_label = QLabel("State: IDLE")
        self.state_label.setFont(QFont("Monospace", 12, QFont.Bold))

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Monospace", 10))
        self.status_label.setStyleSheet("color: #a6adc8;")

        self.progress_label = QLabel("Progress: 0 / 3")
        self.progress_label.setFont(QFont("Monospace", 11, QFont.Bold))

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 3)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a; border-radius: 4px;
                background: #313244;
            }
            QProgressBar::chunk {
                background: #a6e3a1; border-radius: 3px;
            }
        """)

        # Color chips
        self.color_chips: Dict[str, QLabel] = {}
        chips_row = QHBoxLayout()
        for c in ["RED", "BLUE", "GREEN"]:
            chip = QLabel(f"  {c}  ")
            chip.setAlignment(Qt.AlignCenter)
            chip.setFixedSize(90, 28)
            chip.setFont(QFont("Monospace", 9, QFont.Bold))
            chip.setStyleSheet(self._chip_style("pending"))
            self.color_chips[c] = chip
            chips_row.addWidget(chip)
        chips_row.addStretch()

        info_group = QGroupBox("Mission Status")
        info_lay = QGridLayout()
        info_lay.addWidget(self.state_label, 0, 0)
        info_lay.addWidget(self.progress_label, 0, 1, Qt.AlignRight)
        info_lay.addWidget(self.status_label, 1, 0)
        info_lay.addWidget(self.progress_bar, 1, 1)
        info_lay.addLayout(chips_row, 2, 0, 1, 2)
        info_group.setLayout(info_lay)

        # ---- Activity Log ----
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Monospace", 9))
        self.log_box.setMaximumHeight(160)
        self.log_box.setStyleSheet(
            "background: #181825; border: 1px solid #45475a; border-radius: 4px;")

        log_group = QGroupBox("Activity Log")
        log_lay = QVBoxLayout()
        log_lay.addWidget(self.log_box)
        log_group.setLayout(log_lay)

        # ---- Main Layout ----
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.camera_label, 1)
        main_layout.addWidget(slam_group)
        main_layout.addLayout(mission_row)
        main_layout.addWidget(info_group)
        main_layout.addWidget(log_group)
        self.setLayout(main_layout)

        # ---- Signals ----
        bridge.log_signal.connect(self._on_log)
        bridge.state_signal.connect(self._on_state)
        bridge.status_signal.connect(self._on_status)
        bridge.progress_signal.connect(self._on_progress)
        bridge.color_chip_signal.connect(self._on_chip)
        bridge.mission_done_signal.connect(self._on_mission_done)

        # ---- Timers ----
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_camera)
        self.timer.start(50)

        self._teleop_timer = QTimer()
        self._teleop_timer.timeout.connect(self._publish_teleop)
        self._teleop_timer.start(100)

    # -- Chip styling --
    @staticmethod
    def _chip_style(state):
        if state == "active":
            return ("background: #fab387; color: #1e1e2e; "
                    "border-radius: 4px; border: 2px solid #f9e2af;")
        elif state == "done":
            return ("background: #a6e3a1; color: #1e1e2e; "
                    "border-radius: 4px; border: 2px solid #94e2d5;")
        return ("background: #45475a; color: #9399b2; "
                "border-radius: 4px; border: 2px solid #585b70;")

    # -- Signal slots --
    def _on_log(self, msg, level):
        color = LOG_COLORS.get(level, "#cdd6f4")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(
            f'<span style="color:#6c7086">[{ts}]</span> '
            f'<span style="color:{color}">{msg}</span>')
        self.log_box.moveCursor(QTextCursor.End)

    def _on_state(self, s):
        color = STATE_COLORS.get(s, "#888888")
        self.state_label.setText(f"State: {s}")
        self.state_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _on_status(self, s):
        self.status_label.setText(s)

    def _on_progress(self, count):
        self.progress_label.setText(f"Progress: {count} / 3")
        self.progress_bar.setValue(count)

    def _on_chip(self, label, state):
        if label in self.color_chips:
            self.color_chips[label].setStyleSheet(self._chip_style(state))

    def _refresh_camera(self):
        frame = self.node.get_display_frame()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            scaled = QPixmap.fromImage(qimg).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled)
        # Update Nav2 status label
        if self._nav2_proc is not None and self.node.nav2_available:
            self.nav2_status.setText("Nav2: running")
            self.nav2_status.setStyleSheet("color: #a6e3a1;")

    # ---- SLAM Mapping ----
    def _on_start_mapping(self):
        if self._carto_proc is not None:
            self._on_log("Mapping already running.", "warn")
            return
        self._on_log("Starting cartographer...", "info")
        try:
            self._carto_proc = subprocess.Popen(
                ["ros2", "launch", "turtlebot3_manipulation_cartographer",
                 "cartographer.launch.py",
                 "start_rviz:=true", "use_sim:=true"],
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self._on_log(f"Failed to start cartographer: {e}", "err")
            return
        self._mapping_active = True
        self.btn_start_map.setEnabled(False)
        self.btn_save_map.setEnabled(True)
        self.btn_stop_map.setEnabled(True)
        self.teleop_hint.setText(
            "TELEOP ACTIVE: W=fwd  S=back  A=left  D=right  X=stop")
        self.teleop_hint.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self._on_log("Cartographer launched. Drive with W/A/S/D to map.", "ok")
        self.setFocus()

    def _on_save_map(self):
        if not self._mapping_active:
            return
        self._on_log("Saving map...", "info")
        os.makedirs(MAP_DIR, exist_ok=True)
        try:
            result = subprocess.run(
                ["ros2", "run", "nav2_map_server", "map_saver_cli",
                 "-f", os.path.join(MAP_DIR, MAP_NAME),
                 "--ros-args", "-p", "use_sim_time:=true"],
                timeout=30, capture_output=True, text=True)
            if os.path.exists(MAP_YAML):
                self.map_status.setText("Map: saved")
                self.map_status.setStyleSheet("color: #a6e3a1;")
                self.btn_start_nav2.setEnabled(True)
                self._on_log(f"Map saved to {MAP_YAML}", "ok")
            else:
                self._on_log(
                    f"Map save may have failed: {result.stderr[:200]}", "warn")
        except subprocess.TimeoutExpired:
            self._on_log("Map save timed out.", "err")
        except Exception as e:
            self._on_log(f"Map save error: {e}", "err")

    def _on_stop_mapping(self):
        if self._carto_proc is not None:
            self._on_log("Stopping cartographer...", "info")
            try:
                os.killpg(os.getpgid(self._carto_proc.pid), py_signal.SIGINT)
                self._carto_proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(
                        os.getpgid(self._carto_proc.pid), py_signal.SIGKILL)
                except Exception:
                    pass
            self._carto_proc = None
        self._mapping_active = False
        self._teleop_lin = 0.0
        self._teleop_ang = 0.0
        self.node.cmd_pub.publish(Twist())
        self.btn_start_map.setEnabled(True)
        self.btn_save_map.setEnabled(False)
        self.btn_stop_map.setEnabled(False)
        self.teleop_hint.setText(
            "Teleop: W/A/S/D keys (active during mapping)")
        self.teleop_hint.setStyleSheet("color: #585b70;")
        self._on_log("Cartographer stopped.", "info")

    # ---- Nav2 ----
    def _on_start_nav2(self):
        if not os.path.exists(MAP_YAML):
            self._on_log("No map found! Map first, then start Nav2.", "err")
            return
        if self._nav2_proc is not None:
            self._on_log("Nav2 already running.", "warn")
            return
        if self._carto_proc is not None:
            self._on_stop_mapping()

        self._on_log("Starting Nav2 with saved map...", "info")
        self.nav2_status.setText("Nav2: starting...")
        self.nav2_status.setStyleSheet("color: #f9e2af;")
        try:
            self._nav2_proc = subprocess.Popen(
                ["ros2", "launch", "turtlebot3_manipulation_navigation2",
                 "navigation2_use_sim_time.launch.py",
                 f"map_yaml_file:={MAP_YAML}",
                 "start_rviz:=true", "use_sim:=true"],
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self._on_log(f"Failed to start Nav2: {e}", "err")
            self.nav2_status.setText("Nav2: failed")
            self.nav2_status.setStyleSheet("color: #f38ba8;")
            return

        self.btn_start_nav2.setEnabled(False)
        self.btn_stop_nav2.setEnabled(True)
        self._on_log("Nav2 launched. Waiting for action server...", "info")

        def _wait_nav2():
            for _ in range(60):
                time.sleep(2.0)
                if self._nav2_proc is None:
                    return
                if self.node.check_nav2_ready():
                    self.node.nav2_available = True
                    self.bridge.log_signal.emit(
                        "Nav2 ready! NavigateToPose available.", "ok")
                    # Publish initial pose for AMCL
                    odom_pose = self.node.get_odom_pose()
                    if odom_pose:
                        self.node.publish_initial_pose(
                            odom_pose[0], odom_pose[1], odom_pose[2])
                    return
            self.bridge.log_signal.emit(
                "Nav2 action server not found after timeout.", "warn")

        threading.Thread(target=_wait_nav2, daemon=True).start()

    def _on_stop_nav2(self):
        self.node.nav2_available = False
        if self._nav2_proc is not None:
            self._on_log("Stopping Nav2...", "info")
            try:
                os.killpg(os.getpgid(self._nav2_proc.pid), py_signal.SIGINT)
                self._nav2_proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(
                        os.getpgid(self._nav2_proc.pid), py_signal.SIGKILL)
                except Exception:
                    pass
            self._nav2_proc = None
        self.nav2_status.setText("Nav2: offline")
        self.nav2_status.setStyleSheet("color: #6c7086;")
        self.btn_start_nav2.setEnabled(True)
        self.btn_stop_nav2.setEnabled(False)
        self._on_log("Nav2 stopped.", "info")

    # ---- Teleop (keyboard during mapping) ----
    def keyPressEvent(self, event):
        if not self._mapping_active:
            return super().keyPressEvent(event)
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == Qt.Key_W:
            self._teleop_lin = TELEOP_LIN_SPEED
            self._teleop_ang = 0.0
        elif k == Qt.Key_S:
            self._teleop_lin = -TELEOP_LIN_SPEED
            self._teleop_ang = 0.0
        elif k == Qt.Key_A:
            self._teleop_lin = 0.0
            self._teleop_ang = TELEOP_ANG_SPEED
        elif k == Qt.Key_D:
            self._teleop_lin = 0.0
            self._teleop_ang = -TELEOP_ANG_SPEED
        elif k in (Qt.Key_X, Qt.Key_Space):
            self._teleop_lin = 0.0
            self._teleop_ang = 0.0
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not self._mapping_active:
            return super().keyReleaseEvent(event)
        if event.isAutoRepeat():
            return
        k = event.key()
        if k in (Qt.Key_W, Qt.Key_S):
            self._teleop_lin = 0.0
        elif k in (Qt.Key_A, Qt.Key_D):
            self._teleop_ang = 0.0
        else:
            super().keyReleaseEvent(event)

    def _publish_teleop(self):
        if not self._mapping_active:
            return
        if self._teleop_lin != 0 or self._teleop_ang != 0:
            msg = Twist()
            msg.linear.x = self._teleop_lin
            msg.angular.z = self._teleop_ang
            self.node.cmd_pub.publish(msg)

    # ---- Mission ----
    def _on_start(self):
        if self.node._running and self.node._paused:
            self.node._paused = False
            self.btn_pause.setEnabled(True)
            self._on_log("Resumed.", "ok")
            return
        if self.node._running:
            return
        if self._mapping_active:
            self._on_stop_mapping()

        # Reset UI
        for c in self.color_chips:
            self.color_chips[c].setStyleSheet(self._chip_style("pending"))
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0 / 3")
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

        def _worker():
            try:
                self.node.run()
            except Exception as e:
                self.node._log(f"Error: {e}", "err")
                import traceback
                self.node._log(traceback.format_exc(), "err")
            finally:
                self.bridge.mission_done_signal.emit()

        self._ros_thread = threading.Thread(target=_worker, daemon=True)
        self._ros_thread.start()

    def _on_mission_done(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("Pause")

    def _on_pause(self):
        if self.node._running:
            self.node._paused = not self.node._paused
            self.btn_pause.setText(
                "Resume" if self.node._paused else "Pause")
            self._on_log(
                "Paused." if self.node._paused else "Resumed.",
                "warn" if self.node._paused else "ok")

    def _on_stop(self):
        self.node._running = False
        self.node._paused = False
        z = Twist()
        for _ in range(5):
            self.node.cmd_pub.publish(z)
        self._on_log("Stop requested.", "warn")

    def _on_estop(self):
        self.node._estop = True
        self.node._running = False
        self.node._paused = False
        z = Twist()
        for _ in range(10):
            self.node.cmd_pub.publish(z)
        self._on_log("EMERGENCY STOP!", "err")

    def closeEvent(self, event):
        self.timer.stop()
        self._teleop_timer.stop()
        self.node._running = False
        self.node._estop = True
        z = Twist()
        for _ in range(5):
            self.node.cmd_pub.publish(z)
        for proc in [self._carto_proc, self._nav2_proc]:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), py_signal.SIGINT)
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), py_signal.SIGKILL)
                    except Exception:
                        pass
        event.accept()


# =============================================================================
# Main
# =============================================================================
def main():
    rclpy.init()
    bridge = SignalBridge()
    node = NavPickPlaceNode(bridge)

    # Background ROS spin when no mission is running
    def _ros_idle():
        while rclpy.ok():
            if not node._running:
                rclpy.spin_once(node, timeout_sec=0.05)
            else:
                time.sleep(0.05)

    idle_thread = threading.Thread(target=_ros_idle, daemon=True)
    idle_thread.start()

    app = QApplication(sys.argv)
    window = MainWindow(node, bridge)
    window.show()
    ret = app.exec_()

    try:
        node.stop_base()
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    sys.exit(ret)


if __name__ == "__main__":
    main()
