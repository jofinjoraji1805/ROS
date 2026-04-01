#!/usr/bin/env python3
"""
yolo_pick_place.py — YOLO-guided autonomous pick-and-place with REAL grasping.

The robot physically drives up to the table, fine-adjusts its position,
then uses the manipulator arm to actually grab and release cubes.
NO simulated teleport.

State flow per colour (RED → BLUE → GREEN):
  SEARCH_OBJECT → ALIGN_OBJECT → APPROACH_OBJECT → ADJUST_POSITION
  → PICK_OBJECT → BACKUP_PICK → SEARCH_DROP_ZONE → APPROACH_DROP
  → ADJUST_DROP → PLACE_OBJECT → BACKUP_DROP → NEXT_OBJECT
After all: RETURN_HOME → DONE

Usage:
  python3 yolo_pick_place.py     # Gazebo sim must already be running
"""

import sys
import os

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
    "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
)

import math
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFrame,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand as GripperCommandAction
from cv_bridge import CvBridge
from builtin_interfaces.msg import Duration

from ultralytics import YOLO


# =====================================================================
#  Configuration
# =====================================================================
YOLO_MODEL   = "/home/jofin/colcon_ws/src/tb3_pick_place/yolomodel/best.pt"
CAMERA_TOPIC = "/pi_camera/image_raw"
ODOM_TOPIC   = "/odom"
CONF_THRESH  = 0.25              # lower threshold for more reliable detection

# -- Approach parameters ------------------------------------------------
SCAN_ANG_VEL        = 0.15      # rad/s scan rotation
APPROACH_FWD        = 0.08      # m/s forward speed during APPROACH (straight driving)
MAX_ANG_VEL         = 0.25      # rad/s angular clamp (for ALIGN rotation)
BACKUP_VEL          = -0.06     # m/s reverse
BACKUP_PICK_TIME    = 3.5       # seconds backward after pick
BACKUP_DROP_TIME    = 2.5       # seconds backward after drop
VEL_RAMP            = 0.02      # max vel change per tick

# Phase 1: ALIGN_OBJECT — STOP, rotate in place only, center cube
# No forward movement. Just rotate to face the cube.
ALIGN_CENTER_PX     = 15        # ±15px from center = aligned enough to drive
ALIGN_KP            = 1.0       # proportional gain for rotation
ALIGN_LOST_MAX      = 40        # 4s patience before deciding

# Phase 2: APPROACH_OBJECT — drive STRAIGHT with tiny corrections
# Mostly straight driving. Only mild angular correction (max ±0.06 rad/s).
# If cube drifts too much (>50px), stop and go back to ALIGN.
APPROACH_DRIFT_PX   = 50        # max drift before stopping to re-align
APPROACH_MILD_KP    = 0.3       # very mild proportional correction
APPROACH_MILD_MAX   = 0.06      # max angular correction during straight driving
CUBE_LOST_TICKS     = 15        # 1.5s lost → cube under camera → ADJUST

# Phase 3: ADJUST_POSITION — blind final approach
ADJUST_FWD          = 0.020     # m/s very slow
ADJUST_DURATION     = 8.0       # seconds = ~0.16m (close the gap so gripper reaches)

# Zone approach — drive VERY CLOSE to drop zone before releasing
ZONE_CLOSE_AREA     = 0.015     # threshold to switch to final creep
ZONE_APPROACH_TIME  = 40.0
ZONE_FINAL_TIME     = 14.0      # longer final creep — get RIGHT UP to zone
ZONE_FINAL_VEL      = 0.04      # slower for precision

# Return home
RETURN_HOME_THRESH  = 0.20      # metres close-enough

# -- Arm joint poses [j1, j2, j3, j4] ---------------------------------
# Computed via ik_side() from OpenMANIPULATOR-X geometry:
#   BASE_TO_J2_R=0.012  BASE_TO_J2_H=0.0765  LA≈0.1302  LB=0.124
#   GRIPPER_OFFSET=0.10  SIDE_GRASP_TILT=0.0
#   Pick target: r=0.20m (radial from arm base), h=0.075m (cube centre above arm base)
#   Table top z=0.14, cube centre z=0.175, arm base z≈0.10 → h≈0.075
ARM_HOME      = [0.0, -1.0,   0.30,  0.70]   # tucked, camera clear
ARM_READY     = [0.0, -0.38,  0.80, -0.41]   # extended high (above table, safe transition)
ARM_PRE_PICK  = [0.0,  0.05,  1.05, -1.10]   # slightly above grasp height, further forward
ARM_PICK      = [0.0,  0.35,  1.10, -1.45]   # at cube level — reach further forward
ARM_LIFT      = [0.0,  0.05,  1.05, -1.10]   # lifted with cube
ARM_CARRY     = [0.0, -1.0,   0.30,  0.70]   # same as HOME for safe travel
ARM_PRE_DROP  = [0.0,  0.05,  1.05, -1.10]   # same as PRE_PICK
ARM_DROP      = [0.0,  0.35,  1.10, -1.45]   # same as PICK — reach forward to drop

GRIPPER_OPEN  =  0.019   # 52mm gap — clears 25mm cube
GRIPPER_CLOSE = -0.003   # partial close — firm hold without crushing/pushing

# -- State names -------------------------------------------------------
ST_IDLE             = "IDLE"
ST_SEARCH_OBJECT    = "SEARCH_OBJECT"
ST_ALIGN_OBJECT     = "ALIGN_OBJECT"
ST_APPROACH_OBJECT  = "APPROACH_OBJECT"
ST_ADJUST_POSITION  = "ADJUST_POSITION"
ST_PICK_OBJECT      = "PICK_OBJECT"
ST_BACKUP_PICK      = "BACKUP_PICK"
ST_SEARCH_DROP_ZONE = "SEARCH_DROP_ZONE"
ST_APPROACH_DROP    = "APPROACH_DROP"
ST_ADJUST_DROP      = "ADJUST_DROP"
ST_PLACE_OBJECT     = "PLACE_OBJECT"
ST_BACKUP_DROP      = "BACKUP_DROP"
ST_NEXT_OBJECT      = "NEXT_OBJECT"
ST_RETURN_HOME      = "RETURN_HOME"
ST_DONE             = "DONE"


# =====================================================================
#  Data types
# =====================================================================
@dataclass
class Detection:
    cls_id: int
    x1: float; y1: float; x2: float; y2: float
    conf: float

@dataclass
class TaskDef:
    cube_cls: int
    zone_cls: int
    label: str


# =====================================================================
#  Perception Module
# =====================================================================
class PerceptionModule:
    def __init__(self, model_path: str, conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf
        self.names: Dict[int, str] = self.model.names
        _n2i = {v: k for k, v in self.names.items()}

        self.CLS_RED_CUBE   = _n2i["red_cube"]
        self.CLS_RED_ZONE   = _n2i["red_drop_zone"]
        self.CLS_BLUE_CUBE  = _n2i["blue_cube"]
        self.CLS_BLUE_ZONE  = _n2i["blue_drop_zone"]
        self.CLS_GREEN_CUBE = _n2i["green_cube"]
        self.CLS_GREEN_ZONE = _n2i["green_drop_zone"]
        self.CUBE_IDS = {self.CLS_RED_CUBE, self.CLS_BLUE_CUBE, self.CLS_GREEN_CUBE}

        _cm = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 100, 0)}
        self.draw_colors: Dict[int, Tuple[int, ...]] = {}
        for cid, name in self.names.items():
            for key, bgr in _cm.items():
                if key in name:
                    self.draw_colors[cid] = bgr; break

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.conf, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                dets.append(Detection(int(box.cls[0]), x1, y1, x2, y2, float(box.conf[0])))
        return dets

    def find_target(self, dets, target_cls, img_w, img_h):
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
            best = d; best_conf = d.conf
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
            cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 2)
            lbl = f"{self.names[d.cls_id]} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (int(d.x1), int(d.y1)-th-6), (int(d.x1)+tw, int(d.y1)), c, -1)
            cv2.putText(vis, lbl, (int(d.x1), int(d.y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if hud:
            cv2.putText(vis, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return vis

    def build_tasks(self):
        return [
            TaskDef(self.CLS_RED_CUBE,   self.CLS_RED_ZONE,   "RED"),
            TaskDef(self.CLS_BLUE_CUBE,  self.CLS_BLUE_ZONE,  "BLUE"),
            TaskDef(self.CLS_GREEN_CUBE, self.CLS_GREEN_ZONE, "GREEN"),
        ]


# =====================================================================
#  Motion Control
# =====================================================================
class MotionController:
    def __init__(self, publisher):
        self._pub = publisher
        self._prev_lx = 0.0
        self._prev_az = 0.0

    def publish(self, lx=0.0, az=0.0):
        lx = max(-0.10, min(0.12, float(lx)))
        az = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, float(az)))
        dlx = lx - self._prev_lx
        if abs(dlx) > VEL_RAMP:
            lx = self._prev_lx + VEL_RAMP * (1.0 if dlx > 0 else -1.0)
        self._prev_lx = lx; self._prev_az = az
        msg = Twist(); msg.linear.x = lx; msg.angular.z = az
        self._pub.publish(msg)

    def stop(self):
        self._prev_lx = 0.0; self._prev_az = 0.0
        self._pub.publish(Twist())


# =====================================================================
#  Arm Control
# =====================================================================
class ArmController:
    def __init__(self, arm_pub, gripper_ac):
        self._arm_pub = arm_pub
        self._gripper_ac = gripper_ac

    def send_joint(self, positions, duration=2.0):
        msg = JointTrajectory()
        msg.joint_names = ["joint1", "joint2", "joint3", "joint4"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        s = int(duration); ns = int((duration - s) * 1e9)
        pt.time_from_start = Duration(sec=s, nanosec=ns)
        msg.points = [pt]
        self._arm_pub.publish(msg)

    def gripper(self, pos):
        g = GripperCommandAction.Goal()
        g.command.position = float(pos); g.command.max_effort = 20.0
        self._gripper_ac.send_goal_async(g)

    def open_gripper(self):  self.gripper(GRIPPER_OPEN)
    def close_gripper(self): self.gripper(GRIPPER_CLOSE)
    def home(self):          self.send_joint(ARM_HOME, 3.0)


# =====================================================================
#  Robot Controller + State Machine
# =====================================================================
class RobotController(Node):

    def __init__(self):
        super().__init__("yolo_pick_place")
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.get_logger().info(f"Loading YOLO model: {YOLO_MODEL}")
        self.perception = PerceptionModule(YOLO_MODEL, CONF_THRESH)
        self.get_logger().info(f"YOLO ready — classes: {list(self.perception.names.values())}")
        self.tasks = self.perception.build_tasks()

        cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        arm_pub = self.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
        gripper_ac = ActionClient(self, GripperCommandAction, "/gripper_controller/gripper_cmd")
        self.cam_sub = self.create_subscription(Image, CAMERA_TOPIC, self._cam_cb, 1)
        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 10)

        self.motion = MotionController(cmd_pub)
        self.arm = ArmController(arm_pub, gripper_ac)

        self.get_logger().info("Waiting for gripper action server...")
        gripper_ac.wait_for_server(timeout_sec=10.0)
        self.get_logger().info("All services ready!")

        # Shared state
        self.state = ST_IDLE
        self.task_idx = 0
        self.status_text = "Ready -- press START"
        self.running = False
        self.latest_frame = None
        self.annotated_frame = None
        self.detections: List[Detection] = []
        self.detected_label = "--"
        self.detected_distance = "--"

        # Odometry
        self._odom_x = 0.0; self._odom_y = 0.0; self._odom_yaw = 0.0
        self._home_x = 0.0;  self._home_y = 0.0; self._home_yaw = 0.0

        # Timing
        self._phase_start = 0.0
        self._pick_phase = 0; self._pick_timer = 0.0
        self._drop_phase = 0; self._drop_timer = 0.0
        self._zone_in_final = False; self._zone_final_start = 0.0
        self._lost_count = 0
        self._max_area = 0.0             # best area seen during tracking (for lost-cube decisions)

        self.create_timer(0.1, self._tick)

    # == Callbacks =====================================================
    def _odom_cb(self, msg):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        self._odom_x = p.x; self._odom_y = p.y
        self._odom_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    def _cam_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        dets = self.perception.detect(frame)
        lbl = self.tasks[self.task_idx].label if self.task_idx < len(self.tasks) else "DONE"
        vis = self.perception.annotate(frame, dets, f"[{lbl}] {self.state}")
        with self.lock:
            self.latest_frame = frame
            self.annotated_frame = vis
            self.detections = dets

    # == Helpers =======================================================
    def _find(self, target_cls):
        with self.lock:
            dets = list(self.detections); frame = self.latest_frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        r = self.perception.find_target(dets, target_cls, w, h)
        if r:
            self.detected_label = self.perception.names[target_cls]
            a = r[2]
            self.detected_distance = (
                "At Table" if a > 0.03 else
                "Very Close" if a > 0.01 else
                "Close" if a > 0.005 else
                "Medium" if a > 0.002 else
                "Far" if a > 0.001 else "Very Far"
            )
        return r

    def _find_raw(self, target_cls):
        """Return raw Detection object (for bbox width measurement)."""
        with self.lock:
            dets = list(self.detections)
        best, best_conf = None, 0.0
        for d in dets:
            if d.cls_id != target_cls or d.conf <= best_conf:
                continue
            best = d; best_conf = d.conf
        return best

    # == Mission =======================================================
    def start_mission(self):
        self._home_x = self._odom_x; self._home_y = self._odom_y
        self._home_yaw = self._odom_yaw
        self.get_logger().info(
            f"Home: ({self._home_x:.2f}, {self._home_y:.2f}, "
            f"yaw={math.degrees(self._home_yaw):.1f})"
        )
        self.task_idx = 0; self.state = ST_SEARCH_OBJECT; self.running = True
        self.arm.home(); self.arm.open_gripper()
        self.status_text = "Mission started -- searching for RED cube"
        self.get_logger().info(self.status_text)

    def stop_mission(self):
        self.motion.stop(); self.running = False; self.state = ST_IDLE
        self.arm.home()
        self.status_text = "Mission stopped by user"
        self.get_logger().info(self.status_text)

    # == State Machine (10 Hz) =========================================
    def _tick(self):
        if not self.running or self.state in (ST_DONE, ST_IDLE):
            return

        if self.task_idx >= len(self.tasks) and self.state != ST_RETURN_HOME:
            self.motion.stop(); self.arm.home()
            self.state = ST_RETURN_HOME
            self.status_text = "All cubes delivered -- returning home..."
            self.get_logger().info(self.status_text)
            return

        if self.state == ST_RETURN_HOME:
            self._run_return_home(); return

        task = self.tasks[self.task_idx]
        lbl = task.label

        # ------- SEARCH_OBJECT: rotate to find cube -------------------
        if self.state == ST_SEARCH_OBJECT:
            self.status_text = f"[{lbl}] Scanning for {self.perception.names[task.cube_cls]}..."
            target = self._find(task.cube_cls)
            if target:
                self.motion.stop()
                self.state = ST_ALIGN_OBJECT
                self._phase_start = time.time()
                self._lost_count = 0
                self._max_area = 0.0
                self.status_text = f"[{lbl}] Cube detected! Aligning..."
                self.get_logger().info(self.status_text)
            else:
                self.motion.publish(az=SCAN_ANG_VEL)

        # ------- ALIGN_OBJECT: STOP and rotate in place to center cube --
        # NO forward movement. Just face the cube squarely.
        # Once centered (±15px), transition to APPROACH (straight driving).
        elif self.state == ST_ALIGN_OBJECT:
            target = self._find(task.cube_cls)
            if target is None:
                self._lost_count += 1
                self.motion.stop()  # STOP — don't rotate or drive when lost
                if self._lost_count > ALIGN_LOST_MAX:
                    if self._max_area > 0.002:
                        # Was tracking well — keep going (go to APPROACH, NOT search)
                        self.state = ST_APPROACH_OBJECT
                        self._phase_start = time.time()
                        self._lost_count = 0
                        self.status_text = f"[{lbl}] Lost but was close -- driving forward..."
                        self.get_logger().info(self.status_text)
                    else:
                        self.state = ST_SEARCH_OBJECT
                        self._lost_count = 0
                self.status_text = f"[{lbl}] Cube lost -- waiting... ({self._lost_count})"
                return
            self._lost_count = 0
            cx, cy, area = target
            self._max_area = max(self._max_area, area)
            err = cx - 0.5
            pixel_err = err * 640

            if abs(pixel_err) <= ALIGN_CENTER_PX:
                # CENTERED — stop rotation, go to straight approach
                self.motion.stop()
                self.state = ST_APPROACH_OBJECT
                self._phase_start = time.time()
                self._lost_count = 0
                self.status_text = f"[{lbl}] Aligned ({pixel_err:+.0f}px) -- driving straight!"
                self.get_logger().info(self.status_text)
            else:
                # Rotate in place to center cube — NO forward movement
                ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_KP * err))
                # Min angular velocity to overcome friction
                if 0 < abs(ang) < 0.05:
                    ang = 0.05 * (1 if ang > 0 else -1)
                self.motion.publish(lx=0.0, az=ang)
                self.status_text = f"[{lbl}] Aligning  err={pixel_err:+.0f}px  area={area:.4f}"

        # ------- APPROACH_OBJECT: drive STRAIGHT with tiny corrections --
        # Mostly straight. If cube drifts >50px, stop and go back to ALIGN.
        elif self.state == ST_APPROACH_OBJECT:
            target = self._find(task.cube_cls)
            elapsed = time.time() - self._phase_start

            if target is None:
                self._lost_count += 1
                if self._lost_count > CUBE_LOST_TICKS:
                    # Cube under camera → blind final approach to table
                    self.motion.stop()
                    self.state = ST_ADJUST_POSITION
                    self._phase_start = time.time()
                    self.status_text = f"[{lbl}] Cube under camera -- final approach..."
                    self.get_logger().info(self.status_text)
                    return
                # Brief loss — keep driving straight
                self.motion.publish(lx=APPROACH_FWD, az=0.0)
                self.status_text = f"[{lbl}] Brief loss -- driving straight ({self._lost_count})"
                return
            self._lost_count = 0

            cx, cy, area = target
            self._max_area = max(self._max_area, area)
            err = cx - 0.5
            pixel_err = err * 640

            if abs(pixel_err) > APPROACH_DRIFT_PX:
                # Drifted too much — STOP, go back to ALIGN to re-center
                self.motion.stop()
                self.state = ST_ALIGN_OBJECT
                self._phase_start = time.time()
                self._lost_count = 0
                self.status_text = f"[{lbl}] Drift {pixel_err:+.0f}px -- re-aligning..."
                self.get_logger().info(self.status_text)
                return

            if elapsed > CLOSE_DURATION:
                self.motion.stop()
                self.state = ST_ADJUST_POSITION
                self._phase_start = time.time()
                self.status_text = f"[{lbl}] Timeout -- final approach..."
                self.get_logger().info(self.status_text)
                return

            # Drive STRAIGHT with very mild angular correction
            ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                       -APPROACH_MILD_KP * err))
            self.motion.publish(lx=APPROACH_FWD, az=ang)
            self.status_text = f"[{lbl}] Driving straight  err={pixel_err:+.0f}px  area={area:.4f}"

        # ------- ADJUST_POSITION: blind final approach to table ---------
        elif self.state == ST_ADJUST_POSITION:
            elapsed = time.time() - self._phase_start
            target = self._find(task.cube_cls)

            if elapsed < ADJUST_DURATION:
                if target:
                    # Cube briefly visible — use mild correction
                    err = target[0] - 0.5
                    ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                               -APPROACH_MILD_KP * err))
                    self.motion.publish(lx=ADJUST_FWD, az=ang)
                    self.status_text = f"[{lbl}] Final approach (tracking)  {ADJUST_DURATION - elapsed:.1f}s"
                else:
                    # Cube under camera — drive straight
                    self.motion.publish(lx=ADJUST_FWD, az=0.0)
                    self.status_text = f"[{lbl}] Final approach (blind)  {ADJUST_DURATION - elapsed:.1f}s"
            else:
                self.motion.stop()
                time.sleep(0.3)  # settle
                self.state = ST_PICK_OBJECT
                self._pick_phase = 0
                self._pick_timer = time.time()
                self.status_text = f"[{lbl}] In position -- starting grab..."
                self.get_logger().info(self.status_text)

        # ------- PICK_OBJECT: real arm grasp --------------------------
        elif self.state == ST_PICK_OBJECT:
            self._run_pick(lbl)

        # ------- BACKUP_PICK: reverse after grab ----------------------
        elif self.state == ST_BACKUP_PICK:
            elapsed = time.time() - self._phase_start
            if elapsed < BACKUP_PICK_TIME:
                self.motion.publish(lx=BACKUP_VEL)
                self.status_text = f"[{lbl}] Backing up... {BACKUP_PICK_TIME - elapsed:.1f}s"
            else:
                self.motion.stop()
                self.arm.home()
                self.state = ST_SEARCH_DROP_ZONE
                self._lost_count = 0
                self.status_text = f"[{lbl}] Scanning for {self.perception.names[task.zone_cls]}..."
                self.get_logger().info(self.status_text)

        # ------- SEARCH_DROP_ZONE: rotate right -----------------------
        elif self.state == ST_SEARCH_DROP_ZONE:
            self.status_text = f"[{lbl}] Scanning for {self.perception.names[task.zone_cls]}..."
            target = self._find(task.zone_cls)
            if target:
                self.motion.stop()
                self.state = ST_APPROACH_DROP
                self._phase_start = time.time()
                self._zone_in_final = False
                self._lost_count = 0
                self.status_text = f"[{lbl}] Drop zone found! Approaching..."
                self.get_logger().info(self.status_text)
            else:
                self.motion.publish(az=-SCAN_ANG_VEL)

        # ------- APPROACH_DROP: drive toward drop zone ----------------
        elif self.state == ST_APPROACH_DROP:
            target = self._find(task.zone_cls)
            elapsed = time.time() - self._phase_start

            if not self._zone_in_final:
                if target is None:
                    self._lost_count += 1
                    if self._lost_count > 30:
                        self.state = ST_SEARCH_DROP_ZONE
                    else:
                        self.motion.publish(lx=APPROACH_FWD, az=0.0)
                    return
                self._lost_count = 0
                cx, cy, area = target
                err = cx - 0.5
                pixel_err = err * 640

                if area >= ZONE_CLOSE_AREA or elapsed > ZONE_APPROACH_TIME:
                    self.motion.stop()
                    self._zone_in_final = True
                    self._zone_final_start = time.time()
                    self.status_text = f"[{lbl}] Near zone -- final approach..."
                    self.get_logger().info(self.status_text)
                elif abs(pixel_err) > APPROACH_DRIFT_PX:
                    # Off-center — rotate in place to re-align
                    ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_KP * err))
                    self.motion.publish(lx=0.0, az=ang)
                    self.status_text = f"[{lbl}] Aligning to zone  err={pixel_err:+.0f}px"
                else:
                    # Drive straight with mild correction
                    ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                               -APPROACH_MILD_KP * err))
                    self.motion.publish(lx=APPROACH_FWD, az=ang)
                    self.status_text = f"[{lbl}] Driving to zone  err={pixel_err:+.0f}px  area={area:.4f}"
            else:
                fe = time.time() - self._zone_final_start
                if fe < ZONE_FINAL_TIME:
                    if target:
                        err_z = target[0] - 0.5
                        ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                                   -APPROACH_MILD_KP * err_z))
                        self.motion.publish(lx=ZONE_FINAL_VEL, az=ang)
                    else:
                        self.motion.publish(lx=ZONE_FINAL_VEL, az=0.0)
                    self.status_text = f"[{lbl}] Final approach to zone... {ZONE_FINAL_TIME - fe:.1f}s"
                else:
                    self.motion.stop()
                    self.state = ST_ADJUST_DROP
                    self._phase_start = time.time()
                    self.status_text = f"[{lbl}] Adjusting at drop zone..."
                    self.get_logger().info(self.status_text)

        # ------- ADJUST_DROP: fine position at drop zone --------------
        elif self.state == ST_ADJUST_DROP:
            elapsed = time.time() - self._phase_start
            if elapsed < 8.0:
                target = self._find(task.zone_cls)
                if target:
                    err = target[0] - 0.5
                    ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                               -APPROACH_MILD_KP * err))
                    self.motion.publish(lx=ADJUST_FWD, az=ang)
                else:
                    self.motion.publish(lx=ADJUST_FWD, az=0.0)
                self.status_text = f"[{lbl}] Fine adjust at zone... {8.0 - elapsed:.1f}s"
            else:
                self.motion.stop()
                time.sleep(0.3)
                self.state = ST_PLACE_OBJECT
                self._drop_phase = 0; self._drop_timer = time.time()
                self.status_text = f"[{lbl}] In position -- placing cube..."
                self.get_logger().info(self.status_text)

        # ------- PLACE_OBJECT: real arm release -----------------------
        elif self.state == ST_PLACE_OBJECT:
            self._run_drop(lbl)

        # ------- BACKUP_DROP: reverse after placing -------------------
        elif self.state == ST_BACKUP_DROP:
            elapsed = time.time() - self._phase_start
            if elapsed < BACKUP_DROP_TIME:
                self.motion.publish(lx=BACKUP_VEL)
                self.status_text = f"[{lbl}] Retreating... {BACKUP_DROP_TIME - elapsed:.1f}s"
            else:
                self.motion.stop()
                self.state = ST_NEXT_OBJECT

        # ------- NEXT_OBJECT ------------------------------------------
        elif self.state == ST_NEXT_OBJECT:
            self.task_idx += 1
            if self.task_idx < len(self.tasks):
                nxt = self.tasks[self.task_idx].label
                self.arm.home()
                self.state = ST_SEARCH_OBJECT
                self.status_text = f"[{nxt}] Next -- scanning for {nxt} cube..."
                self.get_logger().info(self.status_text)

    # == REAL Pick sequence (IK side-grasp) ============================
    def _run_pick(self, lbl):
        t = time.time() - self._pick_timer
        p = self._pick_phase

        if p == 0:
            self.status_text = f"[{lbl}] Opening gripper wide..."
            self.arm.open_gripper()
            self._pick_phase = 1; self._pick_timer = time.time()

        elif p == 1 and t > 2.0:
            self.status_text = f"[{lbl}] Arm high -- safe above table..."
            self.arm.send_joint(ARM_READY, 3.0)
            self._pick_phase = 2; self._pick_timer = time.time()

        elif p == 2 and t > 3.5:
            self.status_text = f"[{lbl}] Lowering to pre-pick height..."
            self.arm.send_joint(ARM_PRE_PICK, 3.0)
            self._pick_phase = 3; self._pick_timer = time.time()

        elif p == 3 and t > 3.5:
            self.status_text = f"[{lbl}] Side-extending to cube level..."
            self.arm.send_joint(ARM_PICK, 3.0)
            self._pick_phase = 4; self._pick_timer = time.time()

        elif p == 4 and t > 4.0:
            self.status_text = f"[{lbl}] Gripper closing -- step 1 (loose)..."
            self.arm.gripper(0.005)          # barely closing — approaching cube
            self._pick_phase = 5; self._pick_timer = time.time()

        elif p == 5 and t > 1.5:
            self.status_text = f"[{lbl}] Gripper closing -- step 2 (contact)..."
            self.arm.gripper(0.000)          # making contact with cube
            self._pick_phase = 6; self._pick_timer = time.time()

        elif p == 6 and t > 1.5:
            self.status_text = f"[{lbl}] Gripper closing -- step 3 (grip)..."
            self.arm.gripper(GRIPPER_CLOSE)  # firm partial hold (-0.003)
            self._pick_phase = 7; self._pick_timer = time.time()

        elif p == 7 and t > 2.0:
            self.status_text = f"[{lbl}] Grip secured -- lifting..."
            self.arm.send_joint(ARM_LIFT, 3.0)
            self._pick_phase = 8; self._pick_timer = time.time()

        elif p == 8 and t > 3.5:
            self.status_text = f"[{lbl}] Tucking arm for travel..."
            self.arm.send_joint(ARM_CARRY, 3.0)
            self._pick_phase = 9; self._pick_timer = time.time()

        elif p == 9 and t > 3.5:
            self.status_text = f"[{lbl}] Grab complete -- backing up safely..."
            self.get_logger().info(self.status_text)
            self.state = ST_BACKUP_PICK
            self._phase_start = time.time()

    # == REAL Drop sequence (side-grasp release) =======================
    def _run_drop(self, lbl):
        t = time.time() - self._drop_timer
        p = self._drop_phase

        if p == 0:
            self.status_text = f"[{lbl}] Extending arm to pre-drop..."
            self.arm.send_joint(ARM_PRE_DROP, 3.0)
            self._drop_phase = 1; self._drop_timer = time.time()

        elif p == 1 and t > 3.5:
            self.status_text = f"[{lbl}] Lowering to drop height..."
            self.arm.send_joint(ARM_DROP, 3.0)
            self._drop_phase = 2; self._drop_timer = time.time()

        elif p == 2 and t > 4.0:
            self.status_text = f"[{lbl}] OPENING GRIPPER -- releasing cube..."
            self.arm.open_gripper()
            self._drop_phase = 3; self._drop_timer = time.time()

        elif p == 3 and t > 2.5:
            self.status_text = f"[{lbl}] Release confirmed -- retracting..."
            self.arm.send_joint(ARM_LIFT, 3.0)
            self._drop_phase = 4; self._drop_timer = time.time()

        elif p == 4 and t > 3.5:
            self.status_text = f"[{lbl}] Arm -> home..."
            self.arm.home()
            self._drop_phase = 5; self._drop_timer = time.time()

        elif p == 5 and t > 3.0:
            self.status_text = f"[{lbl}] Cube placed at {lbl} drop zone!"
            self.get_logger().info(self.status_text)
            self.state = ST_BACKUP_DROP
            self._phase_start = time.time()

    # == Return Home ===================================================
    def _run_return_home(self):
        dx = self._home_x - self._odom_x
        dy = self._home_y - self._odom_y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < RETURN_HOME_THRESH:
            yaw_err = self._home_yaw - self._odom_yaw
            while yaw_err > math.pi:  yaw_err -= 2*math.pi
            while yaw_err < -math.pi: yaw_err += 2*math.pi
            if abs(yaw_err) > 0.10:
                self.motion.publish(az=max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3*yaw_err)))
                self.status_text = f"Correcting heading... {math.degrees(yaw_err):+.1f} deg"
            else:
                self.motion.stop(); self.arm.home()
                self.state = ST_DONE; self.running = False
                self.status_text = "HOME -- ALL TASKS COMPLETE!"
                self.get_logger().info(self.status_text)
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err = target_yaw - self._odom_yaw
        while yaw_err > math.pi:  yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi

        if abs(yaw_err) > 0.15:
            self.motion.publish(az=max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3*yaw_err)))
            self.status_text = f"Returning -- rotating ({math.degrees(yaw_err):+.1f} deg, {dist:.2f}m)"
        else:
            fwd = min(APPROACH_FWD, 0.5*dist)
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3*yaw_err))
            self.motion.publish(lx=fwd, az=ang)
            self.status_text = f"Returning home -- {dist:.2f}m remaining"


# =====================================================================
#  PyQt5 GUI
# =====================================================================
class PickPlaceGUI(QMainWindow):
    _SC = {
        ST_IDLE: "#95a5a6", ST_SEARCH_OBJECT: "#f39c12",
        ST_ALIGN_OBJECT: "#e67e22", ST_APPROACH_OBJECT: "#d35400",
        ST_ADJUST_POSITION: "#e74c3c", ST_PICK_OBJECT: "#c0392b",
        ST_BACKUP_PICK: "#9b59b6", ST_SEARCH_DROP_ZONE: "#3498db",
        ST_APPROACH_DROP: "#2980b9", ST_ADJUST_DROP: "#2471a3",
        ST_PLACE_OBJECT: "#27ae60", ST_BACKUP_DROP: "#8e44ad",
        ST_NEXT_OBJECT: "#1abc9c", ST_RETURN_HOME: "#f1c40f",
        ST_DONE: "#2ecc71",
    }

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.setWindowTitle("YOLO Pick & Place -- Real Grasping")
        self.setMinimumSize(1000, 580)
        self.setStyleSheet("background-color: #0a0a1a;")

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(8,8,8,8); root.setSpacing(8)

        # Camera
        cf = QFrame(); cf.setStyleSheet("QFrame{background:#111;border:2px solid #333;border-radius:10px}")
        cl = QVBoxLayout(cf)
        ct = QLabel("Camera (YOLO)"); ct.setFont(QFont("Segoe UI",11,QFont.Bold))
        ct.setStyleSheet("color:#aaa;border:none"); ct.setAlignment(Qt.AlignCenter); cl.addWidget(ct)
        self.cam = QLabel("Waiting..."); self.cam.setMinimumSize(640,480)
        self.cam.setAlignment(Qt.AlignCenter); self.cam.setStyleSheet("color:#555;border:none")
        cl.addWidget(self.cam); root.addWidget(cf, stretch=3)

        # Panel
        pf = QFrame(); pf.setStyleSheet("QFrame{background:#111;border:2px solid #333;border-radius:10px}")
        r = QVBoxLayout(pf); r.setSpacing(10)

        t = QLabel("Mission Control"); t.setFont(QFont("Segoe UI",16,QFont.Bold))
        t.setStyleSheet("color:#e94560;border:none"); t.setAlignment(Qt.AlignCenter); r.addWidget(t)

        self.sl = QLabel("IDLE"); self.sl.setFont(QFont("Consolas",13,QFont.Bold))
        self.sl.setWordWrap(True); self.sl.setAlignment(Qt.AlignCenter)
        self.sl.setStyleSheet("padding:12px;background:#16213e;color:#f39c12;border-radius:8px;border:1px solid #1a1a2e")
        r.addWidget(self.sl)

        self.dl = QLabel("Detected: -- | Distance: --"); self.dl.setFont(QFont("Segoe UI",10))
        self.dl.setAlignment(Qt.AlignCenter); self.dl.setStyleSheet("color:#bbb;border:none;padding:4px"); r.addWidget(self.dl)

        self.tl = QLabel("Task: --"); self.tl.setFont(QFont("Segoe UI",11))
        self.tl.setStyleSheet("color:#ccc;border:none;padding:4px"); self.tl.setAlignment(Qt.AlignCenter); r.addWidget(self.tl)

        self.pl = QLabel("Progress: 0/3"); self.pl.setFont(QFont("Segoe UI",11))
        self.pl.setAlignment(Qt.AlignCenter); self.pl.setStyleSheet("color:#7ec8e3;border:none"); r.addWidget(self.pl)

        il = QHBoxLayout(); self.ind = {}
        for c, hx in [("RED","#e74c3c"),("BLUE","#3498db"),("GREEN","#2ecc71")]:
            lb = QLabel(f"  {c}  "); lb.setFont(QFont("Segoe UI",10,QFont.Bold))
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet(f"background:#222;color:{hx};border:2px solid #333;border-radius:6px;padding:6px")
            il.addWidget(lb); self.ind[c] = lb
        r.addLayout(il)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Consolas",9))
        self.log.setStyleSheet("background:#0f3460;color:#e2e2e2;border-radius:8px;padding:6px;border:1px solid #1a1a2e")
        r.addWidget(self.log, stretch=1)

        bl = QHBoxLayout()
        self.sb = QPushButton("START MISSION"); self.sb.setFont(QFont("Segoe UI",12,QFont.Bold))
        self.sb.setCursor(Qt.PointingHandCursor)
        self.sb.setStyleSheet("QPushButton{background:#e94560;color:white;padding:12px;border-radius:10px;border:none}"
                              "QPushButton:hover{background:#c0392b}QPushButton:disabled{background:#555;color:#999}")
        self.sb.clicked.connect(self._start); bl.addWidget(self.sb)

        self.xb = QPushButton("STOP"); self.xb.setFont(QFont("Segoe UI",12,QFont.Bold))
        self.xb.setCursor(Qt.PointingHandCursor); self.xb.setEnabled(False)
        self.xb.setStyleSheet("QPushButton{background:#333;color:#666;padding:12px;border-radius:10px;border:none}")
        self.xb.clicked.connect(self._stop); bl.addWidget(self.xb)
        r.addLayout(bl); root.addWidget(pf, stretch=1)

        self._ls = ""; self._t = QTimer(); self._t.timeout.connect(self._upd); self._t.start(33)

    def _start(self):
        if not self.ctrl.running:
            self.ctrl.start_mission()
            self.sb.setText("RUNNING..."); self.sb.setEnabled(False)
            self.xb.setEnabled(True)
            self.xb.setStyleSheet("QPushButton{background:#e67e22;color:white;padding:12px;border-radius:10px;border:none}"
                                  "QPushButton:hover{background:#d35400}")
            self.log.append("Mission: RED -> BLUE -> GREEN")

    def _stop(self):
        if self.ctrl.running:
            self.ctrl.stop_mission()
            self.sb.setText("START MISSION"); self.sb.setEnabled(True)
            self.xb.setEnabled(False)
            self.xb.setStyleSheet("QPushButton{background:#333;color:#666;padding:12px;border-radius:10px;border:none}")
            self.log.append("STOPPED")

    def _upd(self):
        with self.ctrl.lock:
            f = self.ctrl.annotated_frame
        if f is not None:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qi = QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
            self.cam.setPixmap(QPixmap.fromImage(qi).scaled(self.cam.width(),self.cam.height(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

        st = self.ctrl.state; ss = self.ctrl.status_text; ti = self.ctrl.task_idx
        tasks = self.ctrl.tasks; nt = len(tasks)
        sc = self._SC.get(st, "#ecf0f1")
        self.sl.setText(st.replace("_"," "))
        self.sl.setStyleSheet(f"padding:12px;background:#16213e;color:{sc};border-radius:8px;font-size:14px;font-weight:bold;border:1px solid {sc}")
        self.dl.setText(f"Detected: {self.ctrl.detected_label} | Distance: {self.ctrl.detected_distance}")

        cp = min(ti, nt)
        if st == ST_RETURN_HOME:
            self.tl.setText("Returning to start...")
        elif ti < nt:
            self.tl.setText(f"Current: {tasks[ti].label} cube -> {tasks[ti].label} drop zone")
        else:
            self.tl.setText("All delivered!")
        self.pl.setText(f"Progress: {cp}/{nt}  " + "\u25cf"*cp + "\u25cb"*(nt-cp))

        for i,c in enumerate(["RED","BLUE","GREEN"]):
            hx = {"RED":"#e74c3c","BLUE":"#3498db","GREEN":"#2ecc71"}[c]
            lb = self.ind[c]
            if i < cp:
                lb.setStyleSheet("background:#1a3a1a;color:#2ecc71;border:2px solid #2ecc71;border-radius:6px;padding:6px")
            elif i == ti and st not in (ST_DONE, ST_RETURN_HOME):
                lb.setStyleSheet(f"background:#1a1a3e;color:{hx};border:2px solid {hx};border-radius:6px;padding:6px;font-weight:bold")
            else:
                lb.setStyleSheet(f"background:#222;color:{hx};border:2px solid #333;border-radius:6px;padding:6px")

        if ss != self._ls:
            self._ls = ss; self.log.append(ss)
            sb = self.log.verticalScrollBar(); sb.setValue(sb.maximum())

        if st == ST_DONE:
            self.sb.setText("MISSION COMPLETE")
            self.sb.setStyleSheet("QPushButton{background:#27ae60;color:white;padding:12px;border-radius:10px;border:none}")
            self.xb.setEnabled(False)
            self.xb.setStyleSheet("QPushButton{background:#333;color:#666;padding:12px;border-radius:10px;border:none}")

    def closeEvent(self, e):
        self.ctrl.motion.stop(); self.ctrl.running = False; e.accept()


# =====================================================================
def main():
    rclpy.init()
    ctrl = RobotController()
    threading.Thread(target=rclpy.spin, args=(ctrl,), daemon=True).start()
    app = QApplication(sys.argv); app.setStyle("Fusion")
    gui = PickPlaceGUI(ctrl); gui.show()
    ec = app.exec_()
    ctrl.motion.stop(); ctrl.destroy_node(); rclpy.shutdown(); sys.exit(ec)

if __name__ == "__main__":
    main()
