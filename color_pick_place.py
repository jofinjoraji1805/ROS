#!/usr/bin/env python3
"""
color_pick_place.py  --  OpenCV color-segmentation autonomous pick-and-place
                         with PyQt5 GUI for TurtleBot3 + OpenManipulator-X.

Sequence: RED -> BLUE -> GREEN.
For each color the robot:
  1. Scans (rotates) to find the colored cube via HSV segmentation
  2. Visually servos toward the cube
  3. Picks up the cube (arm sequence + Gazebo teleport)
  4. Backs up to clear the table
  5. Scans for the matching colored drop zone
  6. Approaches the drop zone
  7. Drops the cube (arm + teleport)
  8. Proceeds to next color

Detection uses OpenCV HSV color segmentation with morphological filtering
and contour analysis.  Cubes vs drop zones are distinguished by contour area.

Usage:
  python3 color_pick_place.py
  (Gazebo simulation with pick_place_world.world must already be running)
"""

import sys
import os

# Use system Qt plugins (avoid OpenCV bundled Qt conflicts)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
    "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
)

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Import PyQt5 BEFORE cv2 to avoid Qt conflicts
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
from geometry_msgs.msg import Twist, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand as GripperCommandAction
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from cv_bridge import CvBridge
from builtin_interfaces.msg import Duration


# =====================================================================
#  Configuration
# =====================================================================
CAMERA_TOPIC = "/pi_camera/image_raw"
ROBOT_MODEL = "turtlebot3_manipulation_system"

# -- HSV colour ranges (OpenCV: H 0-179, S 0-255, V 0-255) -----------
# Tuned for pick_place_world.world Gazebo colours:
#   Red   cube (0.9,0.2,0.2)  -> H~0    | zone walls (0.9,0.2,0.6) -> H~163
#   Blue  cube (0.1,0.1,0.9)  -> H~120  | zone walls (0.0,0.7,0.9) -> H~97
#   Green cube (0.1,0.8,0.1)  -> H~60   | zone walls (0.6,0.9,0.1) -> H~41
# Red wraps around 0/179 so needs two ranges.
HSV_RANGES = {
    "red": [
        ((0, 100, 80), (10, 255, 255)),       # low-red (cube)
        ((155, 80, 80), (179, 255, 255)),      # high-red / magenta (zone walls)
    ],
    "blue": [
        ((85, 80, 60), (130, 255, 255)),       # cyan through blue
    ],
    "green": [
        ((35, 80, 60), (85, 255, 255)),        # lime through green
    ],
}

# Contour-area thresholds (fraction of image area) to classify detections
CUBE_AREA_MIN = 0.00015   # ignore specks smaller than this
CUBE_AREA_MAX = 0.025     # cubes are small objects
ZONE_AREA_MIN = 0.004     # drop-zone walls produce bigger contours
ZONE_AREA_MAX = 0.60      # sanity upper bound

# Morphological kernel size for noise filtering
MORPH_KERNEL_SIZE = 5

# -- Motion parameters ------------------------------------------------
SCAN_ANG_VEL        = 0.08     # rad/s  gentle scan rotation
APPROACH_FWD        = 0.08     # m/s    forward during approach
APPROACH_KP         = 0.3      # proportional gain for centering
MAX_ANG_VEL         = 0.10     # rad/s  angular velocity clamp
BACKUP_VEL          = -0.05    # m/s    backward after pick
BACKUP_TIME         = 3.0      # seconds to back up
FINAL_APPROACH_TIME = 8.0      # seconds timed creep to get in arm reach
FINAL_APPROACH_VEL  = 0.04     # m/s    slow final creep
VEL_RAMP            = 0.005    # max velocity change per tick (smooth ramp)
MAX_APPROACH_TIME   = 45.0     # seconds  force final approach if not close

# Proximity thresholds (bbox area fraction)
CUBE_CLOSE_AREA = 0.002
ZONE_CLOSE_AREA = 0.008

# -- Arm joint poses [joint1, joint2, joint3, joint4] ------------------
ARM_HOME     = [0.0, -1.05,  0.35,  0.70]
ARM_PRE_PICK = [0.0,  0.10,  0.05, -0.55]
ARM_PICK     = [0.0,  0.40, -0.10, -0.70]
ARM_LIFT     = [0.0, -0.10,  0.10, -0.40]
ARM_CARRY    = [0.0, -0.50,  0.10,  0.40]
ARM_PRE_DROP = [0.0,  0.10,  0.05, -0.55]
ARM_DROP     = [0.0,  0.30, -0.05, -0.65]

GRIPPER_OPEN  =  0.01
GRIPPER_CLOSE = -0.01

# -- State names -------------------------------------------------------
ST_IDLE          = "IDLE"
ST_SCAN_CUBE     = "SCAN_CUBE"
ST_APPROACH_CUBE = "APPROACH_CUBE"
ST_FINAL_CUBE    = "FINAL_APPROACH_CUBE"
ST_PICK          = "PICK"
ST_BACKUP        = "BACKUP"
ST_SCAN_ZONE     = "SCAN_ZONE"
ST_APPROACH_ZONE = "APPROACH_ZONE"
ST_FINAL_ZONE    = "FINAL_APPROACH_ZONE"
ST_DROP          = "DROP"
ST_NEXT          = "NEXT"
ST_DONE          = "DONE"

# -- Task sequence -----------------------------------------------------
# (color_key, gazebo_model_name, drop_x, drop_y, drop_z, display_label)
TASKS = [
    ("red",   "red_cube",   2.0,  1.5, 0.025, "RED"),
    ("blue",  "blue_cube",  2.0, -1.5, 0.025, "BLUE"),
    ("green", "green_cube", 2.0,  0.0, 0.025, "GREEN"),
]

# BGR draw colours for annotation overlay
DRAW_BGR = {
    "red":   (0, 0, 255),
    "blue":  (255, 100, 0),
    "green": (0, 200, 0),
}


# =====================================================================
#  Vision Module -- OpenCV Colour Segmentation
# =====================================================================
@dataclass
class Detection:
    """Single colour-segmentation detection."""
    color: str          # "red" / "blue" / "green"
    obj_type: str       # "cube" or "zone"
    x1: int
    y1: int
    x2: int
    y2: int
    area_norm: float    # bounding-box area / image area
    cx_norm: float      # centre-x normalised [0..1]
    cy_norm: float      # centre-y normalised [0..1]


class ColorDetector:
    """Detects coloured cubes and drop zones using HSV segmentation."""

    def __init__(self):
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )

    # -----------------------------------------------------------------
    def _make_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """Create binary mask for *color*, combining multiple HSV ranges
        (handles the red wrap-around) and applying morphological filtering."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # Open  -> remove small noise specks
        # Close -> fill small holes inside detected regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask

    # -----------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run colour segmentation on a BGR frame; return all detections."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        img_area = float(h * w)
        detections: List[Detection] = []

        for color in HSV_RANGES:
            mask = self._make_mask(hsv, color)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                area_frac = cnt_area / img_area

                if area_frac < CUBE_AREA_MIN:
                    continue  # noise

                bx, by, bw, bh = cv2.boundingRect(cnt)
                bbox_area = (bw * bh) / img_area

                # Classify by contour area
                if area_frac <= CUBE_AREA_MAX:
                    obj_type = "cube"
                elif area_frac >= ZONE_AREA_MIN:
                    obj_type = "zone"
                else:
                    # Falls in the gap -- use bbox aspect & size heuristic
                    obj_type = "zone" if bbox_area > 0.01 else "cube"

                detections.append(Detection(
                    color=color,
                    obj_type=obj_type,
                    x1=bx, y1=by,
                    x2=bx + bw, y2=by + bh,
                    area_norm=bbox_area,
                    cx_norm=(bx + bw / 2.0) / w,
                    cy_norm=(by + bh / 2.0) / h,
                ))

        return detections

    # -----------------------------------------------------------------
    def annotate(
        self, frame: np.ndarray, detections: List[Detection],
        state: str = "", label: str = "",
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and HUD overlay onto a copy of *frame*."""
        vis = frame.copy()

        for det in detections:
            bgr = DRAW_BGR.get(det.color, (255, 255, 255))
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), bgr, 2)

            text = f"{det.color} {det.obj_type} {det.area_norm:.4f}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis, (det.x1, det.y1 - th - 6),
                (det.x1 + tw, det.y1), bgr, -1
            )
            cv2.putText(
                vis, text, (det.x1, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # HUD: current state
        hud = f"[{label}] {state}" if label else state
        cv2.putText(
            vis, hud, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        return vis


# =====================================================================
#  Motion Control Module
# =====================================================================
class MotionController:
    """Smooth ramped velocity publisher for the mobile base."""

    def __init__(self, publisher):
        self._pub = publisher
        self._prev_lx = 0.0
        self._prev_az = 0.0

    def publish(self, lx: float = 0.0, az: float = 0.0):
        lx = max(-0.08, min(0.08, float(lx)))
        az = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, float(az)))
        # Ramp: limit change per tick
        dlx = lx - self._prev_lx
        daz = az - self._prev_az
        if abs(dlx) > VEL_RAMP:
            lx = self._prev_lx + VEL_RAMP * (1.0 if dlx > 0 else -1.0)
        if abs(daz) > VEL_RAMP:
            az = self._prev_az + VEL_RAMP * (1.0 if daz > 0 else -1.0)
        self._prev_lx = lx
        self._prev_az = az

        msg = Twist()
        msg.linear.x = lx
        msg.angular.z = az
        self._pub.publish(msg)

    def stop(self):
        self._prev_lx = 0.0
        self._prev_az = 0.0
        self._pub.publish(Twist())


# =====================================================================
#  Manipulator Control Module
# =====================================================================
class ArmController:
    """Commands for the OpenManipulator-X arm and gripper."""

    def __init__(self, arm_pub, gripper_ac):
        self._arm_pub = arm_pub
        self._gripper_ac = gripper_ac

    def send_joint(self, positions: list, duration: float = 2.0):
        msg = JointTrajectory()
        msg.joint_names = ["joint1", "joint2", "joint3", "joint4"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        msg.points = [pt]
        self._arm_pub.publish(msg)

    def gripper(self, position: float):
        goal = GripperCommandAction.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 10.0
        self._gripper_ac.send_goal_async(goal)

    def open_gripper(self):
        self.gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.gripper(GRIPPER_CLOSE)

    def home(self):
        self.send_joint(ARM_HOME, 3.0)


# =====================================================================
#  ROS 2 Robot Controller + State Machine
# =====================================================================
class RobotController(Node):
    """Main ROS 2 node: camera processing, state machine, motion + arm."""

    def __init__(self):
        super().__init__("color_pick_place")
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # -- vision --
        self.detector = ColorDetector()

        # -- shared state (read by GUI via self.lock) --
        self.state: str = ST_IDLE
        self.task_idx: int = 0
        self.status_text: str = "Initializing..."
        self.running: bool = False
        self.latest_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.detections: List[Detection] = []
        self.detected_color: str = ""
        self.detected_distance: str = "--"

        # -- timing state --
        self.backup_start = 0.0
        self.approach_zone_start = 0.0
        self.approach_cube_start = 0.0
        self.final_approach_start = 0.0
        self.pick_phase = 0
        self.pick_timer = 0.0
        self.drop_phase = 0
        self.drop_timer = 0.0

        # -- ROS 2 publishers / clients --
        cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        arm_pub = self.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10,
        )
        self.gripper_ac = ActionClient(
            self, GripperCommandAction, "/gripper_controller/gripper_cmd",
        )
        self.cam_sub = self.create_subscription(
            Image, CAMERA_TOPIC, self._cam_cb, 1,
        )
        self.set_state_cli = self.create_client(
            SetEntityState, "/set_entity_state",
        )

        # -- module instances --
        self.motion = MotionController(cmd_pub)
        self.arm = ArmController(arm_pub, self.gripper_ac)

        # -- wait for external services --
        self.get_logger().info("Waiting for gripper action server...")
        self.gripper_ac.wait_for_server(timeout_sec=10.0)
        self.get_logger().info("Waiting for /set_entity_state service...")
        self.set_state_cli.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("All services ready!")

        # -- state-machine timer (10 Hz) --
        self.sm_timer = self.create_timer(0.1, self._tick)
        self.status_text = "Ready -- press START"

    # -----------------------------------------------------------------
    #  Camera callback
    # -----------------------------------------------------------------
    def _cam_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        dets = self.detector.detect(frame)

        # Build HUD label
        tidx = self.task_idx
        label = TASKS[tidx][5] if tidx < len(TASKS) else "DONE"
        vis = self.detector.annotate(frame, dets, self.state, label)

        with self.lock:
            self.latest_frame = frame
            self.annotated_frame = vis
            self.detections = dets

    # -----------------------------------------------------------------
    #  Detection helper
    # -----------------------------------------------------------------
    def _find_target(
        self, color: str, obj_type: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Best detection of (color, obj_type).
        Returns (cx_norm, cy_norm, area_norm) or None."""
        with self.lock:
            dets = list(self.detections)

        best: Optional[Detection] = None
        best_area = 0.0
        for d in dets:
            if d.color == color and d.obj_type == obj_type and d.area_norm > best_area:
                best = d
                best_area = d.area_norm

        if best is None:
            return None

        # Update GUI-facing detection info
        self.detected_color = best.color
        self.detected_distance = self._area_to_distance(best.area_norm)
        return best.cx_norm, best.cy_norm, best.area_norm

    @staticmethod
    def _area_to_distance(area_norm: float) -> str:
        """Rough qualitative distance estimate from bounding-box area."""
        if area_norm > 0.05:
            return "Very Close"
        elif area_norm > 0.01:
            return "Close"
        elif area_norm > 0.003:
            return "Medium"
        elif area_norm > 0.001:
            return "Far"
        else:
            return "Very Far"

    # -----------------------------------------------------------------
    #  Gazebo teleport helper
    # -----------------------------------------------------------------
    def _teleport(self, model: str, x: float, y: float, z: float):
        req = SetEntityState.Request()
        st = EntityState()
        st.name = model
        st.pose.position = Point(x=float(x), y=float(y), z=float(z))
        st.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        st.twist = Twist()
        st.reference_frame = "world"
        req.state = st
        self.set_state_cli.call_async(req)

    # -----------------------------------------------------------------
    #  Mission start / stop
    # -----------------------------------------------------------------
    def start_mission(self):
        self.task_idx = 0
        self.state = ST_SCAN_CUBE
        self.running = True
        self.arm.home()
        self.arm.open_gripper()
        self.status_text = "Mission started -- searching for RED cube"
        self.get_logger().info(self.status_text)

    def stop_mission(self):
        self.motion.stop()
        self.running = False
        self.state = ST_IDLE
        self.arm.home()
        self.status_text = "Mission stopped by user"
        self.get_logger().info(self.status_text)

    # =================================================================
    #  State Machine  (called at 10 Hz)
    # =================================================================
    def _tick(self):
        if not self.running or self.state in (ST_DONE, ST_IDLE):
            return

        if self.task_idx >= len(TASKS):
            self.motion.stop()
            self.state = ST_DONE
            self.status_text = "ALL TASKS COMPLETE!"
            self.running = False
            return

        color_name, model, dx, dy, dz, label = TASKS[self.task_idx]

        # -- SCAN FOR CUBE (rotate left) ------------------------------
        if self.state == ST_SCAN_CUBE:
            self.status_text = (
                f"[{label}] Rotating -- scanning for {color_name} cube..."
            )
            target = self._find_target(color_name, "cube")
            if target:
                self.motion.stop()
                self.state = ST_APPROACH_CUBE
                self.approach_cube_start = time.time()
                self.status_text = (
                    f"[{label}] {color_name} cube detected! Approaching..."
                )
                self.get_logger().info(self.status_text)
            else:
                self.motion.publish(az=SCAN_ANG_VEL)

        # -- APPROACH CUBE --------------------------------------------
        elif self.state == ST_APPROACH_CUBE:
            target = self._find_target(color_name, "cube")
            if target is None:
                # Lost sight -- go back to scanning
                self.state = ST_SCAN_CUBE
                return

            cx, cy, area = target
            err = cx - 0.5
            elapsed = time.time() - self.approach_cube_start
            self.status_text = (
                f"[{label}] Approaching cube  area={area:.5f}  err={err:+.2f}"
            )

            if area >= CUBE_CLOSE_AREA or elapsed > MAX_APPROACH_TIME:
                self.motion.stop()
                self.state = ST_FINAL_CUBE
                self.final_approach_start = time.time()
                reason = (
                    "close enough"
                    if area >= CUBE_CLOSE_AREA
                    else f"timeout {elapsed:.0f}s"
                )
                self.status_text = f"[{label}] {reason} -- final approach..."
                self.get_logger().info(self.status_text)
            else:
                ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -APPROACH_KP * err))
                fwd = APPROACH_FWD * max(0.3, 1.0 - 2.0 * abs(err))
                self.motion.publish(lx=fwd, az=ang)

        # -- FINAL APPROACH CUBE (timed creep) ------------------------
        elif self.state == ST_FINAL_CUBE:
            elapsed = time.time() - self.final_approach_start
            if elapsed < FINAL_APPROACH_TIME:
                target = self._find_target(color_name, "cube")
                ang = 0.0
                if target:
                    err = target[0] - 0.5
                    ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -APPROACH_KP * err))
                self.motion.publish(lx=FINAL_APPROACH_VEL, az=ang)
                self.status_text = (
                    f"[{label}] Final approach... "
                    f"{FINAL_APPROACH_TIME - elapsed:.1f}s remaining"
                )
            else:
                self.motion.stop()
                self.state = ST_PICK
                self.pick_phase = 0
                self.pick_timer = time.time()
                self.status_text = f"[{label}] At cube -- picking up..."
                self.get_logger().info(self.status_text)

        # -- PICK SEQUENCE (phased, non-blocking) ---------------------
        elif self.state == ST_PICK:
            self._run_pick(model, label)

        # -- BACKUP ---------------------------------------------------
        elif self.state == ST_BACKUP:
            elapsed = time.time() - self.backup_start
            if elapsed < BACKUP_TIME:
                self.motion.publish(lx=BACKUP_VEL)
                self.status_text = (
                    f"[{label}] Backing up... {BACKUP_TIME - elapsed:.1f}s"
                )
            else:
                self.motion.stop()
                self.arm.home()
                self.state = ST_SCAN_ZONE
                self.status_text = (
                    f"[{label}] Scanning for {color_name} drop zone..."
                )
                self.get_logger().info(self.status_text)

        # -- SCAN FOR DROP ZONE (rotate right) ------------------------
        elif self.state == ST_SCAN_ZONE:
            self.status_text = (
                f"[{label}] Rotating -- scanning for {color_name} drop zone..."
            )
            target = self._find_target(color_name, "zone")
            if target:
                self.motion.stop()
                self.state = ST_APPROACH_ZONE
                self.approach_zone_start = time.time()
                self.status_text = (
                    f"[{label}] {color_name} drop zone found! Approaching..."
                )
                self.get_logger().info(self.status_text)
            else:
                self.motion.publish(az=-SCAN_ANG_VEL)

        # -- APPROACH DROP ZONE ---------------------------------------
        elif self.state == ST_APPROACH_ZONE:
            target = self._find_target(color_name, "zone")
            if target is None:
                self.state = ST_SCAN_ZONE
                return

            cx, cy, area = target
            err = cx - 0.5
            elapsed = time.time() - self.approach_zone_start
            self.status_text = (
                f"[{label}] Approaching zone  area={area:.5f}  "
                f"t={elapsed:.0f}s"
            )

            if area >= ZONE_CLOSE_AREA or elapsed > 30.0:
                self.motion.stop()
                self.state = ST_FINAL_ZONE
                self.final_approach_start = time.time()
                self.status_text = (
                    f"[{label}] Near drop zone -- final approach..."
                )
                self.get_logger().info(self.status_text)
            else:
                ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -APPROACH_KP * err))
                fwd = APPROACH_FWD * max(0.3, 1.0 - 2.0 * abs(err))
                self.motion.publish(lx=fwd, az=ang)

        # -- FINAL APPROACH ZONE --------------------------------------
        elif self.state == ST_FINAL_ZONE:
            elapsed = time.time() - self.final_approach_start
            if elapsed < 2.0:
                target = self._find_target(color_name, "zone")
                ang = 0.0
                if target:
                    err = target[0] - 0.5
                    ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -APPROACH_KP * err))
                self.motion.publish(lx=FINAL_APPROACH_VEL, az=ang)
                self.status_text = (
                    f"[{label}] Final approach to zone... "
                    f"{2.0 - elapsed:.1f}s"
                )
            else:
                self.motion.stop()
                self.state = ST_DROP
                self.drop_phase = 0
                self.drop_timer = time.time()
                self.status_text = f"[{label}] At drop zone -- dropping cube..."
                self.get_logger().info(self.status_text)

        # -- DROP SEQUENCE (phased) -----------------------------------
        elif self.state == ST_DROP:
            self._run_drop(model, dx, dy, dz, label)

        # -- NEXT COLOUR ----------------------------------------------
        elif self.state == ST_NEXT:
            self.task_idx += 1
            if self.task_idx < len(TASKS):
                nxt = TASKS[self.task_idx][5]
                self.arm.home()
                self.state = ST_SCAN_CUBE
                self.status_text = (
                    f"[{nxt}] Proceeding -- scanning for {nxt} cube..."
                )
                self.get_logger().info(self.status_text)
            else:
                self.motion.stop()
                self.state = ST_DONE
                self.status_text = "ALL TASKS COMPLETE!"
                self.running = False
                self.get_logger().info(self.status_text)

    # -----------------------------------------------------------------
    #  Pick sub-sequence
    # -----------------------------------------------------------------
    def _run_pick(self, model: str, label: str):
        t = time.time() - self.pick_timer

        if self.pick_phase == 0:
            self.status_text = f"[{label}] Opening gripper..."
            self.arm.open_gripper()
            self.pick_phase = 1
            self.pick_timer = time.time()
        elif self.pick_phase == 1 and t > 2.0:
            self.status_text = f"[{label}] Arm extending forward..."
            self.arm.send_joint(ARM_PRE_PICK, 3.0)
            self.pick_phase = 2
            self.pick_timer = time.time()
        elif self.pick_phase == 2 and t > 3.5:
            self.status_text = f"[{label}] Arm lowering to cube..."
            self.arm.send_joint(ARM_PICK, 3.0)
            self.pick_phase = 3
            self.pick_timer = time.time()
        elif self.pick_phase == 3 and t > 3.5:
            self.status_text = f"[{label}] Closing gripper on cube..."
            self.arm.close_gripper()
            self.pick_phase = 4
            self.pick_timer = time.time()
        elif self.pick_phase == 4 and t > 2.0:
            self.status_text = f"[{label}] Lifting cube..."
            self.arm.send_joint(ARM_LIFT, 3.0)
            self._teleport(model, 0.0, 0.0, -10.0)  # hide cube (simulated pick)
            self.pick_phase = 5
            self.pick_timer = time.time()
        elif self.pick_phase == 5 and t > 3.5:
            self.status_text = f"[{label}] Arm -> carry position..."
            self.arm.send_joint(ARM_CARRY, 3.0)
            self.pick_phase = 6
            self.pick_timer = time.time()
        elif self.pick_phase == 6 and t > 3.5:
            self.status_text = f"[{label}] Pick complete -- backing up..."
            self.get_logger().info(self.status_text)
            self.state = ST_BACKUP
            self.backup_start = time.time()

    # -----------------------------------------------------------------
    #  Drop sub-sequence
    # -----------------------------------------------------------------
    def _run_drop(
        self, model: str,
        dx: float, dy: float, dz: float, label: str,
    ):
        t = time.time() - self.drop_timer

        if self.drop_phase == 0:
            self.status_text = f"[{label}] Arm -> pre-drop..."
            self.arm.send_joint(ARM_PRE_DROP, 3.0)
            self.drop_phase = 1
            self.drop_timer = time.time()
        elif self.drop_phase == 1 and t > 3.5:
            self.status_text = f"[{label}] Arm -> drop position..."
            self.arm.send_joint(ARM_DROP, 3.0)
            self.drop_phase = 2
            self.drop_timer = time.time()
        elif self.drop_phase == 2 and t > 3.5:
            self.status_text = f"[{label}] Placing cube at drop zone..."
            self._teleport(model, dx, dy, dz)
            self.drop_phase = 3
            self.drop_timer = time.time()
        elif self.drop_phase == 3 and t > 0.5:
            self.status_text = f"[{label}] Opening gripper..."
            self.arm.open_gripper()
            self.drop_phase = 4
            self.drop_timer = time.time()
        elif self.drop_phase == 4 and t > 2.0:
            self.status_text = f"[{label}] Arm -> home..."
            self.arm.home()
            self.drop_phase = 5
            self.drop_timer = time.time()
        elif self.drop_phase == 5 and t > 2.5:
            self.status_text = (
                f"[{label}] Cube delivered to {label} drop zone!"
            )
            self.get_logger().info(self.status_text)
            self.state = ST_NEXT


# =====================================================================
#  PyQt5 GUI Module
# =====================================================================
class PickPlaceGUI(QMainWindow):
    """Mission-control GUI with live camera, status panel, and action log."""

    def __init__(self, controller: RobotController):
        super().__init__()
        self.ctrl = controller
        self.setWindowTitle(
            "OpenCV Pick & Place -- TurtleBot3 + OpenManipulator-X"
        )
        self.setMinimumSize(1000, 580)
        self.setStyleSheet("background-color: #0a0a1a;")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # == Left: camera view ========================================
        cam_frame = QFrame()
        cam_frame.setStyleSheet(
            "QFrame { background: #111; border: 2px solid #333;"
            " border-radius: 10px; }"
        )
        cam_layout = QVBoxLayout(cam_frame)

        cam_title = QLabel("Camera View (OpenCV Colour Detection)")
        cam_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        cam_title.setStyleSheet("color: #aaa; border: none;")
        cam_title.setAlignment(Qt.AlignCenter)
        cam_layout.addWidget(cam_title)

        self.cam_label = QLabel("Waiting for camera feed...")
        self.cam_label.setMinimumSize(640, 480)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet("color: #555; border: none;")
        cam_layout.addWidget(self.cam_label)

        root.addWidget(cam_frame, stretch=3)

        # == Right: status panel ======================================
        panel = QFrame()
        panel.setStyleSheet(
            "QFrame { background: #111; border: 2px solid #333;"
            " border-radius: 10px; }"
        )
        right = QVBoxLayout(panel)
        right.setSpacing(10)

        title = QLabel("Mission Control")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #e94560; border: none;")
        title.setAlignment(Qt.AlignCenter)
        right.addWidget(title)

        # -- state label --
        self.state_label = QLabel("State: IDLE")
        self.state_label.setFont(QFont("Consolas", 13, QFont.Bold))
        self.state_label.setWordWrap(True)
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setStyleSheet(
            "padding: 12px; background: #16213e; color: #f39c12;"
            "border-radius: 8px; border: 1px solid #1a1a2e;"
        )
        right.addWidget(self.state_label)

        # -- detected object info --
        self.detect_label = QLabel("Detected: --  |  Distance: --")
        self.detect_label.setFont(QFont("Segoe UI", 10))
        self.detect_label.setAlignment(Qt.AlignCenter)
        self.detect_label.setStyleSheet(
            "color: #bbb; border: none; padding: 4px;"
        )
        right.addWidget(self.detect_label)

        # -- current task --
        self.task_label = QLabel("Task: --")
        self.task_label.setFont(QFont("Segoe UI", 11))
        self.task_label.setStyleSheet(
            "color: #ccc; border: none; padding: 4px;"
        )
        self.task_label.setAlignment(Qt.AlignCenter)
        right.addWidget(self.task_label)

        # -- progress --
        self.progress_label = QLabel("Progress: 0 / 3")
        self.progress_label.setFont(QFont("Segoe UI", 11))
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #7ec8e3; border: none;")
        right.addWidget(self.progress_label)

        # -- colour indicators --
        ind_layout = QHBoxLayout()
        self.indicators = {}
        for c, hex_col in [
            ("RED", "#e74c3c"), ("BLUE", "#3498db"), ("GREEN", "#2ecc71"),
        ]:
            lbl = QLabel(f"  {c}  ")
            lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"background: #222; color: {hex_col}; border: 2px solid #333;"
                f"border-radius: 6px; padding: 6px;"
            )
            ind_layout.addWidget(lbl)
            self.indicators[c] = lbl
        right.addLayout(ind_layout)

        # -- log panel --
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setStyleSheet(
            "background: #0f3460; color: #e2e2e2; border-radius: 8px;"
            "padding: 6px; border: 1px solid #1a1a2e;"
        )
        right.addWidget(self.log, stretch=1)

        # -- Start / Stop buttons --
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("START MISSION")
        self.start_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet(
            "QPushButton { background: #e94560; color: white;"
            " padding: 12px; border-radius: 10px; border: none; }"
            "QPushButton:hover { background: #c0392b; }"
            "QPushButton:disabled { background: #555; color: #999; }"
        )
        self.start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background: #555; color: #999;"
            " padding: 12px; border-radius: 10px; border: none; }"
            "QPushButton:hover { background: #7f8c8d; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.stop_btn)

        right.addLayout(btn_layout)

        root.addWidget(panel, stretch=1)

        # -- refresh timer (30 fps) --
        self._last_status = ""
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(33)

    # -----------------------------------------------------------------
    def _on_start(self):
        if not self.ctrl.running:
            self.ctrl.start_mission()
            self.start_btn.setText("RUNNING...")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.stop_btn.setStyleSheet(
                "QPushButton { background: #e67e22; color: white;"
                " padding: 12px; border-radius: 10px; border: none; }"
                "QPushButton:hover { background: #d35400; }"
            )
            self.log.append("Mission started: RED -> BLUE -> GREEN")

    def _on_stop(self):
        if self.ctrl.running:
            self.ctrl.stop_mission()
            self.start_btn.setText("START MISSION")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet(
                "QPushButton { background: #333; color: #666;"
                " padding: 12px; border-radius: 10px; border: none; }"
            )
            self.log.append("Mission STOPPED by user")

    # -----------------------------------------------------------------
    def _update(self):
        # -- camera feed --
        with self.ctrl.lock:
            frame = self.ctrl.annotated_frame

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.cam_label.width(), self.cam_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.cam_label.setPixmap(pix)

        # -- state --
        state = self.ctrl.state
        status = self.ctrl.status_text
        tidx = self.ctrl.task_idx

        state_colors = {
            ST_IDLE: "#95a5a6",
            ST_SCAN_CUBE: "#f39c12", ST_APPROACH_CUBE: "#e67e22",
            ST_FINAL_CUBE: "#d35400",
            ST_PICK: "#e74c3c", ST_BACKUP: "#9b59b6",
            ST_SCAN_ZONE: "#3498db", ST_APPROACH_ZONE: "#2980b9",
            ST_FINAL_ZONE: "#2471a3",
            ST_DROP: "#27ae60", ST_NEXT: "#1abc9c", ST_DONE: "#2ecc71",
        }
        sc = state_colors.get(state, "#ecf0f1")

        nice = state.replace("_", " ")
        self.state_label.setText(nice)
        self.state_label.setStyleSheet(
            f"padding: 12px; background: #16213e; color: {sc};"
            f"border-radius: 8px; font-size: 14px; font-weight: bold;"
            f"border: 1px solid {sc};"
        )

        # -- detected object info --
        det_color = self.ctrl.detected_color or "--"
        det_dist = self.ctrl.detected_distance or "--"
        self.detect_label.setText(
            f"Detected: {det_color.upper()}  |  Distance: {det_dist}"
        )

        # -- task / progress --
        completed = min(tidx, len(TASKS))
        if tidx < len(TASKS):
            clr = TASKS[tidx][5]
            self.task_label.setText(
                f"Current: {clr} cube -> {clr} drop zone"
            )
        else:
            self.task_label.setText("All cubes delivered!")

        dots = "●" * completed + "○" * (len(TASKS) - completed)
        self.progress_label.setText(f"Progress: {completed}/{len(TASKS)}   {dots}")

        # -- colour indicators --
        for i, c in enumerate(["RED", "BLUE", "GREEN"]):
            lbl = self.indicators[c]
            if i < completed:
                lbl.setStyleSheet(
                    "background: #1a3a1a; color: #2ecc71;"
                    "border: 2px solid #2ecc71;"
                    "border-radius: 6px; padding: 6px;"
                )
            elif i == tidx and state != ST_DONE:
                hex_c = {"RED": "#e74c3c", "BLUE": "#3498db", "GREEN": "#2ecc71"}[c]
                lbl.setStyleSheet(
                    f"background: #1a1a3e; color: {hex_c};"
                    f"border: 2px solid {hex_c};"
                    f"border-radius: 6px; padding: 6px; font-weight: bold;"
                )
            else:
                hex_c = {"RED": "#e74c3c", "BLUE": "#3498db", "GREEN": "#2ecc71"}[c]
                lbl.setStyleSheet(
                    f"background: #222; color: {hex_c};"
                    f"border: 2px solid #333;"
                    f"border-radius: 6px; padding: 6px;"
                )

        # -- log --
        if status != self._last_status:
            self._last_status = status
            self.log.append(status)
            sb = self.log.verticalScrollBar()
            sb.setValue(sb.maximum())

        # -- mission done --
        if state == ST_DONE:
            self.start_btn.setText("MISSION COMPLETE")
            self.start_btn.setStyleSheet(
                "QPushButton { background: #27ae60; color: white;"
                " padding: 12px; border-radius: 10px; border: none; }"
            )
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet(
                "QPushButton { background: #333; color: #666;"
                " padding: 12px; border-radius: 10px; border: none; }"
            )

    # -----------------------------------------------------------------
    def closeEvent(self, event):
        self.ctrl.motion.stop()
        self.ctrl.running = False
        event.accept()


# =====================================================================
#  Main
# =====================================================================
def main():
    rclpy.init()
    controller = RobotController()

    # Spin ROS 2 in a background thread so the GUI stays responsive
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(controller,), daemon=True,
    )
    spin_thread.start()

    # PyQt5 on main thread
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = PickPlaceGUI(controller)
    gui.show()

    exit_code = app.exec_()

    controller.motion.stop()
    controller.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
