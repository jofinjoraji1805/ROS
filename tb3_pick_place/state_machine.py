#!/usr/bin/env python3
"""
state_machine.py -- Autonomous pick-and-place with map-based navigation.

Two-phase navigation per cube:
  1. NAV_TO_CUBE: A* path planning to approach point near the table
  2. YOLO visual servo: SEARCH -> ALIGN -> L-ALIGN -> APPROACH -> ADJUST -> PICK

Then for drop:
  1. NAV_TO_ZONE: A* path planning to approach point near drop zone
  2. YOLO visual servo: SEARCH_DROP -> APPROACH_DROP -> ADJUST -> PLACE

LIDAR obstacle avoidance is active in all forward-driving states.
"""

import math
import time
from typing import Callable, List, Optional, Tuple

from .config import (
    ST_IDLE, ST_NAV_TO_CUBE, ST_DRIVE_TO_CUBE, ST_SEARCH_OBJECT, ST_ALIGN_OBJECT,
    ST_LATERAL_ALIGN, ST_APPROACH_OBJECT, ST_ADJUST_POSITION, ST_ALIGN_TABLE, ST_PICK_OBJECT,
    ST_BACKUP_PICK, ST_NAV_TO_ZONE, ST_DRIVE_TO_ZONE, ST_SEARCH_DROP_ZONE,
    ST_APPROACH_DROP, ST_ADJUST_DROP, ST_PLACE_OBJECT,
    ST_BACKUP_DROP, ST_NEXT_OBJECT, ST_RETURN_HOME,
    ST_SEARCH_DOCK, ST_ALIGN_DOCK, ST_APPROACH_DOCK,
    ST_DOCK_CREEP, ST_PARKED, ST_DONE,
    SCAN_ANG_VEL, ALIGN_CENTER_PX, ALIGN_KP, ALIGN_LOST_MAX,
    MAX_ANG_VEL, APPROACH_FWD, APPROACH_DRIFT_PX,
    APPROACH_MILD_KP, APPROACH_MILD_MAX, APPROACH_TIMEOUT,
    CUBE_LOST_TICKS, ADJUST_FWD, ADJUST_DURATION,
    LATERAL_MIN_ANGLE, LATERAL_DRIVE_VEL, LATERAL_YAW_TOLERANCE,
    BACKUP_VEL, BACKUP_PICK_TIME, BACKUP_DROP_TIME,
    ZONE_CLOSE_AREA, ZONE_APPROACH_TIME, ZONE_FINAL_TIME,
    ZONE_FINAL_VEL, ZONE_ADJUST_TIME, RETURN_HOME_THRESH,
    TABLE_STOP_DISTANCE, OBSTACLE_SLOW_DISTANCE, OBSTACLE_SLOW_FACTOR,
    CUBE_NAV_TARGETS, ZONE_NAV_TARGETS, ZONE_BOX_CENTER,
    CUBE_FACE_YAW, ZONE_FACE_YAW, FACE_YAW_TOLERANCE,
    ALIGN_TABLE_KP, ALIGN_TABLE_SLOPE_THRESH, ALIGN_TABLE_TIMEOUT,
    CUBE_TABLE_Y,
    DOCK_POSITION, DOCK_FACE_YAW,
    DOCK_APPROACH_VEL, DOCK_CENTER, DOCK_STOP_ODOM_DIST,
    DOCK_CREEP_VEL, DOCK_CREEP_TIME,
    DOCK_ALIGN_CENTER_PX, DOCK_APPROACH_TIMEOUT,
)
from .perception import PerceptionModule, TaskDef
from .motion import MotionController
from .arm_control import ArmController
from .lidar import LidarProcessor
from .navigator import Navigator


class StateMachine:
    """
    Pick-and-place FSM with map-based A* navigation.
    """

    def __init__(
        self,
        perception: PerceptionModule,
        motion: MotionController,
        arm: ArmController,
        lidar: LidarProcessor,
        navigator: Navigator,
        find_target_fn: Callable[[int], Optional[Tuple[float, float, float]]],
        get_odom_fn: Callable[[], Tuple[float, float, float]],
        log_fn: Callable[[str], None],
        robot_node=None,
    ):
        self.perception = perception
        self.motion = motion
        self.arm = arm
        self.lidar = lidar
        self.navigator = navigator
        self._find = find_target_fn
        self._get_odom = get_odom_fn
        self._log = log_fn
        self._robot = robot_node

        self.tasks: List[TaskDef] = perception.build_tasks()
        self.task_idx = 0
        self.state = ST_IDLE
        self.status_text = "Ready -- press START MISSION"
        self.running = False

        self.detected_label = "--"
        self.detected_distance = "--"

        self._phase_start = 0.0
        self._lost_count = 0
        self._max_area = 0.0
        self._pick_phase = 0
        self._pick_timer = 0.0
        self._drop_phase = 0
        self._drop_timer = 0.0
        self._zone_in_final = False
        self._zone_final_start = 0.0

        self._home_x = 0.0
        self._home_y = 0.0
        self._home_yaw = 0.0
        self._facing = False
        self._face_target = 0.0
        self._perp_yaw = 0.0          # saved perpendicular heading for approach
        self._align_phase = 0         # 0=parallel, 1=center on cube
        self._last_align_ang = 0.0    # last rotation direction during align
        self._lateral_phase = 0       # 0=rotate lateral, 1=drive, 2=rotate back
        self._lateral_target_yaw = 0.0
        self._lateral_drive_dist = 0.0
        self._lateral_start_x = 0.0
        self._lateral_start_y = 0.0
        self._lateral_done = False    # True after L-move; approach uses heading-hold

    # ── Face target yaw ────────────────────────────────────────────

    def _face_yaw_tick(self, lbl: str) -> bool:
        """Rotate to face self._face_target. Returns True when aligned."""
        _, _, yaw = self._get_odom()
        err = self._face_target - yaw
        while err > math.pi:  err -= 2 * math.pi
        while err < -math.pi: err += 2 * math.pi
        if abs(err) < FACE_YAW_TOLERANCE:
            self.motion.stop()
            self._facing = False
            return True
        ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.5 * err))
        if 0 < abs(ang) < 0.05:
            ang = 0.05 * (1 if ang > 0 else -1)
        self.motion.publish(lx=0.0, az=ang)
        self.status_text = f"[{lbl}] Turning to face target... {math.degrees(err):+.1f}\u00b0"
        return False

    # ── LIDAR-safe speed ─────────────────────────────────────────────

    def _safe_fwd(self, desired: float) -> float:
        d = self.lidar.get_front_distance()
        if d < TABLE_STOP_DISTANCE:
            return 0.0
        if d < OBSTACLE_SLOW_DISTANCE:
            factor = (d - TABLE_STOP_DISTANCE) / (
                OBSTACLE_SLOW_DISTANCE - TABLE_STOP_DISTANCE)
            return desired * max(OBSTACLE_SLOW_FACTOR, factor)
        return desired

    # ── Mission control ──────────────────────────────────────────────

    def start_mission(self):
        x, y, yaw = self._get_odom()
        self._home_x, self._home_y, self._home_yaw = x, y, yaw
        self._log(f"Home: ({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.1f})")

        self.task_idx = 0
        self.running = True
        self.arm.home()
        self.arm.open_gripper()

        # Start with navigation to first cube
        task = self.tasks[0]
        self._start_nav_to_cube(task)

    def stop_mission(self):
        self.motion.stop()
        self.navigator.cancel()
        self.running = False
        self.state = ST_IDLE
        self.arm.home()
        self.status_text = "Mission stopped by user"
        self._log(self.status_text)

    def _start_nav_to_cube(self, task: TaskDef):
        lbl = task.label
        self._lateral_done = False
        waypoints = CUBE_NAV_TARGETS.get(lbl)
        if waypoints:
            self._lpath_wps = list(waypoints)  # L-path waypoints
            self._lpath_idx = 0
            self.state = ST_DRIVE_TO_CUBE
            self._phase_start = time.time()
            self.status_text = f"[{lbl}] L-path to {lbl} cube..."
            self._log(self.status_text)
        else:
            self.state = ST_SEARCH_OBJECT
            self._lost_count = 0
            self._max_area = 0.0

    def _start_nav_to_zone(self, task: TaskDef):
        lbl = task.label
        waypoints = ZONE_NAV_TARGETS.get(lbl)
        if waypoints:
            self._zone_wps = list(waypoints)
            self._zone_wp_idx = 0
            self.state = ST_DRIVE_TO_ZONE
            self._phase_start = time.time()
            self.status_text = f"[{lbl}] L-path to {lbl} drop zone..."
            self._log(self.status_text)
        else:
            self.state = ST_SEARCH_DROP_ZONE
            self.status_text = f"[{lbl}] Scanning for {lbl} drop zone..."

    # ── Find + label ─────────────────────────────────────────────────

    def _find_and_label(self, target_cls: int):
        result = self._find(target_cls)
        if result:
            self.detected_label = self.perception.names[target_cls]
            self.detected_distance = PerceptionModule.area_to_distance(result[2])
        return result

    # ── Main tick ────────────────────────────────────────────────────

    def tick(self):
        if not self.running or self.state in (ST_DONE, ST_PARKED, ST_IDLE):
            return

        _dock_states = (ST_RETURN_HOME, ST_SEARCH_DOCK, ST_ALIGN_DOCK,
                        ST_APPROACH_DOCK, ST_DOCK_CREEP)
        if self.task_idx >= len(self.tasks) and self.state not in _dock_states:
            self.motion.stop()
            self.arm.home()
            self.state = ST_RETURN_HOME
            self.status_text = "All cubes delivered -- returning home..."
            self._log(self.status_text)
            # Navigate to dock search area
            if self.navigator.map_loaded:
                self.navigator.navigate_to(DOCK_POSITION[0], DOCK_POSITION[1])
            return

        if self.state == ST_RETURN_HOME:
            self._run_return_home()
            return

        if self.state == ST_SEARCH_DOCK:
            self._search_dock()
            return

        if self.state == ST_ALIGN_DOCK:
            self._align_dock()
            return

        if self.state == ST_APPROACH_DOCK:
            self._approach_dock()
            return

        if self.state == ST_DOCK_CREEP:
            self._dock_creep()
            return

        task = self.tasks[self.task_idx]
        lbl = task.label

        handler = {
            ST_NAV_TO_CUBE: self._nav_to_cube,
            ST_DRIVE_TO_CUBE: self._drive_to_cube,
            ST_SEARCH_OBJECT: self._search_object,
            ST_ALIGN_OBJECT: self._align_object,
            ST_LATERAL_ALIGN: self._lateral_align,
            ST_APPROACH_OBJECT: self._approach_object,
            ST_ADJUST_POSITION: self._adjust_position,
            ST_ALIGN_TABLE: self._align_table,
            ST_PICK_OBJECT: self._pick_object,
            ST_BACKUP_PICK: self._backup_pick,
            ST_NAV_TO_ZONE: self._nav_to_zone,
            ST_DRIVE_TO_ZONE: self._drive_to_zone,
            ST_SEARCH_DROP_ZONE: self._search_drop_zone,
            ST_APPROACH_DROP: self._approach_drop,
            ST_ADJUST_DROP: self._adjust_drop,
            ST_PLACE_OBJECT: self._place_object,
            ST_BACKUP_DROP: self._backup_drop,
            ST_NEXT_OBJECT: self._next_object,
        }.get(self.state)

        if handler:
            handler(task, lbl)


    # ── NAV_TO_CUBE: A* navigation to cube approach point ────────────

    def _nav_to_cube(self, task: TaskDef, lbl: str):
        """Face the table after L-path drive is complete, then search."""
        if self._facing:
            if self._face_yaw_tick(lbl):
                self.state = ST_SEARCH_OBJECT
                self._lost_count = 0
                self._max_area = 0.0
                self._log(f"[{lbl}] Facing table -- searching for cube...")
            return
        # Shouldn't reach here, but just in case
        self._facing = True
        self._face_target = CUBE_FACE_YAW

    # ── NAV_TO_ZONE: face basket then go to place ─────────────────────

    def _nav_to_zone(self, task: TaskDef, lbl: str):
        """After L-path drive completes, face +X and go straight to PLACE."""
        if self._facing:
            if self._face_yaw_tick(lbl):
                self.motion.stop()
                self.state = ST_PLACE_OBJECT
                self._drop_phase = 0
                self._drop_timer = time.time()
                self._log(f"[{lbl}] Facing basket -- dropping cube...")
            return
        self._facing = True
        self._face_target = ZONE_FACE_YAW

    # ── DRIVE_TO_ZONE: odom L-path to drop zone ────────────────────────

    def _drive_to_zone(self, task: TaskDef, lbl: str):
        """L-path drive to drop zone: follow waypoints using odometry."""
        if not hasattr(self, '_zone_wps') or self._zone_wp_idx >= len(self._zone_wps):
            # All waypoints done -- face basket and place
            self.motion.stop()
            self._facing = True
            self._face_target = ZONE_FACE_YAW
            self.state = ST_NAV_TO_ZONE
            self._log(f"[{lbl}] At drop zone -- facing basket...")
            return

        tx, ty = self._zone_wps[self._zone_wp_idx]
        x, y, yaw = self._get_odom()
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.15:
            self._zone_wp_idx += 1
            self._log(f"[{lbl}] Zone waypoint {self._zone_wp_idx}/{len(self._zone_wps)} reached")
            return

        elapsed = time.time() - self._phase_start
        if elapsed > 60.0:
            self.motion.stop()
            self._facing = True
            self._face_target = ZONE_FACE_YAW
            self.state = ST_NAV_TO_ZONE
            self._log(f"[{lbl}] Zone drive timeout -- facing basket...")
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err = target_yaw - yaw
        while yaw_err > math.pi:  yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        if abs(yaw_err) > 0.15:
            ang = max(-0.4, min(0.4, 1.5 * yaw_err))
            self.motion.publish(lx=0.0, az=ang)
        else:
            ang = max(-0.3, min(0.3, 1.0 * yaw_err))
            speed = min(0.15, 0.08 + 0.05 * dist)
            self.motion.publish(lx=speed, az=ang)
        self.status_text = f"[{lbl}] -> zone wp{self._zone_wp_idx+1} d={dist:.2f}m"

    # ── DRIVE_TO_CUBE: simple odom drive to approach point (no map) ────

    def _drive_to_cube(self, task: TaskDef, lbl: str):
        """L-path drive: follow waypoints sequentially using odometry.
        After all waypoints, face the table and start YOLO search."""
        if not hasattr(self, '_lpath_wps') or self._lpath_idx >= len(self._lpath_wps):
            # All waypoints done -- face table and search
            self.motion.stop()
            self._facing = True
            self._face_target = CUBE_FACE_YAW
            self.state = ST_NAV_TO_CUBE
            self._log(f"[{lbl}] At approach point -- facing table...")
            return

        tx, ty = self._lpath_wps[self._lpath_idx]
        x, y, yaw = self._get_odom()
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx * dx + dy * dy)

        # Reached this waypoint?
        if dist < 0.15:
            self._lpath_idx += 1
            self._log(f"[{lbl}] L-path waypoint {self._lpath_idx}/{len(self._lpath_wps)} reached")
            return

        # Timeout safety
        elapsed = time.time() - self._phase_start
        if elapsed > 60.0:
            self.motion.stop()
            self._facing = True
            self._face_target = CUBE_FACE_YAW
            self.state = ST_NAV_TO_CUBE
            self._log(f"[{lbl}] Drive timeout -- facing table...")
            return

        # Drive toward waypoint: rotate first, then drive
        target_yaw = math.atan2(dy, dx)
        yaw_err = target_yaw - yaw
        while yaw_err > math.pi:  yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        if abs(yaw_err) > 0.15:
            # Rotate in place first
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.6 * yaw_err))
            self.motion.publish(lx=0.0, az=ang)
        else:
            # Drive with heading correction
            fwd = self._safe_fwd(min(0.12, 0.5 * dist))
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.4 * yaw_err))
            self.motion.publish(lx=fwd, az=ang)
        wp = self._lpath_idx + 1
        total = len(self._lpath_wps)
        self.status_text = f"[{lbl}] L-path WP{wp}/{total} d={dist:.2f}m"

    # ── SEARCH / ALIGN / APPROACH / ADJUST (YOLO visual servo) ──────

    def _search_object(self, task: TaskDef, lbl: str):
        self.status_text = f"[{lbl}] Scanning for {self.perception.names[task.cube_cls]}..."
        target = self._find_and_label(task.cube_cls)
        if target:
            self.motion.stop()
            self.state = ST_ALIGN_OBJECT
            self._phase_start = time.time()
            self._lost_count = 0
            self._max_area = 0.0
            self.status_text = f"[{lbl}] Cube detected! Aligning..."
            self._log(self.status_text)
        else:
            self.motion.publish(az=SCAN_ANG_VEL)

    def _align_object(self, task: TaskDef, lbl: str):
        target = self._find_and_label(task.cube_cls)
        if target is None:
            self._lost_count += 1
            if self._lost_count > ALIGN_LOST_MAX:
                if self._max_area > 0.002:
                    # Close enough -- just drive forward
                    _, _, yaw = self._get_odom()
                    self._perp_yaw = yaw
                    self.state = ST_APPROACH_OBJECT
                    self._phase_start = time.time()
                    self._lost_count = 0
                    self.status_text = f"[{lbl}] Lost but close -- driving..."
                    self._log(self.status_text)
                else:
                    self.state = ST_SEARCH_OBJECT
                    self._lost_count = 0
            else:
                # Keep rotating slowly in the last direction instead of stopping
                if self._last_align_ang != 0.0:
                    drift_ang = 0.03 * (1 if self._last_align_ang > 0 else -1)
                    self.motion.publish(lx=0.0, az=drift_ang)
                else:
                    self.motion.stop()
            self.status_text = f"[{lbl}] Cube lost ({self._lost_count})"
            return

        self._lost_count = 0
        cx, cy, area = target
        self._max_area = max(self._max_area, area)
        err = cx - 0.5
        pixel_err = err * 640

        if abs(pixel_err) <= ALIGN_CENTER_PX:
            self.motion.stop()
            _, _, yaw = self._get_odom()
            self._perp_yaw = yaw
            self._lost_count = 0
            self._last_align_ang = 0.0

            # Check if robot heading deviated from table-perpendicular
            yaw_diff = yaw - CUBE_FACE_YAW
            while yaw_diff > math.pi:  yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi: yaw_diff += 2 * math.pi

            if abs(yaw_diff) > LATERAL_MIN_ANGLE:
                # L-move needed: strafe laterally, then approach straight
                front_d = self.lidar.get_front_distance()
                self._lateral_drive_dist = abs(front_d * math.sin(yaw_diff))
                # Rotate toward the cube side (perpendicular to table)
                if yaw_diff > 0:
                    self._lateral_target_yaw = CUBE_FACE_YAW + math.pi / 2
                else:
                    self._lateral_target_yaw = CUBE_FACE_YAW - math.pi / 2
                # Normalize
                while self._lateral_target_yaw > math.pi:
                    self._lateral_target_yaw -= 2 * math.pi
                while self._lateral_target_yaw < -math.pi:
                    self._lateral_target_yaw += 2 * math.pi
                self._lateral_phase = 0
                self.state = ST_LATERAL_ALIGN
                self._phase_start = time.time()
                self._log(f"[{lbl}] L-move: lateral {self._lateral_drive_dist:.2f}m (yaw_diff={math.degrees(yaw_diff):+.1f}deg)")
            else:
                self.state = ST_APPROACH_OBJECT
                self._phase_start = time.time()
                self.status_text = f"[{lbl}] Cube centered! Visual servo approach..."
                self._log(self.status_text)
        else:
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_KP * err))
            if 0 < abs(ang) < 0.05:
                ang = 0.05 * (1 if ang > 0 else -1)
            self._last_align_ang = ang
            self.motion.publish(lx=0.0, az=ang)
            self.status_text = f"[{lbl}] Aligning err={pixel_err:+.0f}px"

    # ── LATERAL L-ALIGN: strafe to line up with off-center cube ────────

    def _lateral_align(self, task: TaskDef, lbl: str):
        """L-shaped maneuver: rotate lateral -> drive to align -> rotate back to face table."""
        _, _, cur_yaw = self._get_odom()

        # Phase 0: Rotate to face lateral direction
        if self._lateral_phase == 0:
            err = self._lateral_target_yaw - cur_yaw
            while err > math.pi:  err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi
            if abs(err) < LATERAL_YAW_TOLERANCE:
                self.motion.stop()
                x, y, _ = self._get_odom()
                self._lateral_start_x = x
                self._lateral_start_y = y
                self._lateral_phase = 1
                self._log(f"[{lbl}] L-move: driving lateral {self._lateral_drive_dist:.2f}m")
                return
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.5 * err))
            if 0 < abs(ang) < 0.05:
                ang = 0.05 * (1 if ang > 0 else -1)
            self.motion.publish(lx=0.0, az=ang)
            self.status_text = f"[{lbl}] L-move: rotating lateral {math.degrees(err):+.1f}deg"

        # Phase 1: Drive forward the lateral distance
        elif self._lateral_phase == 1:
            x, y, _ = self._get_odom()
            dx = x - self._lateral_start_x
            dy = y - self._lateral_start_y
            driven = math.sqrt(dx * dx + dy * dy)
            if driven >= self._lateral_drive_dist:
                self.motion.stop()
                self._lateral_phase = 2
                self._log(f"[{lbl}] L-move: rotating to face table")
                return
            # Heading hold while driving
            err = self._lateral_target_yaw - cur_yaw
            while err > math.pi:  err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi
            ang = max(-0.10, min(0.10, 0.5 * err))
            self.motion.publish(lx=LATERAL_DRIVE_VEL, az=ang)
            self.status_text = f"[{lbl}] L-move: lateral {driven:.2f}/{self._lateral_drive_dist:.2f}m"

        # Phase 2: Rotate back to face table (CUBE_FACE_YAW)
        elif self._lateral_phase == 2:
            err = CUBE_FACE_YAW - cur_yaw
            while err > math.pi:  err -= 2 * math.pi
            while err < -math.pi: err += 2 * math.pi
            if abs(err) < LATERAL_YAW_TOLERANCE:
                self.motion.stop()
                self._perp_yaw = CUBE_FACE_YAW
                self._lateral_done = True
                self.state = ST_APPROACH_OBJECT
                self._phase_start = time.time()
                self._lost_count = 0
                self._log(f"[{lbl}] L-move done! Approaching table straight...")
                return
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.5 * err))
            if 0 < abs(ang) < 0.05:
                ang = 0.05 * (1 if ang > 0 else -1)
            self.motion.publish(lx=0.0, az=ang)
            self.status_text = f"[{lbl}] L-move: facing table {math.degrees(err):+.1f}deg"

    def _approach_object(self, task: TaskDef, lbl: str):
        """Drive toward cube using continuous YOLO visual servoing.
        Keeps the cube centered in the camera frame while driving forward,
        so the robot converges directly in front of the cube."""
        elapsed = time.time() - self._phase_start
        front_d = self.lidar.get_front_distance()

        # Stop when LIDAR detects table -- align
        if front_d < TABLE_STOP_DISTANCE:
            self.motion.stop()
            time.sleep(0.3)
            self.state = ST_ALIGN_TABLE
            self._phase_start = time.time()
            self._align_phase = 0
            self.status_text = f"[{lbl}] At table ({front_d:.2f}m) -- aligning..."
            self._log(self.status_text)
            return

        if elapsed > APPROACH_TIMEOUT:
            self.motion.stop()
            self.state = ST_ADJUST_POSITION
            self._phase_start = time.time()
            return

        # After L-move: heading-hold with gentle YOLO trim (don't re-steer)
        # Without L-move: full YOLO visual servo
        target = self._find_and_label(task.cube_cls)
        if self._lateral_done:
            # Heading-hold on CUBE_FACE_YAW — drive straight to table
            _, _, cur_yaw = self._get_odom()
            yaw_err = self._perp_yaw - cur_yaw
            while yaw_err > math.pi:  yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            ang = max(-0.10, min(0.10, 0.8 * yaw_err))
            # Very gentle YOLO trim (±0.02) — just nudge, don't steer
            if target is not None:
                self._lost_count = 0
                trim = max(-0.02, min(0.02, -0.05 * (target[0] - 0.5)))
                ang = max(-0.10, min(0.10, ang + trim))
                pixel_err = (target[0] - 0.5) * 640
            else:
                self._lost_count += 1
                pixel_err = 0
                if self._lost_count > CUBE_LOST_TICKS:
                    self.motion.stop()
                    self.state = ST_SEARCH_OBJECT
                    self._lost_count = 0
                    return
        elif target is not None:
            self._lost_count = 0
            cx = target[0]
            pixel_err = (cx - 0.5) * 640
            # Proportional steering to center cube in camera
            ang = max(-0.15, min(0.15, -APPROACH_MILD_KP * (cx - 0.5)))
            # Update saved heading for fallback
            _, _, yaw = self._get_odom()
            self._perp_yaw = yaw
        else:
            self._lost_count += 1
            pixel_err = 0
            # Fallback: hold last known good heading
            _, _, cur_yaw = self._get_odom()
            yaw_err = self._perp_yaw - cur_yaw
            while yaw_err > math.pi:  yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            ang = max(-0.10, min(0.10, 0.8 * yaw_err))
            if self._lost_count > CUBE_LOST_TICKS:
                self.motion.stop()
                self.state = ST_SEARCH_OBJECT
                self._lost_count = 0
                return

        fwd = self._safe_fwd(APPROACH_FWD)
        self.motion.publish(lx=fwd, az=ang)
        self.status_text = f"[{lbl}] Approach err={pixel_err:+.0f}px d={front_d:.2f}m"

    def _adjust_position(self, task: TaskDef, lbl: str):
        elapsed = time.time() - self._phase_start
        front_d = self.lidar.get_front_distance()

        if front_d < TABLE_STOP_DISTANCE:
            self.motion.stop()
            time.sleep(0.3)
            self.state = ST_ALIGN_TABLE
            self._phase_start = time.time()
            self._align_phase = 0
            self.status_text = f"[{lbl}] At table ({front_d:.2f}m) -- aligning..."
            self._log(self.status_text)
            return

        if elapsed > ADJUST_DURATION:
            self.motion.stop()
            time.sleep(0.3)
            self.state = ST_ALIGN_TABLE
            self._phase_start = time.time()
            self._align_phase = 0
            self.status_text = f"[{lbl}] Adjust timeout -- aligning..."
            self._log(self.status_text)
            return

        # Heading-hold: drive straight on saved heading
        _, _, cur_yaw = self._get_odom()
        yaw_err = self._perp_yaw - cur_yaw
        while yaw_err > math.pi:  yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi
        ang = max(-0.10, min(0.10, 0.8 * yaw_err))
        self.motion.publish(lx=ADJUST_FWD, az=ang)
        self.status_text = f"[{lbl}] Straight creep d={front_d:.2f}m {ADJUST_DURATION - elapsed:.1f}s"

    # ── ALIGN_TABLE: LiDAR parallel + YOLO cube center + final creep ───

    def _align_table(self, task: TaskDef, lbl: str):
        """Three-phase alignment at table (lateral centering handled by odom approach):
        Phase 0: Rotate until perpendicular to table (LiDAR slope ≈ 0)
        Phase 1: Rotate until cube is centered in camera (YOLO)
        Phase 2: Final creep forward to pick distance while keeping cube centered
        """
        elapsed = time.time() - self._phase_start

        if elapsed > ALIGN_TABLE_TIMEOUT:
            self.motion.stop()
            self._log(f"[{lbl}] Align timeout -- picking")
            self._align_phase = 0
            self.state = ST_PICK_OBJECT
            self._pick_phase = 0
            self._pick_timer = time.time()
            return

        # ── Phase 0: LiDAR parallel alignment ──
        if self._align_phase == 0:
            result = self.lidar.fit_front_line(fov_deg=30.0)
            if result is None:
                self.motion.stop()
                self._align_phase = 1
                self._phase_start = time.time()
                self._lost_count = 0
                self._log(f"[{lbl}] No table line -- centering on cube...")
                return

            slope, intercept = result
            angle_err = math.atan(slope)

            if abs(angle_err) < ALIGN_TABLE_SLOPE_THRESH:
                self.motion.stop()
                self._log(f"[{lbl}] Perpendicular (err={math.degrees(angle_err):.1f}deg) -- centering on cube...")
                self._align_phase = 1
                self._phase_start = time.time()
                self._lost_count = 0
                return

            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_TABLE_KP * angle_err))
            if 0 < abs(ang) < 0.03:
                ang = 0.03 * (1 if ang > 0 else -1)
            self.motion.publish(lx=0.0, az=ang)
            self.status_text = f"[{lbl}] Aligning perpendicular err={math.degrees(angle_err):+.1f}deg"

        # ── Phase 1: YOLO cube centering ──
        elif self._align_phase == 1:
            target = self._find_and_label(task.cube_cls)
            if target is None:
                self._lost_count += 1
                if self._last_align_ang != 0.0 and self._lost_count < 30:
                    drift = 0.02 * (1 if self._last_align_ang > 0 else -1)
                    self.motion.publish(lx=0.0, az=drift)
                else:
                    self.motion.stop()
                if self._lost_count > 50:
                    self._log(f"[{lbl}] Cube lost -- final creep anyway")
                    self._align_phase = 2
                    self._phase_start = time.time()
                    _, _, yaw = self._get_odom()
                    self._perp_yaw = yaw
                return

            self._lost_count = 0
            cx, cy, area = target
            pixel_err = (cx - 0.5) * 640

            if abs(pixel_err) < ALIGN_CENTER_PX:
                self.motion.stop()
                _, _, yaw = self._get_odom()
                self._perp_yaw = yaw
                self._log(f"[{lbl}] Cube centered (err={pixel_err:+.0f}px) -- final creep...")
                self._align_phase = 2
                self._phase_start = time.time()
                self._last_align_ang = 0.0
                return

            norm_err = (cx - 0.5) / 0.5
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -0.4 * norm_err))
            if 0 < abs(ang) < 0.03:
                ang = 0.03 * (1 if ang > 0 else -1)
            self._last_align_ang = ang
            self.motion.publish(lx=0.0, az=ang)
            self.status_text = f"[{lbl}] Centering cube err={pixel_err:+.0f}px"

        # ── Phase 2: Final creep to pick distance ──
        elif self._align_phase == 2:
            front_d = self.lidar.get_front_distance()
            pick_dist = 0.19  # target distance from table

            if front_d <= pick_dist:
                self.motion.stop()
                # Re-center check: verify cube is centered before picking
                target = self._find_and_label(task.cube_cls)
                if target is not None:
                    pixel_err = (target[0] - 0.5) * 640
                    if abs(pixel_err) > ALIGN_CENTER_PX:
                        # Cube drifted — rotate to re-center
                        norm_err = (target[0] - 0.5) / 0.5
                        ang = max(-0.15, min(0.15, -0.3 * norm_err))
                        self.motion.publish(lx=0.0, az=ang)
                        self.status_text = f"[{lbl}] Re-centering at pick dist err={pixel_err:+.0f}px"
                        return
                self._log(f"[{lbl}] At pick distance ({front_d:.2f}m) centered -- picking!")
                self._align_phase = 0
                self.state = ST_PICK_OBJECT
                self._pick_phase = 0
                self._pick_timer = time.time()
                return

            # Creep forward slowly with heading-hold + stronger YOLO tracking
            _, _, cur_yaw = self._get_odom()
            yaw_err = self._perp_yaw - cur_yaw
            while yaw_err > math.pi:  yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX, 0.5 * yaw_err))

            target = self._find_and_label(task.cube_cls)
            if target is not None:
                trim = max(-0.04, min(0.04, -0.10 * (target[0] - 0.5)))
                ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX, ang + trim))

            self.motion.publish(lx=0.02, az=ang)
            self.status_text = f"[{lbl}] Final creep d={front_d:.2f}m -> {pick_dist:.2f}m"

    # ── PICK sequence ────────────────────────────────────────────────

    def _pick_object(self, task: TaskDef, lbl: str):
        t = time.time() - self._pick_timer
        p = self._pick_phase

        # Grip: close visually + IFRA attach for reliable hold
        if p == 0:
            self.motion.stop()
            self.arm.open_gripper()
            self._pick_phase = 1; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Opening gripper..."
        elif p == 1 and t > 1.5:
            self.arm.pre_pick(3.0, color=lbl)
            self._pick_phase = 2; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Arm above cube..."
        elif p == 2 and t > 3.5:
            self.arm.pick(3.0, color=lbl)
            self._pick_phase = 3; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Descending to cube..."
        elif p == 3 and t > 4.0:
            # Close gripper visually
            self.arm.close_gripper()
            self._pick_phase = 4; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Gripping cube..."
            self._log(f"[{lbl}] Gripper closing...")
        elif p == 4 and t > 2.0:
            # IFRA attach to lock cube
            if self._robot:
                self._robot.attach_cube(lbl)
            self._pick_phase = 5; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Cube locked!"
            self._log(f"[{lbl}] IFRA attach — cube secured!")
        elif p == 5 and t > 1.5:
            self.arm.lift(4.0, color=lbl)
            self._pick_phase = 6; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Lifting..."
        elif p == 6 and t > 5.0:
            self.arm.carry(3.0)
            self._pick_phase = 7; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Carrying..."
            self.status_text = f"[{lbl}] Lifting..."
        elif p == 6 and t > 5.0:
            self.arm.carry(3.0)
            self._pick_phase = 7; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Carrying..."
        elif p == 7 and t > 4.0:
            self.status_text = f"[{lbl}] Grab complete -- backing up..."
            self._log(self.status_text)
            self.state = ST_BACKUP_PICK
            self._phase_start = time.time()

    def _backup_pick(self, task: TaskDef, lbl: str):
        elapsed = time.time() - self._phase_start
        if elapsed < BACKUP_PICK_TIME:
            self.motion.publish(lx=BACKUP_VEL)
            self.status_text = f"[{lbl}] Backing up... {BACKUP_PICK_TIME - elapsed:.1f}s"
        else:
            self.motion.stop()
            self.arm.home()
            # Navigate to drop zone using map
            self._start_nav_to_zone(task)

    # ── DROP ZONE: nav + visual servo + place ────────────────────────

    def _search_drop_zone(self, task: TaskDef, lbl: str):
        self.status_text = f"[{lbl}] Scanning for {self.perception.names[task.zone_cls]}..."
        target = self._find_and_label(task.zone_cls)
        if target:
            self.motion.stop()
            self.state = ST_APPROACH_DROP
            self._phase_start = time.time()
            self._zone_in_final = False
            self._lost_count = 0
            self.status_text = f"[{lbl}] Drop zone found!"
            self._log(self.status_text)
        else:
            self.motion.publish(az=-SCAN_ANG_VEL)

    def _approach_drop(self, task: TaskDef, lbl: str):
        """Approach basket. Baskets are ground-level -- LIDAR can't see them.
        Use pure visual servo + time-based approach."""
        target = self._find_and_label(task.zone_cls)
        elapsed = time.time() - self._phase_start

        if not self._zone_in_final:
            # Coarse approach: drive toward zone using YOLO
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
                self.status_text = f"[{lbl}] Near basket -- final creep..."
                self._log(self.status_text)
            elif abs(pixel_err) > APPROACH_DRIFT_PX:
                ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_KP * err))
                self.motion.publish(lx=0.0, az=ang)
            else:
                ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                           -APPROACH_MILD_KP * err))
                self.motion.publish(lx=APPROACH_FWD, az=ang)
                self.status_text = f"[{lbl}] -> basket area={area:.4f}"
        else:
            # Final creep: drive close to basket (no LIDAR stop)
            fe = time.time() - self._zone_final_start
            if fe < ZONE_FINAL_TIME:
                if target:
                    err = target[0] - 0.5
                    ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                               -APPROACH_MILD_KP * err))
                    self.motion.publish(lx=ZONE_FINAL_VEL, az=ang)
                else:
                    self.motion.publish(lx=ZONE_FINAL_VEL, az=0.0)
                self.status_text = f"[{lbl}] Creeping to basket... {ZONE_FINAL_TIME - fe:.1f}s"
            else:
                self.motion.stop()
                self.state = ST_ADJUST_DROP
                self._phase_start = time.time()
                self.status_text = f"[{lbl}] At basket -- adjusting..."
                self._log(self.status_text)

    def _adjust_drop(self, task: TaskDef, lbl: str):
        """Final adjust at basket. Time-based only (baskets invisible to LIDAR)."""
        elapsed = time.time() - self._phase_start

        if elapsed >= ZONE_ADJUST_TIME:
            self.motion.stop()
            time.sleep(0.3)
            self.state = ST_PLACE_OBJECT
            self._drop_phase = 0
            self._drop_timer = time.time()
            self.status_text = f"[{lbl}] Over basket -- dropping from above..."
            self._log(self.status_text)
            return

        # Keep creeping with visual tracking
        target = self._find_and_label(task.zone_cls)
        if target:
            err = target[0] - 0.5
            ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX,
                       -APPROACH_MILD_KP * err))
            self.motion.publish(lx=ADJUST_FWD, az=ang)
        else:
            self.motion.publish(lx=ADJUST_FWD, az=0.0)
        self.status_text = f"[{lbl}] Adjust at basket {ZONE_ADJUST_TIME - elapsed:.1f}s"

    # ── PLACE: drop cube INTO basket from above ──────────────────────

    def _place_object(self, task: TaskDef, lbl: str):
        """Drop cube into basket. Robot is at box edge facing +X.
        Arm extends over the wall, opens gripper, cube drops into box center."""
        t = time.time() - self._drop_timer
        p = self._drop_phase

        if p == 0:
            self.motion.stop()
            self.arm.drop_extend(3.0)
            self._drop_phase = 1; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] Extending arm over basket wall..."
        elif p == 1 and t > 4.0:
            self.arm.drop_over(3.0)
            self._drop_phase = 2; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] Positioning over basket center..."
        elif p == 2 and t > 4.0:
            # IFRA detach + open gripper
            if self._robot:
                self._robot.detach_cube(lbl)
            self.arm.open_gripper()
            self._drop_phase = 3; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] DROPPING -- cube into basket!"
            self._log(f"[{lbl}] Cube detached + gripper opened!")
        elif p == 3 and t > 2.5:
            self.arm.drop_retreat(3.0)
            self._drop_phase = 4; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] Retracting arm..."
        elif p == 4 and t > 3.5:
            self.arm.home()
            self._drop_phase = 5; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] Arm home..."
        elif p == 5 and t > 3.0:
            self.status_text = f"[{lbl}] Cube dropped into {lbl} basket!"
            self._log(self.status_text)
            self.state = ST_BACKUP_DROP
            self._phase_start = time.time()

    def _backup_drop(self, task: TaskDef, lbl: str):
        elapsed = time.time() - self._phase_start
        if elapsed < BACKUP_DROP_TIME:
            self.motion.publish(lx=BACKUP_VEL)
            self.status_text = f"[{lbl}] Retreating... {BACKUP_DROP_TIME - elapsed:.1f}s"
        else:
            self.motion.stop()
            self.state = ST_NEXT_OBJECT

    def _next_object(self, task: TaskDef, lbl: str):
        self.task_idx += 1
        if self.task_idx < len(self.tasks):
            nxt_task = self.tasks[self.task_idx]
            nxt = nxt_task.label
            self.arm.home()
            # Navigate to next cube via map
            self._start_nav_to_cube(nxt_task)

    # ── Return home ──────────────────────────────────────────────────

    def _run_return_home(self):
        # Navigate to dock search area, then start YOLO dock search
        if self.navigator.is_active():
            self.navigator.tick()
            self.status_text = f"Returning: {self.navigator.status_text}"
            if self.navigator.is_done():
                self._start_dock_search()
            elif self.navigator.has_failed():
                self._simple_drive_to_dock_area()
            return

        self._simple_drive_to_dock_area()

    def _simple_drive_to_dock_area(self):
        x, y, yaw = self._get_odom()
        dx = DOCK_POSITION[0] - x
        dy = DOCK_POSITION[1] - y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < RETURN_HOME_THRESH:
            self.motion.stop()
            self._start_dock_search()
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err = target_yaw - yaw
        while yaw_err > math.pi:  yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        if abs(yaw_err) > 0.15:
            self.motion.publish(
                az=max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3 * yaw_err)))
        else:
            fwd = min(APPROACH_FWD, 0.5 * dist)
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3 * yaw_err))
            self.motion.publish(lx=fwd, az=ang)
        self.status_text = f"Driving to dock area -- {dist:.2f}m"

    # ── Charging dock (YOLO visual servo) ──────────────────────────

    def _start_dock_search(self):
        """Face the dock direction and begin YOLO search."""
        self.motion.stop()
        self.arm.home()
        self.state = ST_SEARCH_DOCK
        self._lost_count = 0
        self._dock_facing = False
        self.status_text = "Searching for charging dock..."
        self._log(self.status_text)

    def _search_dock(self):
        """Rotate to find charging_dock via YOLO, first face south."""
        # First turn to face the dock direction
        if not self._dock_facing:
            x, y, yaw = self._get_odom()
            yaw_err = DOCK_FACE_YAW - yaw
            while yaw_err > math.pi:  yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            if abs(yaw_err) > 0.08:
                self.motion.publish(
                    az=max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.4 * yaw_err)))
                self.status_text = f"Facing dock area... {math.degrees(yaw_err):.0f}deg"
                return
            self.motion.stop()
            self._dock_facing = True

        # Search for dock via YOLO
        dock_cls = self.perception.CLS_DOCK
        target = self._find(dock_cls)
        if target:
            self.motion.stop()
            self.state = ST_ALIGN_DOCK
            self._lost_count = 0
            self._max_area = 0.0
            self._last_align_ang = 0.0
            self.status_text = "Charging dock detected! Aligning..."
            self._log(self.status_text)
        else:
            # Slow scan to find it
            self.motion.publish(az=SCAN_ANG_VEL)
            self.status_text = "Scanning for charging dock..."

    def _align_dock(self):
        """Center the charging dock in the camera frame."""
        dock_cls = self.perception.CLS_DOCK
        target = self._find(dock_cls)

        if target is None:
            self._lost_count += 1
            if self._lost_count > ALIGN_LOST_MAX:
                self.state = ST_SEARCH_DOCK
                self._lost_count = 0
                self._dock_facing = True  # already roughly facing
            else:
                if self._last_align_ang != 0.0:
                    drift = 0.03 * (1 if self._last_align_ang > 0 else -1)
                    self.motion.publish(az=drift)
                else:
                    self.motion.stop()
            self.status_text = f"Dock lost ({self._lost_count})"
            return

        self._lost_count = 0
        cx, cy, area = target
        self._max_area = max(self._max_area, area)
        err = cx - 0.5
        pixel_err = err * 640

        if abs(pixel_err) <= DOCK_ALIGN_CENTER_PX:
            self.motion.stop()
            self.state = ST_APPROACH_DOCK
            self._phase_start = time.time()
            self._lost_count = 0
            self._last_align_ang = 0.0
            self.status_text = "Dock centered! Approaching..."
            self._log(self.status_text)
        else:
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_KP * err))
            if 0 < abs(ang) < 0.05:
                ang = 0.05 * (1 if ang > 0 else -1)
            self._last_align_ang = ang
            self.motion.publish(az=ang)
            self.status_text = f"Aligning dock err={pixel_err:+.0f}px"

    def _approach_dock(self):
        """Drive toward dock using YOLO visual servo, stop by odom distance."""
        elapsed = time.time() - self._phase_start
        x, y, yaw = self._get_odom()
        dx = DOCK_CENTER[0] - x
        dy = DOCK_CENTER[1] - y
        odom_dist = math.sqrt(dx * dx + dy * dy)

        # Close enough by odometry — start final creep
        if odom_dist < DOCK_STOP_ODOM_DIST:
            self.motion.stop()
            self.state = ST_DOCK_CREEP
            self._dock_creep_start = time.time()
            self.status_text = f"At dock ({odom_dist:.2f}m) — creeping into position..."
            self._log(self.status_text)
            return

        if elapsed > DOCK_APPROACH_TIMEOUT:
            # Timeout — just creep from here
            self.motion.stop()
            self.state = ST_DOCK_CREEP
            self._dock_creep_start = time.time()
            self.status_text = "Approach timeout — creeping into dock..."
            self._log(self.status_text)
            return

        # Visual servo: keep dock centered while driving forward
        dock_cls = self.perception.CLS_DOCK
        target = self._find(dock_cls)

        if target is not None:
            self._lost_count = 0
            cx = target[0]
            pixel_err = (cx - 0.5) * 640
            if abs(pixel_err) > DOCK_ALIGN_CENTER_PX * 3:
                ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_KP * (cx - 0.5)))
                self.motion.publish(lx=0.0, az=ang)
            else:
                ang = max(-0.10, min(0.10, -0.3 * (cx - 0.5)))
                self.motion.publish(lx=DOCK_APPROACH_VEL, az=ang)
            self.status_text = f"Approaching dock odom:{odom_dist:.2f}m"
        else:
            self._lost_count += 1
            if self._lost_count > 60:
                # Lost dock too long — re-search
                self.motion.stop()
                self.state = ST_SEARCH_DOCK
                self._dock_facing = True
                self._lost_count = 0
            else:
                # Drive forward slowly, hope to reacquire
                self.motion.publish(lx=DOCK_APPROACH_VEL * 0.5)
            self.status_text = f"Dock lost, driving... ({self._lost_count})"

    def _dock_creep(self):
        """Slow forward creep into the dock."""
        elapsed = time.time() - self._dock_creep_start
        if elapsed < DOCK_CREEP_TIME:
            self.motion.publish(lx=DOCK_CREEP_VEL)
            self.status_text = f"Docking... {DOCK_CREEP_TIME - elapsed:.1f}s"
        else:
            self.motion.stop()
            self.arm.stow()
            self.state = ST_PARKED
            self.running = False
            self.status_text = "Robot parked at charging dock -- Charging -- Mission Complete!"
            self._log(self.status_text)
            # Change dock LED from orange to red (charging indicator)
            if self._robot is not None:
                self._robot.set_dock_led_color(1.0, 0.0, 0.0)
