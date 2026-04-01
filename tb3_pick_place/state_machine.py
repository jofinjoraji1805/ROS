#!/usr/bin/env python3
"""
state_machine.py -- Autonomous pick-and-place with map-based navigation.

Two-phase navigation per cube:
  1. NAV_TO_CUBE: A* path planning to approach point near the table
  2. YOLO visual servo: SEARCH -> ALIGN -> APPROACH -> ADJUST -> PICK

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
    ST_APPROACH_OBJECT, ST_ADJUST_POSITION, ST_ALIGN_TABLE, ST_PICK_OBJECT,
    ST_BACKUP_PICK, ST_NAV_TO_ZONE, ST_SEARCH_DROP_ZONE,
    ST_APPROACH_DROP, ST_ADJUST_DROP, ST_PLACE_OBJECT,
    ST_BACKUP_DROP, ST_NEXT_OBJECT, ST_RETURN_HOME, ST_DONE,
    SCAN_ANG_VEL, ALIGN_CENTER_PX, ALIGN_KP, ALIGN_LOST_MAX,
    MAX_ANG_VEL, APPROACH_FWD, APPROACH_DRIFT_PX,
    APPROACH_MILD_KP, APPROACH_MILD_MAX, APPROACH_TIMEOUT,
    CUBE_LOST_TICKS, ADJUST_FWD, ADJUST_DURATION,
    BACKUP_VEL, BACKUP_PICK_TIME, BACKUP_DROP_TIME,
    ZONE_CLOSE_AREA, ZONE_APPROACH_TIME, ZONE_FINAL_TIME,
    ZONE_FINAL_VEL, ZONE_ADJUST_TIME, RETURN_HOME_THRESH,
    TABLE_STOP_DISTANCE, OBSTACLE_SLOW_DISTANCE, OBSTACLE_SLOW_FACTOR,
    CUBE_NAV_TARGETS, ZONE_NAV_TARGETS,
    CUBE_FACE_YAW, ZONE_FACE_YAW, FACE_YAW_TOLERANCE,
    ALIGN_TABLE_KP, ALIGN_TABLE_SLOPE_THRESH, ALIGN_TABLE_TIMEOUT,
    CUBE_TABLE_Y,
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
        target = ZONE_NAV_TARGETS.get(lbl)
        if target and self.navigator.map_loaded:
            ok = self.navigator.navigate_to(*target)
            if ok:
                self.state = ST_NAV_TO_ZONE
                self.status_text = f"[{lbl}] Navigating to {lbl} drop zone..."
                self._log(self.status_text)
                return
        # Fallback
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
        if not self.running or self.state in (ST_DONE, ST_IDLE):
            return

        if self.task_idx >= len(self.tasks) and self.state != ST_RETURN_HOME:
            self.motion.stop()
            self.arm.home()
            self.state = ST_RETURN_HOME
            self.status_text = "All cubes delivered -- returning home..."
            self._log(self.status_text)
            # Navigate home
            if self.navigator.map_loaded:
                self.navigator.navigate_to(self._home_x, self._home_y)
            return

        if self.state == ST_RETURN_HOME:
            self._run_return_home()
            return

        task = self.tasks[self.task_idx]
        lbl = task.label

        handler = {
            ST_NAV_TO_CUBE: self._nav_to_cube,
            ST_DRIVE_TO_CUBE: self._drive_to_cube,
            ST_SEARCH_OBJECT: self._search_object,
            ST_ALIGN_OBJECT: self._align_object,
            ST_APPROACH_OBJECT: self._approach_object,
            ST_ADJUST_POSITION: self._adjust_position,
            ST_ALIGN_TABLE: self._align_table,
            ST_PICK_OBJECT: self._pick_object,
            ST_BACKUP_PICK: self._backup_pick,
            ST_NAV_TO_ZONE: self._nav_to_zone,
            ST_SEARCH_DROP_ZONE: self._search_drop_zone,
            ST_APPROACH_DROP: self._approach_drop,
            ST_ADJUST_DROP: self._adjust_drop,
            ST_PLACE_OBJECT: self._place_object,
            ST_BACKUP_DROP: self._backup_drop,
            ST_NEXT_OBJECT: self._next_object,
        }.get(self.state)

        if handler:
            handler(task, lbl)

        # Keep cube following robot during carry/nav states
        if self._robot and self.state in (
            ST_BACKUP_PICK, ST_NAV_TO_ZONE, ST_SEARCH_DROP_ZONE,
            ST_APPROACH_DROP, ST_ADJUST_DROP, ST_PLACE_OBJECT,
        ):
            self._robot.carry_cube(lbl)

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

    # ── NAV_TO_ZONE: A* navigation to drop zone approach point ───────

    def _nav_to_zone(self, task: TaskDef, lbl: str):
        if self._facing:
            if self._face_yaw_tick(lbl):
                self.state = ST_SEARCH_DROP_ZONE
                self._lost_count = 0
                self._log(f"[{lbl}] Facing basket -- scanning...")
            return

        self.navigator.tick()
        self.status_text = f"[{lbl}] {self.navigator.status_text}"

        if self.navigator.is_done():
            self._facing = True
            self._face_target = ZONE_FACE_YAW
            self._log(f"[{lbl}] Near zone -- turning to face basket...")
        elif self.navigator.has_failed():
            self._facing = True
            self._face_target = ZONE_FACE_YAW
            self._log(f"[{lbl}] Nav failed -- turning to face basket...")

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
            # Save current heading as fallback if cube is lost during approach
            _, _, yaw = self._get_odom()
            self._perp_yaw = yaw
            self.state = ST_APPROACH_OBJECT
            self._phase_start = time.time()
            self._lost_count = 0
            self._last_align_ang = 0.0
            self.status_text = f"[{lbl}] Cube centered! Visual servo approach..."
            self._log(self.status_text)
        else:
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, -ALIGN_KP * err))
            if 0 < abs(ang) < 0.05:
                ang = 0.05 * (1 if ang > 0 else -1)
            self._last_align_ang = ang
            self.motion.publish(lx=0.0, az=ang)
            self.status_text = f"[{lbl}] Aligning err={pixel_err:+.0f}px"

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

        # YOLO visual servo: continuously steer to keep cube centered
        target = self._find_and_label(task.cube_cls)
        if target is not None:
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
            pick_dist = 0.20  # target distance from table

            if front_d <= pick_dist:
                self.motion.stop()
                self._log(f"[{lbl}] At pick distance ({front_d:.2f}m) -- picking!")
                self._align_phase = 0
                self.state = ST_PICK_OBJECT
                self._pick_phase = 0
                self._pick_timer = time.time()
                return

            # Creep forward slowly with heading-hold + gentle YOLO tracking
            _, _, cur_yaw = self._get_odom()
            yaw_err = self._perp_yaw - cur_yaw
            while yaw_err > math.pi:  yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX, 0.5 * yaw_err))

            target = self._find_and_label(task.cube_cls)
            if target is not None:
                trim = max(-0.02, min(0.02, -0.05 * (target[0] - 0.5)))
                ang = max(-APPROACH_MILD_MAX, min(APPROACH_MILD_MAX, ang + trim))

            self.motion.publish(lx=0.02, az=ang)
            self.status_text = f"[{lbl}] Final creep d={front_d:.2f}m -> {pick_dist:.2f}m"

    # ── PICK sequence ────────────────────────────────────────────────

    def _pick_object(self, task: TaskDef, lbl: str):
        t = time.time() - self._pick_timer
        p = self._pick_phase

        # Simulated pick: teleport cube to gripper early to avoid arm collision.
        # Sequence: open gripper → ready → pre_pick → TELEPORT cube → close → lift → carry
        if p == 0:
            self.motion.stop()
            self.arm.open_gripper()
            self._pick_phase = 1; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Opening gripper..."
        elif p == 1 and t > 2.0:
            self.arm.ready(3.0)
            self._pick_phase = 2; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Arm ready..."
        elif p == 2 and t > 4.0:
            # Teleport cube to gripper BEFORE arm reaches down
            if self._robot:
                self._robot.attach_cube(lbl)
            self.arm.pre_pick(3.0, color=lbl)
            self._pick_phase = 3; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Grabbing cube..."
            self._log(f"[{lbl}] Cube grabbed (simulated grasp)")
        elif p == 3 and t > 4.0:
            # Close gripper around the teleported cube
            self.arm.gripper(-0.010, effort=20.0)
            self._pick_phase = 4; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Closing gripper..."
        elif p == 4 and t > 2.0:
            self.arm.lift(3.0, color=lbl)
            self._pick_phase = 5; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Lifting..."
        elif p == 5 and t > 4.0:
            self.arm.carry(3.0)
            self._pick_phase = 6; self._pick_timer = time.time()
            self.status_text = f"[{lbl}] Carrying..."
        elif p == 6 and t > 4.0:
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
        """Drop cube from above into basket. Arm extends over basket,
        gripper opens, cube falls in."""
        t = time.time() - self._drop_timer
        p = self._drop_phase

        if p == 0:
            self.arm.drop_extend(3.0)
            self._drop_phase = 1; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] Extending arm above basket..."
        elif p == 1 and t > 3.5:
            self.arm.drop_over(3.0)
            self._drop_phase = 2; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] Positioning over basket..."
        elif p == 2 and t > 3.5:
            self.arm.open_gripper()
            # Simulated drop: place cube at drop zone location
            if self._robot:
                x, y, _ = self._get_odom()
                drop_x = x + 0.20 * math.cos(self._get_odom()[2])
                drop_y = y + 0.20 * math.sin(self._get_odom()[2])
                self._robot.drop_cube(lbl, drop_x, drop_y, 0.05)
            self._drop_phase = 3; self._drop_timer = time.time()
            self.status_text = f"[{lbl}] DROPPING -- cube falling into basket!"
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
        # Use navigator if active
        if self.navigator.is_active():
            self.navigator.tick()
            self.status_text = f"Returning: {self.navigator.status_text}"
            if self.navigator.is_done():
                self.motion.stop()
                self.arm.home()
                self.state = ST_DONE
                self.running = False
                self.status_text = "HOME -- ALL TASKS COMPLETE!"
                self._log(self.status_text)
            elif self.navigator.has_failed():
                # Fallback: simple drive home
                self._simple_drive_home()
            return

        self._simple_drive_home()

    def _simple_drive_home(self):
        x, y, yaw = self._get_odom()
        dx = self._home_x - x
        dy = self._home_y - y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < RETURN_HOME_THRESH:
            yaw_err = self._home_yaw - yaw
            while yaw_err > math.pi:  yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi
            if abs(yaw_err) > 0.10:
                self.motion.publish(
                    az=max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3 * yaw_err)))
            else:
                self.motion.stop()
                self.arm.home()
                self.state = ST_DONE
                self.running = False
                self.status_text = "HOME -- ALL TASKS COMPLETE!"
                self._log(self.status_text)
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err = target_yaw - yaw
        while yaw_err > math.pi:  yaw_err -= 2 * math.pi
        while yaw_err < -math.pi: yaw_err += 2 * math.pi

        if abs(yaw_err) > 0.15:
            self.motion.publish(
                az=max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3 * yaw_err)))
        else:
            fwd = self._safe_fwd(min(APPROACH_FWD, 0.5 * dist))
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.3 * yaw_err))
            self.motion.publish(lx=fwd, az=ang)
        self.status_text = f"Returning home -- {dist:.2f}m"
