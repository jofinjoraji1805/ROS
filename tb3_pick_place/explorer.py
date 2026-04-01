#!/usr/bin/env python3
"""
explorer.py -- Autonomous room exploration for SLAM map building.

Drives through a predefined set of waypoints that cover the entire room.
While driving, the LIDAR processor builds an occupancy grid map.
Once exploration is complete, the map is saved to disk.

The explorer uses a simple go-to-point controller with LIDAR obstacle
avoidance to navigate between waypoints safely.
"""

import math
import time
import os
import cv2
import numpy as np
from typing import Callable, List, Tuple

from .config import (
    EXPLORE_WAYPOINTS, MAP_SAVE_DIR,
    MAX_ANG_VEL, TABLE_STOP_DISTANCE,
    MAP_SIZE, MAP_RESOLUTION, MAP_ORIGIN_OFFSET,
)
from .motion import MotionController
from .lidar import LidarProcessor


# Exploration speeds
EXPLORE_FWD = 0.12          # m/s forward (a bit faster for mapping)
EXPLORE_ANG = 0.30          # rad/s rotation
WP_REACH_DIST = 0.25        # m - close enough to waypoint
WP_HEADING_TOL = 0.12       # rad - heading tolerance
OBSTACLE_DODGE_ANG = 0.35   # rad/s dodge rotation
EXPLORE_OBSTACLE_DIST = 0.30  # m - obstacle avoidance during explore


class Explorer:
    """
    Autonomous room exploration state machine.

    States: IDLE -> ROTATING -> DRIVING -> ARRIVED -> SAVING -> DONE
    """

    ST_IDLE = "EXPLORE_IDLE"
    ST_ROTATING = "EXPLORE_ROTATING"
    ST_DRIVING = "EXPLORE_DRIVING"
    ST_ARRIVED = "EXPLORE_ARRIVED"
    ST_SAVING = "EXPLORE_SAVING"
    ST_DONE = "EXPLORE_DONE"

    def __init__(
        self,
        motion: MotionController,
        lidar: LidarProcessor,
        get_odom_fn: Callable[[], Tuple[float, float, float]],
        log_fn: Callable[[str], None],
    ):
        self.motion = motion
        self.lidar = lidar
        self._get_odom = get_odom_fn
        self._log = log_fn

        self.waypoints: List[Tuple[float, float]] = list(EXPLORE_WAYPOINTS)
        self.wp_idx = 0
        self.state = self.ST_IDLE
        self.status_text = "Exploration ready"
        self.running = False
        self.progress = 0.0  # 0.0 to 1.0

    def start(self):
        """Begin autonomous exploration."""
        self.wp_idx = 0
        self.state = self.ST_ROTATING
        self.running = True
        self.status_text = "Exploration started -- mapping the room..."
        self._log(self.status_text)

    def stop(self):
        self.motion.stop()
        self.running = False
        self.state = self.ST_IDLE
        self.status_text = "Exploration stopped"
        self._log(self.status_text)

    def tick(self):
        """Call at 10 Hz."""
        if not self.running or self.state in (self.ST_DONE, self.ST_IDLE):
            return

        if self.state == self.ST_SAVING:
            self._save_map()
            return

        if self.wp_idx >= len(self.waypoints):
            self.motion.stop()
            self.state = self.ST_SAVING
            self.status_text = "Exploration complete -- saving map..."
            self._log(self.status_text)
            return

        x, y, yaw = self._get_odom()
        tx, ty = self.waypoints[self.wp_idx]
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx * dx + dy * dy)
        target_yaw = math.atan2(dy, dx)

        yaw_err = target_yaw - yaw
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi

        self.progress = self.wp_idx / len(self.waypoints)
        front_d = self.lidar.get_front_distance()

        if self.state == self.ST_ROTATING:
            if dist < WP_REACH_DIST:
                # Already at waypoint
                self._advance_wp()
                return

            if abs(yaw_err) < WP_HEADING_TOL:
                self.motion.stop()
                self.state = self.ST_DRIVING
                self.status_text = (
                    f"Explore: heading OK -- driving to WP "
                    f"{self.wp_idx + 1}/{len(self.waypoints)}")
            else:
                ang = max(-EXPLORE_ANG, min(EXPLORE_ANG, 1.0 * yaw_err))
                if 0 < abs(ang) < 0.08:
                    ang = 0.08 * (1 if ang > 0 else -1)
                self.motion.publish(lx=0.0, az=ang)
                self.status_text = (
                    f"Explore: rotating to WP {self.wp_idx + 1} "
                    f"({math.degrees(yaw_err):+.0f} deg)")

        elif self.state == self.ST_DRIVING:
            if dist < WP_REACH_DIST:
                self._advance_wp()
                return

            # Obstacle avoidance during exploration
            if front_d < EXPLORE_OBSTACLE_DIST:
                # Obstacle ahead -- dodge by rotating away
                dodge_dir = 1.0 if yaw_err > 0 else -1.0
                self.motion.publish(lx=0.0, az=dodge_dir * OBSTACLE_DODGE_ANG)
                self.status_text = (
                    f"Explore: obstacle at {front_d:.2f}m -- dodging...")
                return

            # Recheck heading while driving
            if abs(yaw_err) > 0.4:
                self.state = self.ST_ROTATING
                self.motion.stop()
                return

            # Drive with mild heading correction
            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, 0.8 * yaw_err))
            fwd = EXPLORE_FWD
            # Slow down near obstacles
            if front_d < 0.50:
                fwd *= 0.5
            self.motion.publish(lx=fwd, az=ang)
            self.status_text = (
                f"Explore: WP {self.wp_idx + 1}/{len(self.waypoints)} "
                f"dist={dist:.2f}m  lidar={front_d:.2f}m")

    def _advance_wp(self):
        """Move to next waypoint."""
        self.wp_idx += 1
        self.motion.stop()
        if self.wp_idx < len(self.waypoints):
            self.state = self.ST_ROTATING
            self.status_text = (
                f"Explore: reached WP {self.wp_idx} -- "
                f"heading to WP {self.wp_idx + 1}")
            self._log(self.status_text)
        self.progress = self.wp_idx / len(self.waypoints)

    def _save_map(self):
        """Save the occupancy grid map to disk."""
        os.makedirs(MAP_SAVE_DIR, exist_ok=True)

        # Save raw numpy grid
        grid = self.lidar.map_grid.copy()
        np_path = os.path.join(MAP_SAVE_DIR, "map_grid.npy")
        np.save(np_path, grid)

        # Save as image (PGM-style for Nav2 compatibility)
        img_path = os.path.join(MAP_SAVE_DIR, "map.pgm")
        # Flip vertically for standard map orientation
        cv2.imwrite(img_path, np.flipud(grid))

        # Save as colour PNG for visualization
        vis_path = os.path.join(MAP_SAVE_DIR, "map_visual.png")
        vis = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
        # Mark unknown as grey-blue, free as white, occupied as black
        cv2.imwrite(vis_path, np.flipud(vis))

        # Save YAML metadata
        yaml_path = os.path.join(MAP_SAVE_DIR, "map.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"image: map.pgm\n")
            f.write(f"resolution: {MAP_RESOLUTION}\n")
            f.write(f"origin: [{-MAP_ORIGIN_OFFSET}, {-MAP_ORIGIN_OFFSET}, 0.0]\n")
            f.write(f"negate: 0\n")
            f.write(f"occupied_thresh: 0.65\n")
            f.write(f"free_thresh: 0.196\n")

        self._log(f"Map saved to {MAP_SAVE_DIR}")
        self._log(f"  Grid: {np_path}")
        self._log(f"  Image: {img_path}")
        self._log(f"  YAML: {yaml_path}")

        self.state = self.ST_DONE
        self.running = False
        self.progress = 1.0
        self.status_text = "MAP SAVED -- ready for mission!"
        self._log(self.status_text)
