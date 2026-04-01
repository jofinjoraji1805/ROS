#!/usr/bin/env python3
"""
lidar.py -- LIDAR processing for obstacle avoidance and simple SLAM mapping.

Processes /scan (LaserScan) data to provide:
  - Front obstacle distance (for table collision avoidance)
  - Simple occupancy grid map built from accumulated scans + odometry
"""

import math
import numpy as np
from typing import Optional, Tuple

from .config import (
    LIDAR_FRONT_HALF_ANGLE, TABLE_STOP_DISTANCE,
    MAP_SIZE, MAP_RESOLUTION, MAP_ORIGIN_OFFSET,
)


class LidarProcessor:
    """Processes laser scan data for obstacle detection and mapping."""

    def __init__(self):
        # Latest scan data
        self.front_distance: float = float('inf')
        self.ranges: Optional[np.ndarray] = None
        self.angle_min: float = 0.0
        self.angle_increment: float = 0.0

        # Simple occupancy grid map
        grid_cells = int(MAP_SIZE / MAP_RESOLUTION)
        self.map_grid = np.full((grid_cells, grid_cells), 127, dtype=np.uint8)
        self._grid_cells = grid_cells

    def update_scan(self, ranges, angle_min, angle_max, angle_increment,
                    range_min, range_max):
        """Update from a LaserScan message."""
        self.ranges = np.array(ranges, dtype=np.float32)
        self.angle_min = angle_min
        self.angle_increment = angle_increment

        # Replace inf/nan with max range
        valid = np.isfinite(self.ranges) & (self.ranges >= range_min)
        self.ranges[~valid] = range_max

        # Compute front obstacle distance
        # Front is around angle=0, check ±FRONT_HALF_ANGLE
        n = len(self.ranges)
        if n == 0:
            return

        angles = angle_min + np.arange(n) * angle_increment

        # Normalise angles to [-pi, pi]
        angles = (angles + math.pi) % (2 * math.pi) - math.pi

        # Front cone: angles near 0
        front_mask = np.abs(angles) < LIDAR_FRONT_HALF_ANGLE
        if np.any(front_mask):
            front_ranges = self.ranges[front_mask]
            valid_front = front_ranges[
                (front_ranges > range_min) & (front_ranges < range_max)]
            if len(valid_front) > 0:
                self.front_distance = float(np.min(valid_front))
            else:
                self.front_distance = float('inf')
        else:
            self.front_distance = float('inf')

    def update_map(self, odom_x: float, odom_y: float, odom_yaw: float):
        """Add current scan points to the occupancy grid map."""
        if self.ranges is None:
            return

        n = len(self.ranges)
        angles = self.angle_min + np.arange(n) * self.angle_increment

        # World-frame coordinates of each scan point
        world_angles = angles + odom_yaw
        xs = odom_x + self.ranges * np.cos(world_angles)
        ys = odom_y + self.ranges * np.sin(world_angles)

        # Convert to grid coordinates
        gx = ((xs + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION).astype(int)
        gy = ((ys + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION).astype(int)

        # Mark robot position as free
        rx = int((odom_x + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)
        ry = int((odom_y + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)
        if 0 <= rx < self._grid_cells and 0 <= ry < self._grid_cells:
            # Mark small area around robot as free
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < self._grid_cells and 0 <= ny < self._grid_cells:
                        self.map_grid[ny, nx] = 255  # free

        # Mark obstacle points
        valid = (
            (gx >= 0) & (gx < self._grid_cells) &
            (gy >= 0) & (gy < self._grid_cells) &
            (self.ranges > 0.12) & (self.ranges < 3.4)
        )
        gx_v = gx[valid]
        gy_v = gy[valid]
        self.map_grid[gy_v, gx_v] = 0  # occupied

        # Ray-trace free space (simplified: mark midpoints)
        mid_xs = odom_x + self.ranges * 0.5 * np.cos(world_angles)
        mid_ys = odom_y + self.ranges * 0.5 * np.sin(world_angles)
        mgx = ((mid_xs + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION).astype(int)
        mgy = ((mid_ys + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION).astype(int)
        mid_valid = (
            (mgx >= 0) & (mgx < self._grid_cells) &
            (mgy >= 0) & (mgy < self._grid_cells) &
            (self.ranges > 0.12)
        )
        self.map_grid[mgy[mid_valid], mgx[mid_valid]] = 255  # free

    def fit_front_line(self, fov_deg: float = 30.0):
        """Fit a line to LiDAR points in the front arc.
        Returns (slope, intercept) or None if not enough points.
        slope ≈ 0 means robot is parallel to the surface ahead."""
        if self.ranges is None or len(self.ranges) == 0:
            return None
        n = len(self.ranges)
        angles = self.angle_min + np.arange(n) * self.angle_increment
        angles = (angles + math.pi) % (2 * math.pi) - math.pi
        fov_rad = math.radians(fov_deg)
        mask = np.abs(angles) < fov_rad
        r = self.ranges[mask].copy()
        a = angles[mask]
        valid = (r > 0.05) & (r < 1.5) & np.isfinite(r)
        if np.sum(valid) < 5:
            return None
        x = r[valid] * np.cos(a[valid])
        y = r[valid] * np.sin(a[valid])
        try:
            coeffs = np.polyfit(x, y, 1)
            return float(coeffs[0]), float(coeffs[1])
        except (np.linalg.LinAlgError, ValueError):
            return None

    def find_table_center_angle(self, fov_deg: float = 60.0, max_dist: float = 0.40):
        """Find the angular center of the table surface in the front arc.

        Scans the front LiDAR arc for a cluster of close-range readings
        (the table surface). Returns the angle to the center of that cluster,
        or None if no table surface found.

        Returns: angle in radians (negative = table center is to the right,
                 positive = to the left). 0 = centered.
        """
        if self.ranges is None or len(self.ranges) == 0:
            return None

        n = len(self.ranges)
        angles = self.angle_min + np.arange(n) * self.angle_increment
        angles = (angles + math.pi) % (2 * math.pi) - math.pi
        fov_rad = math.radians(fov_deg)

        # Get readings in the front arc that are close (table surface)
        mask = (np.abs(angles) < fov_rad) & (self.ranges > 0.05) & (self.ranges < max_dist)
        if np.sum(mask) < 3:
            return None

        table_angles = angles[mask]

        # The table center is the midpoint of the angular span
        center_angle = float((np.min(table_angles) + np.max(table_angles)) / 2.0)
        table_span = float(np.max(table_angles) - np.min(table_angles))

        # Only valid if the table spans a reasonable arc (not just a sliver)
        if table_span < math.radians(5.0):
            return None

        return center_angle

    def is_obstacle_ahead(self) -> bool:
        """True if obstacle within TABLE_STOP_DISTANCE in front."""
        return self.front_distance < TABLE_STOP_DISTANCE

    def get_front_distance(self) -> float:
        """Minimum distance to obstacle in front cone."""
        return self.front_distance

    def get_map_image(self, odom_x: float, odom_y: float,
                      odom_yaw: float) -> np.ndarray:
        """
        Return a coloured BGR map image with robot position marked.
        Grey=unknown, White=free, Black=occupied, Green arrow=robot.
        """
        import cv2

        # Convert greyscale to BGR
        img = cv2.cvtColor(self.map_grid, cv2.COLOR_GRAY2BGR)

        # Robot position
        rx = int((odom_x + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)
        ry = int((odom_y + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)

        if 0 <= rx < self._grid_cells and 0 <= ry < self._grid_cells:
            # Draw robot as green circle + direction arrow
            cv2.circle(img, (rx, ry), 4, (0, 255, 0), -1)
            # Arrow showing heading
            arrow_len = 10
            ax = int(rx + arrow_len * math.cos(odom_yaw))
            ay = int(ry + arrow_len * math.sin(odom_yaw))
            cv2.arrowedLine(img, (rx, ry), (ax, ay), (0, 200, 255), 2,
                            tipLength=0.4)

        return img
