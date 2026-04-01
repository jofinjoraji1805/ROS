#!/usr/bin/env python3
"""
navigator.py -- Map-aware navigation with A* path planning.

Loads a saved occupancy grid map, inflates obstacles to create a costmap,
and uses A* to plan collision-free paths. A path-following controller
drives the robot along the planned path.
"""

import math
import heapq
import os
import cv2
import numpy as np
from typing import Callable, List, Optional, Tuple

from .config import (
    MAP_SAVE_DIR, MAP_RESOLUTION, MAP_ORIGIN_OFFSET,
    MAX_ANG_VEL, TABLE_STOP_DISTANCE,
)
from .motion import MotionController
from .lidar import LidarProcessor

# Navigator tuning
NAV_FWD = 0.10                 # m/s cruise speed
NAV_SLOW_FWD = 0.05            # m/s near obstacles
NAV_ANG_KP = 1.2               # heading P-gain
NAV_WP_REACH = 0.15            # m - waypoint reached
NAV_HEADING_TOL = 0.15         # rad
NAV_GOAL_REACH = 0.25          # m - final goal reached
COSTMAP_INFLATE_RADIUS = 8     # pixels (~0.20m at 0.025 resolution)
NAV_OBSTACLE_STOP = 0.22       # m - emergency stop
NAV_PATH_SIMPLIFY = 3          # skip every N path points


class Navigator:
    """
    Map-based point-to-point navigation.

    Usage:
        nav.load_map()
        nav.navigate_to(x, y)
        # then call nav.tick() at 10 Hz
        # check nav.is_done() or nav.has_failed()
    """

    ST_IDLE = "NAV_IDLE"
    ST_PLANNING = "NAV_PLANNING"
    ST_ROTATING = "NAV_ROTATING"
    ST_DRIVING = "NAV_DRIVING"
    ST_REACHED = "NAV_REACHED"
    ST_FAILED = "NAV_FAILED"

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

        self.state = self.ST_IDLE
        self.status_text = ""

        # Map data
        self.map_grid: Optional[np.ndarray] = None
        self.costmap: Optional[np.ndarray] = None
        self._grid_cells = 0
        self.map_loaded = False

        # Current path
        self.path: List[Tuple[float, float]] = []
        self.path_idx = 0
        self.goal_x = 0.0
        self.goal_y = 0.0

    # ── Map loading ──────────────────────────────────────────────────

    def load_map(self) -> bool:
        """Load saved map and create inflated costmap."""
        np_path = os.path.join(MAP_SAVE_DIR, "map_grid.npy")
        if not os.path.exists(np_path):
            self._log(f"No saved map at {np_path}")
            return False

        self.map_grid = np.load(np_path)
        self._grid_cells = self.map_grid.shape[0]

        # Create costmap: inflate obstacles
        # Binary obstacle mask (occupied = pixels < 50)
        obstacle_mask = (self.map_grid < 50).astype(np.uint8)

        # Dilate obstacles by robot radius
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (COSTMAP_INFLATE_RADIUS * 2 + 1, COSTMAP_INFLATE_RADIUS * 2 + 1))
        inflated = cv2.dilate(obstacle_mask, kernel)

        # Costmap: 0=free, 255=obstacle/inflated
        self.costmap = inflated * 255
        self.map_loaded = True

        self._log(f"Map loaded: {self._grid_cells}x{self._grid_cells}, "
                  f"obstacles inflated by {COSTMAP_INFLATE_RADIUS}px")
        return True

    # ── World <-> Grid conversion ────────────────────────────────────

    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int((wx + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)
        gy = int((wy + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = gx * MAP_RESOLUTION - MAP_ORIGIN_OFFSET
        wy = gy * MAP_RESOLUTION - MAP_ORIGIN_OFFSET
        return wx, wy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self._grid_cells and 0 <= gy < self._grid_cells

    # ── A* path planning ─────────────────────────────────────────────

    def _plan_path(self, sx: float, sy: float,
                   gx: float, gy: float) -> List[Tuple[float, float]]:
        """A* on the costmap. Returns list of world (x,y) waypoints."""
        if self.costmap is None:
            return []

        start = self._world_to_grid(sx, sy)
        goal = self._world_to_grid(gx, gy)

        # Clamp to bounds
        start = (max(0, min(self._grid_cells - 1, start[0])),
                 max(0, min(self._grid_cells - 1, start[1])))
        goal = (max(0, min(self._grid_cells - 1, goal[0])),
                max(0, min(self._grid_cells - 1, goal[1])))

        # If goal is in obstacle, find nearest free cell
        if self.costmap[goal[1], goal[0]] > 0:
            goal = self._nearest_free(goal[0], goal[1])
            if goal is None:
                return []

        # A* search
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        def heuristic(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        # 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        diag_cost = 1.414

        iterations = 0
        max_iterations = self._grid_cells * self._grid_cells

        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path_grid = []
                while current in came_from:
                    path_grid.append(current)
                    current = came_from[current]
                path_grid.reverse()

                # Convert to world coordinates, simplify
                path_world = []
                for i, (px, py) in enumerate(path_grid):
                    if i % NAV_PATH_SIMPLIFY == 0 or i == len(path_grid) - 1:
                        path_world.append(self._grid_to_world(px, py))

                # Always end with exact goal
                path_world.append((gx, gy))
                return path_world

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if not self._in_bounds(nx, ny):
                    continue
                if self.costmap[ny, nx] > 0:
                    continue

                cost = diag_cost if (dx != 0 and dy != 0) else 1.0
                tentative = g_score[current] + cost

                if (nx, ny) not in g_score or tentative < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative
                    f = tentative + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

        self._log("A* failed to find path!")
        return []

    def _nearest_free(self, gx: int, gy: int) -> Optional[Tuple[int, int]]:
        """Find nearest free cell to (gx, gy)."""
        for r in range(1, 30):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if (self._in_bounds(nx, ny) and
                            self.costmap[ny, nx] == 0):
                        return (nx, ny)
        return None

    # ── Navigation control ───────────────────────────────────────────

    def navigate_to(self, x: float, y: float) -> bool:
        """Start navigating to world position (x, y)."""
        if not self.map_loaded:
            self._log("Cannot navigate -- no map loaded!")
            return False

        self.goal_x = x
        self.goal_y = y

        ox, oy, _ = self._get_odom()
        self.path = self._plan_path(ox, oy, x, y)

        if not self.path:
            self._log(f"No path found to ({x:.2f}, {y:.2f})")
            self.state = self.ST_FAILED
            return False

        self.path_idx = 0
        self.state = self.ST_ROTATING
        self._log(f"Path planned: {len(self.path)} waypoints to "
                  f"({x:.2f}, {y:.2f})")
        return True

    def cancel(self):
        self.motion.stop()
        self.state = self.ST_IDLE
        self.path = []

    def is_done(self) -> bool:
        return self.state == self.ST_REACHED

    def has_failed(self) -> bool:
        return self.state == self.ST_FAILED

    def is_active(self) -> bool:
        return self.state in (self.ST_ROTATING, self.ST_DRIVING,
                              self.ST_PLANNING)

    def tick(self):
        """Call at 10 Hz while navigating."""
        if self.state not in (self.ST_ROTATING, self.ST_DRIVING):
            return

        if self.path_idx >= len(self.path):
            self.motion.stop()
            self.state = self.ST_REACHED
            self.status_text = "Navigation: goal reached!"
            return

        x, y, yaw = self._get_odom()
        tx, ty = self.path[self.path_idx]
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx * dx + dy * dy)
        target_yaw = math.atan2(dy, dx)

        # Check if at final goal
        goal_dist = math.sqrt(
            (self.goal_x - x) ** 2 + (self.goal_y - y) ** 2)
        if goal_dist < NAV_GOAL_REACH:
            self.motion.stop()
            self.state = self.ST_REACHED
            self.status_text = "Navigation: goal reached!"
            self._log(self.status_text)
            return

        # Advance to next waypoint if close
        if dist < NAV_WP_REACH and self.path_idx < len(self.path) - 1:
            self.path_idx += 1
            return

        yaw_err = target_yaw - yaw
        while yaw_err > math.pi:
            yaw_err -= 2 * math.pi
        while yaw_err < -math.pi:
            yaw_err += 2 * math.pi

        front_d = self.lidar.get_front_distance()

        # Emergency obstacle stop
        if front_d < NAV_OBSTACLE_STOP:
            self.motion.stop()
            # Try replanning
            self.path = self._plan_path(x, y, self.goal_x, self.goal_y)
            if self.path:
                self.path_idx = 0
                self.state = self.ST_ROTATING
                self.status_text = "Navigation: replanning around obstacle..."
                self._log(self.status_text)
            else:
                self.state = self.ST_FAILED
                self.status_text = "Navigation: blocked, no path!"
            return

        if self.state == self.ST_ROTATING:
            if abs(yaw_err) < NAV_HEADING_TOL:
                self.state = self.ST_DRIVING
            else:
                ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL,
                           NAV_ANG_KP * yaw_err))
                self.motion.publish(lx=0.0, az=ang)
                self.status_text = (
                    f"Nav: rotating {math.degrees(yaw_err):+.0f} deg "
                    f"[WP {self.path_idx + 1}/{len(self.path)}]")

        elif self.state == self.ST_DRIVING:
            if abs(yaw_err) > 0.5:
                self.state = self.ST_ROTATING
                self.motion.stop()
                return

            fwd = NAV_FWD
            if front_d < 0.50:
                fwd = NAV_SLOW_FWD

            ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL,
                       NAV_ANG_KP * yaw_err))
            self.motion.publish(lx=fwd, az=ang)
            self.status_text = (
                f"Nav: WP {self.path_idx + 1}/{len(self.path)} "
                f"d={dist:.2f}m  lidar={front_d:.2f}m")

    # ── Visualization ────────────────────────────────────────────────

    def get_nav_map_image(self, odom_x: float, odom_y: float,
                          odom_yaw: float) -> Optional[np.ndarray]:
        """Return map image with path and robot overlaid."""
        if self.map_grid is None:
            return None

        # Base map in colour
        img = cv2.cvtColor(self.map_grid, cv2.COLOR_GRAY2BGR)

        # Draw inflated obstacles in dark red
        if self.costmap is not None:
            inflate_only = (self.costmap > 0) & (self.map_grid >= 50)
            img[inflate_only] = (40, 40, 120)

        # Draw planned path in cyan
        if self.path:
            for i in range(len(self.path) - 1):
                p1 = self._world_to_grid(*self.path[i])
                p2 = self._world_to_grid(*self.path[i + 1])
                cv2.line(img, p1, p2, (255, 255, 0), 1)

            # Goal marker
            gp = self._world_to_grid(self.goal_x, self.goal_y)
            cv2.circle(img, gp, 6, (0, 0, 255), 2)
            cv2.drawMarker(img, gp, (0, 0, 255),
                           cv2.MARKER_CROSS, 10, 2)

        # Robot position
        rx = int((odom_x + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)
        ry = int((odom_y + MAP_ORIGIN_OFFSET) / MAP_RESOLUTION)
        if 0 <= rx < self._grid_cells and 0 <= ry < self._grid_cells:
            cv2.circle(img, (rx, ry), 5, (0, 255, 0), -1)
            arrow_len = 12
            ax = int(rx + arrow_len * math.cos(odom_yaw))
            ay = int(ry + arrow_len * math.sin(odom_yaw))
            cv2.arrowedLine(img, (rx, ry), (ax, ay), (0, 200, 255), 2,
                            tipLength=0.4)

        return img
