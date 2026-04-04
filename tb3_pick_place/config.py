#!/usr/bin/env python3
"""
config.py -- All constants, tuning parameters, and arm poses for the
TurtleBot3 + OpenMANIPULATOR-X autonomous pick-and-place system.

Two-phase operation:
  Phase 1 (MAPPING): Auto-explore room, build SLAM map, save to disk
  Phase 2 (MISSION): Load map, navigate with A*, pick and place cubes
"""

# ── YOLO ──────────────────────────────────────────────────────────────
YOLO_MODEL = "/home/jofin/colcon_ws/src/tb3_pick_place/yolomodel/best.pt"
CONF_THRESH = 0.15

# ── ROS Topics ────────────────────────────────────────────────────────
CAMERA_TOPIC = "/pi_camera/image_raw"
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"
SCAN_TOPIC = "/scan"
ARM_TRAJECTORY_TOPIC = "/arm_controller/joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

# ── LIDAR / Obstacle avoidance ───────────────────────────────────────
LIDAR_FRONT_HALF_ANGLE = 0.35     # rad (~20 deg each side)
TABLE_STOP_DISTANCE = 0.23        # m - stop further back, align, then final creep
OBSTACLE_SLOW_DISTANCE = 0.40     # m - start slowing
OBSTACLE_SLOW_FACTOR = 0.4

# ── Map (occupancy grid) ────────────────────────────────────────────
MAP_SIZE = 12.0                   # metres (12x12m area)
MAP_RESOLUTION = 0.025            # m/pixel -> 480x480 grid
MAP_ORIGIN_OFFSET = 6.0           # centre offset
MAP_SAVE_DIR = "/home/jofin/colcon_ws/maps"

# ── Exploration waypoints ────────────────────────────────────────────
# Robot spawns at (0.5, -2.0). These waypoints cover the room:
#   Left side (pick tables), right side (drop zones), corridors.
EXPLORE_WAYPOINTS = [
    # Start area -> move north through centre
    (0.5, -1.0),
    (0.5, 0.0),
    (0.5, 1.0),
    (0.5, 2.0),
    # Swing left to pick-table row
    (-0.3, 2.0),
    (-0.3, 1.0),
    (-0.3, 0.0),
    (-0.3, -1.0),
    (-0.3, -2.0),
    # Back to centre, go right to drop-zone row
    (0.5, -2.0),
    (1.3, -2.0),
    (1.3, -1.0),
    (1.3, 0.0),
    (1.3, 1.0),
    (1.3, 2.0),
    # Sweep back through middle
    (0.5, 2.0),
    (0.5, 0.0),
    # Return to start
    (0.5, -2.0),
]

# ── Known positions for navigation ──────────────────────────────────
# L-path approach: robot first drives to staging Y (same Y as table),
# then turns and drives straight -X toward the table.
# Each entry is a list of (x, y) waypoints forming the L-path.
CUBE_NAV_TARGETS = {
    "RED":   [(0.5, -1.5), (0.0, -1.5)],   # stage at y=-1.5, then approach
    "BLUE":  [(0.5, 0.0),  (0.0, 0.0)],    # stage at y=0.0, then approach
    "GREEN": [(0.5, 1.5),  (0.0, 1.5)],    # stage at y=1.5, then approach
}
ZONE_NAV_TARGETS = {
    "RED":   [(0.5, 1.5),  (1.78, 1.5)],    # L-path: right at box back wall (x=1.80)
    "BLUE":  [(0.5, -1.5), (1.78, -1.5)],   # L-path: right at box back wall
    "GREEN": [(0.5, 0.0),  (1.78, 0.0)],    # L-path: right at box back wall
}
# Drop zone box centers (for teleport drop)
ZONE_BOX_CENTER = {
    "RED":   (2.0, 1.5),
    "BLUE":  (2.0, -1.5),
    "GREEN": (2.0, 0.0),
}

# Known table Y-coordinates for odom-based lateral correction during approach
CUBE_TABLE_Y = {
    "RED":   -1.5,
    "BLUE":   0.0,
    "GREEN":  1.5,
}

# Face direction after arriving at nav target (perpendicular to object)
CUBE_FACE_YAW = 3.14159      # face -X (toward tables at x=-1.0)
ZONE_FACE_YAW = 0.0          # face +X (toward baskets at x=2.0)
FACE_YAW_TOLERANCE = 0.03    # rad (~1.7 deg)

# ── Scanning ──────────────────────────────────────────────────────────
SCAN_ANG_VEL = 0.15

# ── Alignment ─────────────────────────────────────────────────────────
ALIGN_CENTER_PX = 10
ALIGN_KP = 1.0
ALIGN_LOST_MAX = 100
MAX_ANG_VEL = 0.25

# ── Approach ──────────────────────────────────────────────────────────
APPROACH_FWD = 0.08
APPROACH_DRIFT_PX = 50
APPROACH_MILD_KP = 0.3
APPROACH_MILD_MAX = 0.06
APPROACH_TIMEOUT = 45.0
CUBE_LOST_TICKS = 80

# ── Adjust ────────────────────────────────────────────────────────────
ADJUST_FWD = 0.020
ADJUST_DURATION = 15.0            # longer creep to get very close

# ── Backup ────────────────────────────────────────────────────────────
BACKUP_VEL = -0.06
BACKUP_PICK_TIME = 3.5
BACKUP_DROP_TIME = 2.5

# ── Velocity ramping ─────────────────────────────────────────────────
VEL_RAMP = 0.02

# ── Drop-zone approach (baskets are ground-level, LIDAR can't see) ───
ZONE_CLOSE_AREA = 0.015
ZONE_APPROACH_TIME = 40.0
ZONE_FINAL_TIME = 16.0           # longer creep -- baskets are invisible to LIDAR
ZONE_FINAL_VEL = 0.04
ZONE_ADJUST_TIME = 10.0

# ── Return home ──────────────────────────────────────────────────────
RETURN_HOME_THRESH = 0.20

# ── Charging dock ────────────────────────────────────────────────────
DOCK_POSITION = (0.5, -2.3)          # (x, y) — waypoint to start dock search
DOCK_FACE_YAW = -1.5708              # face -Y (toward dock) to start search
DOCK_APPROACH_VEL = 0.06             # forward vel during visual servo approach
DOCK_STOP_DISTANCE = 0.30            # LIDAR distance to stop approach and creep
DOCK_CREEP_VEL = 0.04                # slow final creep into dock
DOCK_CREEP_TIME = 4.0                # seconds to creep into final park position
DOCK_ALIGN_CENTER_PX = 15            # centering tolerance (pixels)
DOCK_APPROACH_TIMEOUT = 30.0         # max seconds for approach

# ── Arm joint poses [joint1, joint2, joint3, joint4] ─────────────────
# HOME: arm folded back compactly, out of camera view
ARM_HOME = [0.0, -1.0, 0.3, 0.7]
# READY: arm raised, preparing to extend toward cube
ARM_READY = [0.0, -0.6, 0.3, 0.3]
# PRE_PICK: arm extended forward, gripper clearly ABOVE the cube
ARM_PRE_PICK = [0.0, 0.20, 0.50, -0.70]
# PICK: fingertips at cube mid-height (wider 30mm cube)
ARM_PICK = [0.0, 0.38, 0.50, -0.88]
# LIFT: raise cube off table
ARM_LIFT = [0.0, -0.1, 0.55, -0.45]

# RED cube: same as default
ARM_PRE_PICK_RED = [0.0, 0.20, 0.50, -0.70]
ARM_PICK_RED = [0.0, 0.38, 0.50, -0.88]
ARM_LIFT_RED = [0.0, -0.1, 0.55, -0.45]
# CARRY: compact carry position, cube held close to body
ARM_CARRY = [0.0, -0.8, 0.2, 0.6]
# Drop into basket: arm extended FORWARD, gripper reaches PAST the wall and INTO the box
ARM_DROP_EXTEND = [0.0, 0.05, 0.05, -0.10]     # arm stretched forward, slightly above wall height
ARM_DROP_OVER = [0.0, 0.25, 0.05, -0.30]       # shoulder tilts down, gripper drops well inside box
ARM_DROP_RETREAT = [0.0, -0.50, 0.30, 0.20]    # pull arm back after release

# ── Gripper ──────────────────────────────────────────────────────────
GRIPPER_OPEN = 0.019
GRIPPER_CLOSE = -0.015            # maximum close — squeeze to cube width
GRIPPER_EFFORT = 200.0            # max effort to squeeze and hold

# ── Manual teleop ────────────────────────────────────────────────────
TELEOP_LINEAR = 0.10
TELEOP_ANGULAR = 0.30

# ── State names ──────────────────────────────────────────────────────
ST_IDLE = "IDLE"
ST_NAV_TO_CUBE = "NAV_TO_CUBE"
ST_DRIVE_TO_CUBE = "DRIVE_TO_CUBE"
ST_SEARCH_OBJECT = "SEARCH_OBJECT"
ST_ALIGN_OBJECT = "ALIGN_OBJECT"
ST_APPROACH_OBJECT = "APPROACH_OBJECT"
ST_ADJUST_POSITION = "ADJUST_POSITION"
ST_ALIGN_TABLE = "ALIGN_TABLE"
ST_PICK_OBJECT = "PICK_OBJECT"
ST_BACKUP_PICK = "BACKUP_PICK"
ST_NAV_TO_ZONE = "NAV_TO_ZONE"
ST_DRIVE_TO_ZONE = "DRIVE_TO_ZONE"
ST_SEARCH_DROP_ZONE = "SEARCH_DROP_ZONE"
ST_APPROACH_DROP = "APPROACH_DROP"
ST_ADJUST_DROP = "ADJUST_DROP"
ST_PLACE_OBJECT = "PLACE_OBJECT"
ST_BACKUP_DROP = "BACKUP_DROP"
ST_NEXT_OBJECT = "NEXT_OBJECT"
ST_RETURN_HOME = "RETURN_HOME"
ST_SEARCH_DOCK = "SEARCH_DOCK"
ST_ALIGN_DOCK = "ALIGN_DOCK"
ST_APPROACH_DOCK = "APPROACH_DOCK"
ST_DOCK_CREEP = "DOCK_CREEP"
ST_PARKED = "PARKED"
ST_DONE = "DONE"

# ── Table alignment (LiDAR parallel) ─────────────────────────────────
ALIGN_TABLE_KP = 1.2
ALIGN_TABLE_SLOPE_THRESH = 0.02
ALIGN_TABLE_TIMEOUT = 15.0

TASK_ORDER = ["RED", "BLUE", "GREEN"]
