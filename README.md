# TurtleBot3 Waffle Pi + OpenMANIPULATOR-X — Autonomous Pick & Place

Autonomous mobile goods delivery system using TurtleBot3 with manipulator. The robot navigates a room, detects colored cubes using YOLO, picks them from tables, and delivers them to matching drop zones, then returns to a charging dock.

## Features

- **YOLO Object Detection**: Detects red, green, blue cubes and drop zones using a custom-trained YOLOv8 model
- **Visual Servo Control**: Aligns and approaches cubes using continuous YOLO-based visual feedback
- **L-Formation Approach**: When a cube is off-center, the robot performs an L-shaped lateral alignment maneuver before approaching the table perpendicularly
- **LiDAR-Based Alignment**: Uses LiDAR slope fitting to align perpendicular to the table surface
- **IFRA Link Attacher**: Reliable grasp using Gazebo fixed-joint attachment (physics grip alone is unreliable for small objects)
- **Contact Sensor Grip**: Gripper fingers detect cube contact before triggering the attach
- **A* Path Planning**: Map-based navigation with obstacle avoidance using a pre-built SLAM map
- **Charging Dock**: YOLO-guided visual servo docking after all tasks complete
- **GUI Dashboard**: PyQt5 interface with status bar, nav map, and mission control buttons

## World Layout

| Location | Object | Position |
|----------|--------|----------|
| Table 1 (top-left) | Red cube | (-1.0, 1.5) |
| Table 2 (middle-left) | Green cube | (-1.0, 0.0) |
| Table 3 (bottom-left) | Blue cube | (-1.0, -1.5) |
| Drop zone (top-right) | Red zone | (2.0, 1.5) |
| Drop zone (middle-right) | Green zone | (2.0, 0.0) |
| Drop zone (bottom-right) | Blue zone | (2.0, -1.5) |
| Charging dock | Dock | (0.5, -3.1) |

All cubes and drop zones have black borders for visual clarity.

## Prerequisites

```bash
# Core dependencies
sudo apt install ros-humble-dynamixel-sdk ros-humble-ros2-control \
  ros-humble-ros2-controllers ros-humble-gripper-controllers \
  ros-humble-hardware-interface ros-humble-xacro

# Python dependencies
pip install ultralytics PyQt5 numpy

# Clone turtlebot3_manipulation
cd ~/colcon_ws/src/
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_manipulation.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
```

## Install

```bash
cd ~/colcon_ws
colcon build --symlink-install
source install/setup.bash
```

## Launch

Single command launches everything (Gazebo, robot, YOLO node, GUI):

```bash
export DISPLAY=:1
export TURTLEBOT3_MODEL=waffle_pi
source /opt/ros/humble/setup.bash
source /usr/share/gazebo-11/setup.bash
source ~/colcon_ws/install/setup.bash
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/home/jofin/colcon_ws/install/ros2_linkattacher/lib
ros2 launch tb3_pick_place pick_place.launch.py
```

Or use the launch script:

```bash
bash ~/launch.sh
```

## Usage

1. Wait for Gazebo and the GUI to open
2. Click **BUILD MAP** to auto-explore and build a SLAM map
3. Click **START MISSION** to begin autonomous pick-and-place
4. The robot will:
   - Navigate to each table using A* path planning
   - Detect the cube using YOLO
   - Perform L-formation alignment if the cube is off-center
   - Pick the cube using the manipulator arm
   - Navigate to the matching drop zone
   - Place the cube in the drop zone
   - Return to the charging dock after all tasks complete

## State Flow

```
NAV_TO_CUBE -> SEARCH -> ALIGN -> LATERAL_ALIGN -> APPROACH -> ALIGN_TABLE -> PICK
-> BACKUP -> NAV_TO_ZONE -> SEARCH_DROP -> APPROACH_DROP -> PLACE -> BACKUP_DROP
-> NEXT -> ... -> RETURN_HOME -> SEARCH_DOCK -> ALIGN_DOCK -> APPROACH_DOCK -> PARKED
```

## Key Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| YOLO_MODEL | yolomodel/best.pt | Custom YOLOv8 model (7 classes) |
| TABLE_STOP_DISTANCE | 0.23m | LiDAR distance to stop at table |
| ALIGN_CENTER_PX | 10px | Pixel tolerance for cube centering |
| LATERAL_MIN_ANGLE | 0.08 rad | Min angle to trigger L-formation |
| GRIPPER_OPEN | 0.019 | Gripper open position |
| GRIPPER_CLOSE | -0.015 | Gripper close position |

## YOLO Classes

| ID | Class |
|----|-------|
| 0 | red_cube |
| 1 | green_cube |
| 2 | blue_cube |
| 3 | red_drop_zone |
| 4 | green_drop_zone |
| 5 | blue_drop_zone |
| 6 | charging_dock |

## Troubleshooting

- **gzclient crashes on launch**: Gazebo rendering bug with delayed start. The launch file handles this with a timed delay.
- **Camera not rendering**: Ensure `source /usr/share/gazebo-11/setup.bash` is run before launch.
- **Cube not attaching**: Verify the IFRA LinkAttacher plugin is loaded. Check `GAZEBO_PLUGIN_PATH` includes the `ros2_linkattacher/lib` directory.
- **Robot approaching at angle**: The L-formation alignment corrects for off-center cubes. Check `LATERAL_MIN_ANGLE` threshold.
