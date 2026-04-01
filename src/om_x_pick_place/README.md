# OpenMANIPULATOR-X Pick & Place – Setup & Terminal Commands

## Package Structure

```
omx_pick_place/
├── package.xml
├── setup.py / setup.cfg
├── resource/omx_pick_place
├── omx_pick_place/
│   ├── __init__.py
│   └── pick_and_place.py        ← main pick-and-place node
├── launch/
│   └── omx_pick_place.launch.py ← Gazebo + robot + controllers
├── worlds/
│   └── pick_place.world         ← walls + table + box + drop target
├── config/
│   └── controllers.yaml         ← ros2_control config (fallback)
└── scripts/
    └── pick_and_place.py        ← standalone copy
```

---

## Prerequisites (one-time install if missing)

```bash
cd ~/colcon_ws/src

# ROBOTIS packages (ROS 2 Humble)
git clone -b humble-devel https://github.com/ROBOTIS-GIT/open_manipulator_x.git
git clone -b humble-devel https://github.com/ROBOTIS-GIT/open_manipulator_x_simulations.git
git clone -b humble-devel https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble-devel https://github.com/ROBOTIS-GIT/dynamixel-workbench.git
git clone -b humble-devel https://github.com/ROBOTIS-GIT/robotis_manipulator.git
git clone -b humble-devel https://github.com/ROBOTIS-GIT/open_manipulator_msgs.git

# System deps
sudo apt update && sudo apt install -y \
  ros-humble-gazebo-ros ros-humble-gazebo-ros2-control \
  ros-humble-controller-manager ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller ros-humble-position-controllers \
  ros-humble-robot-state-publisher ros-humble-xacro \
  ros-humble-control-msgs ros-humble-trajectory-msgs

cd ~/colcon_ws && colcon build --symlink-install && source install/setup.bash
```

---

## Step 1 – Copy package & build

```bash
cp -r omx_pick_place ~/colcon_ws/src/
cd ~/colcon_ws
colcon build --packages-select omx_pick_place --symlink-install
source install/setup.bash
```

## Step 2 – Launch Gazebo + Robot + Controllers

```bash
# ─── Terminal 1 ───
source ~/colcon_ws/install/setup.bash
ros2 launch omx_pick_place omx_pick_place.launch.py
```

Wait ~10 s for Gazebo to load. Verify controllers:

```bash
# ─── Terminal 2 ───
source ~/colcon_ws/install/setup.bash
ros2 control list_controllers
```

Expected output:
```
joint_state_broadcaster  active
arm_controller           active
gripper_controller       active
```

## Step 3 – Run Pick & Place

```bash
# ─── Terminal 3 ───
source ~/colcon_ws/install/setup.bash
ros2 run omx_pick_place pick_and_place
```

### Sequence (9 steps)

| Step | Action | Description |
|------|--------|-------------|
| 1 | HOME | Home position, open gripper |
| 2 | PRE_PICK | Above unit_box_0 |
| 3 | PICK | Descend to grasp |
| 4 | CLOSE | Close gripper |
| 5 | LIFT | Lift box |
| 6 | PRE_DROP | Rotate 180° to drop zone |
| 7 | DROP | Lower to release |
| 8 | OPEN | Release box on green target |
| 9 | RETRACT | Return home |

## Alternative: Run directly without building

```bash
source ~/colcon_ws/install/setup.bash
python3 ~/colcon_ws/src/omx_pick_place/scripts/pick_and_place.py
```

## Quick Manual Test

```bash
# Move arm
ros2 action send_goal /arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [joint1,joint2,joint3,joint4], \
  points: [{positions: [0.5,-0.5,0.3,0.5], time_from_start: {sec: 3}}]}}"

# Open gripper
ros2 action send_goal /gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand \
  "{command: {position: 0.019, max_effort: 5.0}}"
```

## Tuning

Joint waypoints in `pick_and_place.py` are approximate IK. To fine-tune:
1. Check object pose in Gazebo (right-click → Properties)
2. Modify waypoint constants (PRE_PICK, PICK, PRE_DROP, DROP)
3. Or use MoveIt for precise IK:
   ```bash
   ros2 launch open_manipulator_x_moveit_config moveit.launch.py use_sim:=true
   ```

## World Layout

```
              Wall_20 (top)
  ┌─────────────────────────────────┐
  │                                 │
  │   drop_target    [table]        │
  │   (green disc)   ┌──────┐      │
  │   (-0.2, 0)      │ OMX  │      │ Wall_19
  │                   │(0,0) │      │
  │   unit_box_0     └──────┘      │
  │   (red, 0.2, 0)                │
  │              [Door]             │
  └─────────────────────────────────┘
              Wall_18 (bottom)
```
