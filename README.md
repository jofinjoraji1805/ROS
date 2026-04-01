# TurtleBot3 Waffle Pi + OpenMANIPULATOR-X — Pick & Place

## Prerequisites

### 1. Install TurtleBot3 Manipulation packages

```bash
# Core dependencies
sudo apt install ros-humble-dynamixel-sdk ros-humble-ros2-control \
  ros-humble-ros2-controllers ros-humble-gripper-controllers \
  ros-humble-hardware-interface ros-humble-xacro ros-humble-moveit*

# Clone turtlebot3_manipulation
cd ~/colcon_ws/src/
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_manipulation.git

# Clone turtlebot3_simulations (for Gazebo)
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

# Build
cd ~/colcon_ws && colcon build --symlink-install
source install/setup.bash
```

### 2. Set TurtleBot3 model (add to ~/.bashrc)

```bash
echo 'export TURTLEBOT3_MODEL=waffle_pi' >> ~/.bashrc
source ~/.bashrc
```

---

## Install this package

```bash
cp -r tb3_pick_place ~/colcon_ws/src/
cd ~/colcon_ws
colcon build --packages-select tb3_pick_place --symlink-install
source install/setup.bash
```

---

## Run Pick & Place

### Terminal 1 — Launch Gazebo + Robot

```bash
export TURTLEBOT3_MODEL=waffle_pi
source ~/colcon_ws/install/setup.bash
ros2 launch tb3_pick_place pick_place.launch.py
```

Wait until you see all controllers loaded:
- `joint_state_broadcaster`
- `arm_controller`
- `gripper_controller`

### Terminal 2 — Verify services

```bash
source ~/colcon_ws/install/setup.bash

# Verify controllers
ros2 control list_controllers

# Verify Gazebo state services (required for virtual attach)
ros2 service list | grep entity
# Must show: /get_entity_state and /set_entity_state
```

### Terminal 3 — Run pick-and-place

```bash
source ~/colcon_ws/install/setup.bash
python3 ~/colcon_ws/src/tb3_pick_place/scripts/pick_place_tb3.py
```

---

## Alternative: Use official launch + custom world separately

If the wrapped launch doesn't work, launch manually:

```bash
# Terminal 1: Gazebo with custom world
export TURTLEBOT3_MODEL=waffle_pi
source ~/colcon_ws/install/setup.bash
WORLD=$(ros2 pkg prefix tb3_pick_place)/share/tb3_pick_place/worlds/pick_place_world.world
ros2 launch turtlebot3_manipulation_gazebo gazebo.launch.py world:=$WORLD

# Terminal 2: Pick and place
source ~/colcon_ws/install/setup.bash
python3 ~/colcon_ws/src/tb3_pick_place/scripts/pick_place_tb3.py
```

---

## How It Works

1. **Discovery**: Script queries Gazebo to find the robot entity name and gripper link
2. **Position sensing**: Gets arm base (link1) and box world positions from Gazebo
3. **Dynamic IK**: Computes joint angles relative to the actual arm base position
4. **Virtual attach**: After closing gripper, continuously teleports the box to follow the gripper link (Gazebo Classic can't reliably grip with prismatic joints)
5. **Release**: Stops teleporting at drop location, box falls naturally

## Troubleshooting

- **"arm_controller NOT found"**: Controllers haven't loaded yet. Wait longer or check `ros2 control list_controllers`
- **"get_entity_state NOT found"**: The `libgazebo_ros_state.so` plugin is missing. It's included in the world file — make sure you're using the custom world
- **"Cannot find robot entity"**: The Gazebo model name doesn't match. Run `ros2 service call /get_entity_state gazebo_msgs/srv/GetEntityState "{name: 'YOUR_NAME'}"` to test names
- **"IK FAILED"**: Box is too far from arm. Move table closer or reposition the turtlebot
- **Robot not visible**: Set `GAZEBO_MODEL_PATH` to include turtlebot3 mesh directories
