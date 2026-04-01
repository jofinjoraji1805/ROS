"""
Launch: Gazebo world + OpenMANIPULATOR-X + ros2_control controllers.

Packages required in ~/colcon_ws:
  - open_manipulator_x_description
  - open_manipulator_x_gazebo  (or equivalent with ros2_control config)
  - gazebo_ros, gazebo_ros2_control
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_omx_desc = get_package_share_directory('open_manipulator_x_description')

    world_file = os.path.join(
        os.path.expanduser('~'), 'colcon_ws', 'src',
        'omx_pick_place', 'worlds', 'pick_place.world'
    )

    xacro_file = os.path.join(
        pkg_omx_desc, 'urdf', 'open_manipulator_x_robot.urdf.xacro'
    )

    robot_description = Command([
        FindExecutable(name='xacro'), ' ', xacro_file,
        ' use_sim:=true',
    ])

    # ── Gazebo ─────────────────────────────────────────────
    gz_server = ExecuteProcess(
        cmd=['gzserver', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             world_file],
        output='screen',
    )
    gz_client = ExecuteProcess(cmd=['gzclient'], output='screen')

    # ── Robot State Publisher ──────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        output='screen',
    )

    # ── Spawn robot on table (z=0.40) ─────────────────────
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'open_manipulator_x',
            '-x', '0.0', '-y', '0.0', '-z', '0.40',
        ],
        output='screen',
    )

    # ── Controllers (delayed to let Gazebo settle) ────────
    load_jsb = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller',
             '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen',
    )
    load_arm = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller',
             '--set-state', 'active', 'arm_controller'],
        output='screen',
    )
    load_grip = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller',
             '--set-state', 'active', 'gripper_controller'],
        output='screen',
    )

    return LaunchDescription([
        gz_server,
        gz_client,
        rsp,
        spawn,
        TimerAction(period=5.0,  actions=[load_jsb]),
        TimerAction(period=7.0,  actions=[load_arm]),
        TimerAction(period=9.0,  actions=[load_grip]),
    ])
