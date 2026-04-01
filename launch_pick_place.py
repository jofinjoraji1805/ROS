#!/usr/bin/env python3
"""
Custom launch file to spawn OpenMANIPULATOR-X into your factory.world in Gazebo.
Works with the packages you already have installed.

Usage:
  ros2 launch launch_pick_place.py
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    # ── Paths ────────────────────────────────────────────────────────────
    description_pkg = get_package_share_directory('open_manipulator_x_description')
    world_file = os.path.expanduser('~/colcon_ws/factory.world')

    # ── Robot Description (URDF from xacro) ──────────────────────────────
    xacro_file = os.path.join(description_pkg, 'urdf', 'open_manipulator_x_robot.urdf.xacro')

    # Try common xacro file names
    if not os.path.exists(xacro_file):
        xacro_file = os.path.join(description_pkg, 'urdf', 'open_manipulator_x.urdf.xacro')
    if not os.path.exists(xacro_file):
        xacro_file = os.path.join(description_pkg, 'urdf', 'open_manipulator_x.urdf')

    # Process xacro → URDF string
    if xacro_file.endswith('.xacro'):
        robot_description = xacro.process_file(xacro_file).toxml()
    else:
        with open(xacro_file, 'r') as f:
            robot_description = f.read()

    # ── Robot spawn position (near the cube at ~0.0, 0.29, 0.09) ────────
    # Place the arm base so the cube is within reach
    spawn_x = '0.0'
    spawn_y = '0.0'
    spawn_z = '0.0'

    return LaunchDescription([
        # 1. Launch Gazebo with factory world
        ExecuteProcess(
            cmd=[
                'gazebo', '--verbose',
                '-s', 'libgazebo_ros_factory.so',
                '-s', 'libgazebo_ros_init.so',
                world_file
            ],
            output='screen',
        ),

        # 2. Publish robot description to /robot_description
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }],
        ),

        # 3. Spawn the robot into Gazebo (delayed 3s to let Gazebo start)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-topic', 'robot_description',
                        '-entity', 'open_manipulator_x',
                        '-x', spawn_x,
                        '-y', spawn_y,
                        '-z', spawn_z,
                    ],
                    output='screen',
                ),
            ],
        ),
    ])
