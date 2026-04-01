#!/usr/bin/env python3
"""
Launch: TurtleBot3 Waffle Pi + OpenMANIPULATOR-X in custom world with pick-place objects.

Uses the official turtlebot3_manipulation_gazebo launch as a base,
but overrides the world file.
"""
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument,
    SetEnvironmentVariable, TimerAction, ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_pick_place')
    world_file = os.path.join(pkg_share, 'worlds', 'pick_place_world.world')

    # TurtleBot3 manipulation Gazebo package
    tb3_manip_gazebo = get_package_share_directory('turtlebot3_manipulation_gazebo')

    # Set TURTLEBOT3_MODEL (required)
    set_tb3_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value='waffle_pi'
    )

    # Include the official turtlebot3_manipulation_gazebo launch
    # Override the world parameter
    # Spawn robot at the START position (away from the table)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_manip_gazebo, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'start_rviz': 'false',
            'use_sim_time': 'true',
            'x_pose': '0.5',
            'y_pose': '-2.0',
            'yaw': '1.5708',
        }.items(),
    )

    # Launch the pick-place GUI node after a delay for controllers to start
    pick_place_node = TimerAction(
        period=15.0,
        actions=[
            Node(
                package='tb3_pick_place',
                executable='yolo_pick_place',
                name='yolo_pick_place',
                output='screen',
                additional_env={'DISPLAY': os.environ.get('DISPLAY', ':1')},
            ),
        ],
    )

    return LaunchDescription([
        set_tb3_model,
        gazebo_launch,
        pick_place_node,
    ])
