#!/usr/bin/env python3
"""
Launch: TurtleBot3 Waffle Pi + OpenMANIPULATOR-X in custom world with pick-place objects.

Uses the official turtlebot3_manipulation_gazebo base launch for robot description
and controllers, but launches gzserver/gzclient explicitly to control startup order.
gzclient is delayed to avoid crashing before gzserver rendering initialises.
"""
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, SetEnvironmentVariable, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('tb3_pick_place')
    world_file = os.path.join(pkg_share, 'worlds', 'pick_place_world.world')
    tb3_manip_gazebo = get_package_share_directory('turtlebot3_manipulation_gazebo')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    set_tb3_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL', value='waffle_pi')

    # Gazebo resource path (needed for gzserver camera rendering / OGRE shaders)
    gazebo_resource = os.path.join('/usr', 'share', 'gazebo-11')
    set_gazebo_resource = SetEnvironmentVariable(
        name='GAZEBO_RESOURCE_PATH',
        value=os.environ.get('GAZEBO_RESOURCE_PATH', gazebo_resource))
    set_gazebo_model = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=os.environ.get('GAZEBO_MODEL_PATH',
                             os.path.join(gazebo_resource, 'models')))

    # Robot description + controllers (upstream base.launch.py)
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_manip_gazebo, 'launch', 'base.launch.py')),
        launch_arguments={'start_rviz': 'false', 'use_sim': 'true'}.items())

    # gzserver
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_file, 'verbose': 'false'}.items())

    # gzclient delayed
    gzclient = TimerAction(period=10.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')))])

    # Spawn robot
    spawn_robot = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'turtlebot3_manipulation_system',
            '-x', '0.5', '-y', '-3.0', '-z', '0.01',
            '-R', '0.00', '-P', '0.00', '-Y', '1.5708'],
        output='screen')

    # Pick-place node (delayed for controllers)
    pick_place_node = TimerAction(period=18.0, actions=[
        Node(
            package='tb3_pick_place', executable='yolo_pick_place',
            name='yolo_pick_place', output='screen',
            additional_env={'DISPLAY': os.environ.get('DISPLAY', ':1')})])

    return LaunchDescription([
        set_tb3_model,
        set_gazebo_resource,
        set_gazebo_model,
        base_launch,
        gzserver,
        gzclient,
        spawn_robot,
        pick_place_node,
    ])
