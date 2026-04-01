#!/usr/bin/env python3
"""
Launch: Gazebo world + OpenMANIPULATOR-X + ros2_control controllers
Uses open_manipulator_x_description (no _gazebo package needed)
"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():
    # ── Package paths ──
    pkg_pick = get_package_share_directory('om_x_pick_place')
    pkg_desc = get_package_share_directory('open_manipulator_x_description')
    pkg_bringup = get_package_share_directory('open_manipulator_x_bringup')

    world_file = os.path.join(pkg_pick, 'worlds', 'pick_place_world.world')
    controller_yaml = os.path.join(pkg_bringup, 'config', 'gazebo_controller_manager.yaml')

    # ── Process URDF with use_sim:=true ──
    urdf_file = os.path.join(pkg_desc, 'urdf', 'open_manipulator_x_robot.urdf.xacro')
    robot_description_content = xacro.process_file(
        urdf_file,
        mappings={'use_sim': 'true', 'prefix': ''}
    ).toxml()

    robot_description = {'robot_description': robot_description_content}

    # ── Gazebo ──
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world_file,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        output='screen',
    )

    # ── Robot State Publisher ──
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # ── Spawn robot on the table (z=0.06) ──
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'open_manipulator_x',
            '-topic', '/robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.06',
        ],
        output='screen',
    )

    # ── Load controllers (chained with delays) ──
    load_jsb = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen',
    )

    load_arm = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'arm_controller'],
        output='screen',
    )

    load_gripper = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'gripper_controller'],
        output='screen',
    )

    # Wait for Gazebo + spawn, then load controllers in sequence
    delayed_jsb = TimerAction(period=6.0, actions=[load_jsb])

    arm_after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=load_jsb, on_exit=[load_arm])
    )

    gripper_after_arm = RegisterEventHandler(
        OnProcessExit(target_action=load_arm, on_exit=[load_gripper])
    )

    return LaunchDescription([
        gazebo,
        robot_state_pub,
        spawn_robot,
        delayed_jsb,
        arm_after_jsb,
        gripper_after_arm,
    ])
