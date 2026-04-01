#!/usr/bin/env python3
"""
Pick and Place script for OpenMANIPULATOR-X in Gazebo.
Picks up 'unit_box_0' and drops it at a specified location.

Cube location (from world state):
  x ≈ 0.002, y ≈ 0.294, z ≈ 0.092  (very small cube ~2.3cm x 4cm x 6.1cm)

Uses:
  - MoveIt2 Python interface for arm motion planning
  - /goal_tool_control service for gripper
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from open_manipulator_msgs.srv import SetJointPosition, SetKinematicsPose
from open_manipulator_msgs.msg import JointPosition, KinematicsPose

from geometry_msgs.msg import Pose
import time
import sys


class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        self.callback_group = ReentrantCallbackGroup()

        # ---------- Service Clients ----------
        # Joint-space goal
        self.joint_client = self.create_client(
            SetJointPosition,
            '/goal_joint_space_path',
            callback_group=self.callback_group,
        )
        # Task-space (Cartesian) goal
        self.task_client = self.create_client(
            SetKinematicsPose,
            '/goal_task_space_path',
            callback_group=self.callback_group,
        )
        # Gripper (tool) control
        self.gripper_client = self.create_client(
            SetJointPosition,
            '/goal_tool_control',
            callback_group=self.callback_group,
        )

        self.get_logger().info('Waiting for OpenMANIPULATOR-X services...')
        self.joint_client.wait_for_service(timeout_sec=10.0)
        self.task_client.wait_for_service(timeout_sec=10.0)
        self.gripper_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('All services available!')

        # ---- Cube location (from Gazebo world state) ----
        # unit_box_0 state pose: x=0.001696, y=0.293801, z=0.091739
        # The manipulator base is at (0, 0, 0) in the robot frame.
        # Adjust these if your robot base is not at the world origin.
        self.cube_x = 0.002
        self.cube_y = 0.294
        self.cube_z = 0.092   # top of cube ≈ z + half-height

        # ---- Drop location (change as desired) ----
        self.drop_x = 0.20
        self.drop_y = -0.15
        self.drop_z = 0.10

        # ---- Gripper limits (radians) ----
        self.GRIPPER_OPEN = 0.01     # fully open
        self.GRIPPER_CLOSE = -0.01   # closed on small cube

        # ---- Motion path time (seconds) ----
        self.PATH_TIME = 2.0

    # ================================================================
    #  Low-level helpers
    # ================================================================

    def send_joint_goal(self, joint_angles: list, path_time: float = 2.0):
        """Send a joint-space goal. joint_angles = [j1, j2, j3, j4] in rad."""
        req = SetJointPosition.Request()
        req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4']
        req.joint_position.position = joint_angles
        req.path_time = path_time

        future = self.joint_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=path_time + 5.0)
        if future.result() is not None:
            self.get_logger().info(f'Joint goal sent: {joint_angles}')
        else:
            self.get_logger().error('Joint goal service call failed')
        time.sleep(path_time + 0.5)

    def send_task_goal(self, x: float, y: float, z: float, path_time: float = 2.0):
        """Send a Cartesian (task-space) goal for the end-effector."""
        req = SetKinematicsPose.Request()
        req.end_effector_name = 'gripper'
        req.kinematics_pose.pose.position.x = x
        req.kinematics_pose.pose.position.y = y
        req.kinematics_pose.pose.position.z = z
        # Keep gripper pointing down (default orientation)
        req.kinematics_pose.pose.orientation.w = 1.0
        req.path_time = path_time

        future = self.task_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=path_time + 5.0)
        if future.result() is not None:
            self.get_logger().info(f'Task goal sent: ({x:.3f}, {y:.3f}, {z:.3f})')
        else:
            self.get_logger().error('Task goal service call failed')
        time.sleep(path_time + 0.5)

    def set_gripper(self, position: float, path_time: float = 1.0):
        """Open or close the gripper. position in radians."""
        req = SetJointPosition.Request()
        req.joint_position.joint_name = ['gripper']
        req.joint_position.position = [position]
        req.path_time = path_time

        future = self.gripper_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=path_time + 5.0)
        state = 'OPEN' if position > 0 else 'CLOSED'
        if future.result() is not None:
            self.get_logger().info(f'Gripper {state} ({position:.3f} rad)')
        else:
            self.get_logger().error('Gripper service call failed')
        time.sleep(path_time + 0.5)

    # ================================================================
    #  High-level pick & place
    # ================================================================

    def go_home(self):
        """Move to the home (init) position."""
        self.get_logger().info('>>> Moving to HOME position')
        self.send_joint_goal([0.0, -1.05, 0.35, 0.70], self.PATH_TIME)

    def pick(self):
        """Pick up the cube at the known location."""
        self.get_logger().info('========== PICK SEQUENCE ==========')

        # 1. Open gripper
        self.get_logger().info('Step 1: Opening gripper')
        self.set_gripper(self.GRIPPER_OPEN)

        # 2. Move above the cube (approach from above)
        above_z = self.cube_z + 0.08  # 8 cm above cube
        self.get_logger().info(f'Step 2: Moving above cube ({self.cube_x}, {self.cube_y}, {above_z})')
        self.send_task_goal(self.cube_x, self.cube_y, above_z, self.PATH_TIME)

        # 3. Descend to grasp height
        grasp_z = self.cube_z + 0.01  # just above base of cube
        self.get_logger().info(f'Step 3: Descending to grasp ({self.cube_x}, {self.cube_y}, {grasp_z})')
        self.send_task_goal(self.cube_x, self.cube_y, grasp_z, self.PATH_TIME)

        # 4. Close gripper
        self.get_logger().info('Step 4: Closing gripper')
        self.set_gripper(self.GRIPPER_CLOSE)
        time.sleep(1.0)  # let gripper settle

        # 5. Lift the cube
        lift_z = self.cube_z + 0.12
        self.get_logger().info(f'Step 5: Lifting cube to z={lift_z}')
        self.send_task_goal(self.cube_x, self.cube_y, lift_z, self.PATH_TIME)

    def place(self):
        """Place the cube at the drop location."""
        self.get_logger().info('========== PLACE SEQUENCE ==========')

        # 1. Move above drop location
        above_z = self.drop_z + 0.08
        self.get_logger().info(f'Step 1: Moving above drop ({self.drop_x}, {self.drop_y}, {above_z})')
        self.send_task_goal(self.drop_x, self.drop_y, above_z, self.PATH_TIME)

        # 2. Lower to drop height
        self.get_logger().info(f'Step 2: Lowering to drop height ({self.drop_x}, {self.drop_y}, {self.drop_z})')
        self.send_task_goal(self.drop_x, self.drop_y, self.drop_z, self.PATH_TIME)

        # 3. Open gripper (release cube)
        self.get_logger().info('Step 3: Releasing cube')
        self.set_gripper(self.GRIPPER_OPEN)
        time.sleep(1.0)

        # 4. Retreat upward
        self.get_logger().info('Step 4: Retreating upward')
        self.send_task_goal(self.drop_x, self.drop_y, above_z, self.PATH_TIME)

    def execute(self):
        """Full pick-and-place sequence."""
        self.get_logger().info('===== Starting Pick and Place =====')

        self.go_home()
        time.sleep(1.0)

        self.pick()
        time.sleep(1.0)

        self.place()
        time.sleep(1.0)

        self.go_home()

        self.get_logger().info('===== Pick and Place Complete! =====')


def main():
    rclpy.init()
    node = PickAndPlace()
    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
