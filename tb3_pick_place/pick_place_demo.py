#!/usr/bin/env python3

import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand

from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


class TB3PickPlaceDemo(Node):
    def __init__(self):
        super().__init__('tb3_pick_place_demo')

        self.arm_action = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        self.gripper_action = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        self.set_entity_state_cli = self.create_client(
            SetEntityState,
            '/gazebo/set_entity_state'
        )

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        self.HOME = [0.0, -1.05, 0.35, 0.70]
        self.PRE_GRASP = [0.0, -0.78, 0.30, 1.05]
        self.GRASP = [0.0, -0.92, 0.52, 0.72]
        self.LIFT = [0.0, -0.55, 0.10, 1.15]
        self.PRE_PLACE = [-1.00, -0.62, 0.22, 1.18]
        self.PLACE = [-1.00, -0.84, 0.42, 0.86]
        self.RETREAT = [-1.00, -0.55, 0.10, 1.10]

        self.CUBE_NAME = 'pick_cube'

        self.PICK_POSE = (0.32, 0.00, 0.165)
        self.LIFT_POSE = (0.25, 0.00, 0.26)
        self.PLACE_POSE = (0.18, -0.26, 0.165)

    def wait_for_interfaces(self):
        self.get_logger().info('Waiting for arm controller...')
        self.arm_action.wait_for_server()

        self.get_logger().info('Waiting for gripper controller...')
        self.gripper_action.wait_for_server()

        self.get_logger().info('Waiting for Gazebo set_entity_state service...')
        while not self.set_entity_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('...still waiting for /gazebo/set_entity_state')

        self.get_logger().info('All required interfaces are ready.')

    def move_arm(self, positions: List[float], duration_sec: float = 3.0):
        goal_msg = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(
            sec=int(duration_sec),
            nanosec=int((duration_sec % 1.0) * 1e9)
        )
        traj.points.append(point)

        goal_msg.trajectory = traj

        self.get_logger().info(f'Moving arm to: {positions}')
        send_goal_future = self.arm_action.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            raise RuntimeError('Arm trajectory goal rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        time.sleep(0.5)

    def command_gripper(self, position: float, max_effort: float = 10.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self.get_logger().info(f'Gripper command position={position:.4f}')
        send_goal_future = self.gripper_action.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            raise RuntimeError('Gripper goal rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        time.sleep(1.0)

    def set_cube_pose(self, x: float, y: float, z: float):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = self.CUBE_NAME
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.reference_frame = 'world'
        req.state = state

        future = self.set_entity_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None or not response.success:
            raise RuntimeError('Failed to set cube pose')

    def run_demo(self):
        self.wait_for_interfaces()

        self.get_logger().info('Starting pick and place demo')

        self.set_cube_pose(*self.PICK_POSE)

        self.command_gripper(0.019, 10.0)
        self.move_arm(self.HOME, 3.0)
        self.move_arm(self.PRE_GRASP, 3.0)
        self.move_arm(self.GRASP, 2.5)

        self.command_gripper(-0.010, 20.0)
        self.set_cube_pose(*self.LIFT_POSE)

        self.move_arm(self.LIFT, 2.5)
        self.move_arm(self.PRE_PLACE, 3.0)
        self.move_arm(self.PLACE, 2.5)

        self.set_cube_pose(*self.PLACE_POSE)
        self.command_gripper(0.019, 10.0)

        self.move_arm(self.RETREAT, 2.5)
        self.move_arm(self.HOME, 3.0)

        self.get_logger().info('Pick and place finished successfully')


def main(args=None):
    rclpy.init(args=args)
    node = TB3PickPlaceDemo()

    try:
        node.run_demo()
    except Exception as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
