#!/usr/bin/env python3
"""
pick_and_place.py  –  OpenMANIPULATOR-X pick-and-place for unit_box_0

Workflow:
  1. Move arm to HOME
  2. Open gripper
  3. Move to PRE-PICK (above box)
  4. Move to PICK  (grasp height)
  5. Close gripper
  6. Lift (PRE-PICK again)
  7. Move to PRE-PLACE (above drop)
  8. Move to PLACE
  9. Open gripper
 10. Retreat to HOME

Coordinates are in the robot base_link frame.
Robot sits at world (0, 0, 0.06).  Box at world (0.15, 0.0, 0.08).
So in base_link frame the box is roughly at x≈0.15 y≈0.0 z≈0.02.
Drop location at x≈0.15 y≈-0.15 z≈0.02.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    PositionConstraint, OrientationConstraint,
    BoundingVolume, RobotState, PlanningOptions,
    WorkspaceParameters,
)
from geometry_msgs.msg import (
    PoseStamped, Pose, Point, Quaternion, Vector3,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState

import math
import time
import numpy as np


# ────────────────────────── Config ──────────────────────────
ARM_GROUP = "arm"
GRIPPER_GROUP = "gripper"
PLANNING_FRAME = "world"         # MoveIt planning frame
EE_LINK = "end_effector_link"    # end-effector link

# Joint names (OpenMANIPULATOR-X)
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
GRIPPER_JOINTS = ["gripper", "gripper_sub"]

# Gripper controller topic
GRIPPER_CONTROLLER_TOPIC = "/gripper_controller/commands"
ARM_CONTROLLER_TOPIC = "/arm_controller/commands"

# Positions in base_link frame  (metres)
# Adjust these if the robot base is offset from world origin
ROBOT_BASE_Z = 0.06   # table height

# Box centre in world frame
BOX_X, BOX_Y, BOX_Z = 0.15, 0.0, 0.08
# Convert to robot base frame (robot at 0,0,ROBOT_BASE_Z)
PICK_X = BOX_X
PICK_Y = BOX_Y
PICK_Z = BOX_Z - ROBOT_BASE_Z + 0.02   # centre of box + small offset

# Drop location
DROP_X, DROP_Y = 0.15, -0.15
DROP_Z = PICK_Z

# Heights
APPROACH_HEIGHT = 0.08  # above table surface (robot frame)

# Joint configs (radians) — known safe poses for OpenMANIPULATOR-X
HOME_JOINTS    = [0.0, -1.05, 0.35, 0.70]
GRIPPER_OPEN   = [0.01]
GRIPPER_CLOSED = [-0.01]


class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place')
        self.cb_group = ReentrantCallbackGroup()

        # Action client for MoveGroup
        self._move_group_ac = ActionClient(
            self, MoveGroup, 'move_action',
            callback_group=self.cb_group,
        )

        # Publishers for direct joint trajectory control (fallback)
        self.arm_pub = self.create_publisher(
            Float64MultiArray, ARM_CONTROLLER_TOPIC, 10)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, GRIPPER_CONTROLLER_TOPIC, 10)

        # Joint state subscriber
        self._current_joint_state = None
        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10,
            callback_group=self.cb_group,
        )

        self.get_logger().info("PickAndPlace node initialised. Waiting for MoveGroup action server...")

    def _js_cb(self, msg: JointState):
        self._current_joint_state = msg

    # ─────────────── Direct joint command helpers ───────────────
    def send_arm_joints(self, positions: list, wait: float = 3.0):
        """Send arm joint positions via the controller topic."""
        msg = Float64MultiArray()
        msg.data = positions
        self.arm_pub.publish(msg)
        self.get_logger().info(f"Arm cmd: {positions}")
        time.sleep(wait)

    def send_gripper(self, positions: list, wait: float = 1.5):
        """Send gripper position via the controller topic."""
        msg = Float64MultiArray()
        msg.data = positions
        self.gripper_pub.publish(msg)
        self.get_logger().info(f"Gripper cmd: {positions}")
        time.sleep(wait)

    # ─────────── MoveIt2 Cartesian goal helper ─────────────
    def plan_and_execute_pose(self, x, y, z, qx=0.0, qy=0.707, qz=0.0, qw=0.707,
                               timeout=10.0):
        """
        Plan to a Cartesian pose using MoveGroup action.
        Default orientation: end-effector pointing downward.
        """
        if not self._move_group_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("MoveGroup action server not available – using joint fallback")
            return False

        goal = MoveGroup.Goal()

        # Motion plan request
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = timeout
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3
        req.pipeline_id = "move_group"

        # Workspace
        ws = WorkspaceParameters()
        ws.header.frame_id = PLANNING_FRAME
        ws.min_corner = Vector3(x=-1.0, y=-1.0, z=-0.1)
        ws.max_corner = Vector3(x=1.0, y=1.0, z=1.0)
        req.workspace_parameters = ws

        # Pose goal
        target_pose = PoseStamped()
        target_pose.header.frame_id = PLANNING_FRAME
        target_pose.pose.position = Point(x=x, y=y, z=z)
        target_pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = PLANNING_FRAME
        pos_constraint.link_name = EE_LINK
        pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.005]
        bv.primitives = [sp]
        bv.primitive_poses = [target_pose.pose]
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0

        # Orientation constraint
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = PLANNING_FRAME
        ori_constraint.link_name = EE_LINK
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints = [pos_constraint]
        constraints.orientation_constraints = [ori_constraint]
        req.goal_constraints = [constraints]

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        future = self._move_group_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout + 10.0)
        result = result_future.result()

        if result and result.result.error_code.val == 1:  # SUCCESS
            self.get_logger().info(f"Reached pose ({x:.3f}, {y:.3f}, {z:.3f})")
            return True
        else:
            ec = result.result.error_code.val if result else -999
            self.get_logger().warn(f"MoveGroup failed (error_code={ec})")
            return False

    # ──────── MoveIt2 joint goal helper ────────
    def plan_and_execute_joints(self, joint_values: list, timeout=10.0):
        if not self._move_group_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("MoveGroup not available – direct publish")
            self.send_arm_joints(joint_values)
            return True

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.num_planning_attempts = 10
        req.allowed_planning_time = timeout
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()
        for jname, jval in zip(ARM_JOINTS, joint_values):
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = jval
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True

        future = self._move_group_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().warn("Joint goal rejected – using direct publish")
            self.send_arm_joints(joint_values)
            return True

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=timeout + 10.0)
        res = rf.result()
        if res and res.result.error_code.val == 1:
            self.get_logger().info("Joint goal reached via MoveIt2")
            return True
        else:
            self.get_logger().warn("Joint goal via MoveIt failed – direct publish fallback")
            self.send_arm_joints(joint_values)
            return True

    # ═══════════════════ MAIN SEQUENCE ═══════════════════
    def run(self):
        self.get_logger().info("=" * 50)
        self.get_logger().info("  PICK AND PLACE – Starting Sequence")
        self.get_logger().info("=" * 50)

        time.sleep(2.0)  # settle

        # 1. HOME
        self.get_logger().info("[1/10] Moving to HOME")
        self.plan_and_execute_joints(HOME_JOINTS)

        # 2. Open gripper
        self.get_logger().info("[2/10] Opening gripper")
        self.send_gripper(GRIPPER_OPEN)

        # 3. PRE-PICK (above box)
        self.get_logger().info(f"[3/10] Moving to PRE-PICK above box")
        ok = self.plan_and_execute_pose(PICK_X, PICK_Y, APPROACH_HEIGHT)
        if not ok:
            self.get_logger().warn("Cartesian PRE-PICK failed, trying joint fallback")
            # Approximate pre-pick joints (arm stretched forward, elbow up)
            self.send_arm_joints([0.0, 0.0, -0.5, 1.0])

        # 4. PICK (descend to box)
        self.get_logger().info(f"[4/10] Descending to PICK (z={PICK_Z:.3f})")
        ok = self.plan_and_execute_pose(PICK_X, PICK_Y, PICK_Z)
        if not ok:
            self.send_arm_joints([0.0, 0.35, -0.25, 0.70])

        # 5. Close gripper
        self.get_logger().info("[5/10] Closing gripper – grasping box")
        self.send_gripper(GRIPPER_CLOSED, wait=2.0)

        # 6. Lift
        self.get_logger().info("[6/10] Lifting (PRE-PICK)")
        ok = self.plan_and_execute_pose(PICK_X, PICK_Y, APPROACH_HEIGHT)
        if not ok:
            self.send_arm_joints([0.0, 0.0, -0.5, 1.0])

        # 7. PRE-PLACE (above drop)
        self.get_logger().info(f"[7/10] Moving to PRE-PLACE above drop location")
        ok = self.plan_and_execute_pose(DROP_X, DROP_Y, APPROACH_HEIGHT)
        if not ok:
            self.send_arm_joints([-0.78, 0.0, -0.5, 1.0])

        # 8. PLACE (descend)
        self.get_logger().info(f"[8/10] Descending to PLACE (z={DROP_Z:.3f})")
        ok = self.plan_and_execute_pose(DROP_X, DROP_Y, DROP_Z)
        if not ok:
            self.send_arm_joints([-0.78, 0.35, -0.25, 0.70])

        # 9. Open gripper – release
        self.get_logger().info("[9/10] Opening gripper – releasing box")
        self.send_gripper(GRIPPER_OPEN, wait=2.0)

        # 10. HOME
        self.get_logger().info("[10/10] Returning to HOME")
        self.plan_and_execute_joints(HOME_JOINTS)

        self.get_logger().info("=" * 50)
        self.get_logger().info("  PICK AND PLACE – Complete!")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
