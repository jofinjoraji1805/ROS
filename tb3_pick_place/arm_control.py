#!/usr/bin/env python3
"""
arm_control.py -- OpenMANIPULATOR-X arm and gripper controller.
"""

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand as GripperCommandAction
from builtin_interfaces.msg import Duration

from .config import (
    ARM_HOME, ARM_READY, ARM_PRE_PICK, ARM_PICK,
    ARM_LIFT, ARM_CARRY,
    ARM_PRE_PICK_RED, ARM_PICK_RED, ARM_LIFT_RED,
    ARM_DROP_EXTEND, ARM_DROP_OVER, ARM_DROP_RETREAT,
    GRIPPER_OPEN, GRIPPER_CLOSE, GRIPPER_EFFORT,
)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]


class ArmController:
    """Controls the 4-DOF arm and 1-DOF gripper."""

    def __init__(self, arm_pub, gripper_action_client):
        self._arm_pub = arm_pub
        self._gripper_ac = gripper_action_client

    def send_joint(self, positions, duration: float = 2.0):
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        s = int(duration)
        ns = int((duration - s) * 1e9)
        pt.time_from_start = Duration(sec=s, nanosec=ns)
        msg.points = [pt]
        self._arm_pub.publish(msg)

    # ── Pick poses ───────────────────────────────────────────────────

    def home(self, dur: float = 3.0):
        self.send_joint(ARM_HOME, dur)

    def ready(self, dur: float = 3.0):
        self.send_joint(ARM_READY, dur)

    def pre_pick(self, dur: float = 3.0, color: str = ""):
        pose = ARM_PRE_PICK_RED if color == "RED" else ARM_PRE_PICK
        self.send_joint(pose, dur)

    def pick(self, dur: float = 3.0, color: str = ""):
        pose = ARM_PICK_RED if color == "RED" else ARM_PICK
        self.send_joint(pose, dur)

    def lift(self, dur: float = 3.0, color: str = ""):
        pose = ARM_LIFT_RED if color == "RED" else ARM_LIFT
        self.send_joint(pose, dur)

    def carry(self, dur: float = 3.0):
        self.send_joint(ARM_CARRY, dur)

    # ── Drop-into-basket poses (from above) ──────────────────────────

    def drop_extend(self, dur: float = 3.0):
        """Extend arm forward + high to clear basket walls."""
        self.send_joint(ARM_DROP_EXTEND, dur)

    def drop_over(self, dur: float = 3.0):
        """Lower arm to position gripper above basket opening."""
        self.send_joint(ARM_DROP_OVER, dur)

    def drop_retreat(self, dur: float = 3.0):
        """Pull arm back after dropping cube."""
        self.send_joint(ARM_DROP_RETREAT, dur)

    # ── Gripper ──────────────────────────────────────────────────────

    def gripper(self, position: float, effort: float = GRIPPER_EFFORT):
        goal = GripperCommandAction.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = effort
        self._gripper_ac.send_goal_async(goal)

    def open_gripper(self):
        self.gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.gripper(GRIPPER_CLOSE)

    def close_gripper_slow(self):
        """Close gripper to grasp position."""
        self.gripper(GRIPPER_CLOSE, effort=GRIPPER_EFFORT)

    def close_gripper_medium(self):
        """Same close."""
        self.gripper(GRIPPER_CLOSE, effort=GRIPPER_EFFORT)

    def close_gripper_firm(self):
        """Same close."""
        self.gripper(GRIPPER_CLOSE, effort=GRIPPER_EFFORT)

    def gripper_to(self, position: float, effort: float = GRIPPER_EFFORT):
        """Move gripper to a specific position."""
        self.gripper(position, effort)
