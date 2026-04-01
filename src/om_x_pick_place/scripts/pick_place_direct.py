#!/usr/bin/env python3
"""
pick_place_direct.py — OpenMANIPULATOR-X pick-and-place with virtual attach

Gazebo Classic's prismatic grippers can't reliably grip objects via friction.
This script uses Gazebo's set_entity_state service to "attach" the box to the
gripper link during transport, then releases it at the drop location.

No extra packages needed — uses only gazebo_msgs (already installed).
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3

import time
import math
import threading



# ─── Config ───
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

BOX_MODEL = "unit_box_0"
GRIPPER_LINK = "open_manipulator_x::gripper_left_link"
# Alternative if the above doesn't work:
GRIPPER_LINK_ALT = "open_manipulator_x::link5"

# ─── Robot geometry (from URDF) ───
BASE_TO_J2_R = 0.012
BASE_TO_J2_H = 0.0765
J2_J3_X = 0.024
J2_J3_Z = 0.128
LA = math.sqrt(J2_J3_X**2 + J2_J3_Z**2)
BETA = math.atan2(J2_J3_X, J2_J3_Z)
LB = 0.124
LC = 0.126
GRIPPER_OFFSET = 0.0817


def ik_down(target_r, gripper_h):
    """IK for gripper pointing straight down."""
    phi = math.pi / 2.0
    wr = target_r
    wh = gripper_h + GRIPPER_OFFSET
    dr = wr - BASE_TO_J2_R
    dh = wh - BASE_TO_J2_H
    d = math.sqrt(dr**2 + dh**2)
    if d > LA + LB - 0.001 or d < abs(LA - LB) + 0.001:
        return None
    psi = math.atan2(dh, dr)
    cos_d = (LA**2 + d**2 - LB**2) / (2 * LA * d)
    cos_d = max(-1.0, min(1.0, cos_d))
    delta = math.acos(cos_d)
    for sign in [1, -1]:
        angle_A = psi + sign * delta
        q2 = math.pi / 2.0 - BETA - angle_A
        if q2 < -1.5 or q2 > 1.5:
            continue
        j3r = LA * math.sin(q2 + BETA)
        j3h = LA * math.cos(q2 + BETA)
        q3 = -math.atan2(dh - j3h, dr - j3r) - q2
        q4 = phi - q2 - q3
        if -1.5 <= q3 <= 1.4 and -1.7 <= q4 <= 1.97:
            return (q2, q3, q4)
    return None


# ─── Waypoints ───
BOX_R = 0.15
PICK_H = 0.08
APPROACH_H = 0.10
PLACE_H = 0.08

DROP_R = math.sqrt(0.15**2 + 0.15**2)
DROP_THETA = math.atan2(-0.15, 0.15)

_pp = ik_down(BOX_R, APPROACH_H)
_pk = ik_down(BOX_R, PICK_H)
_dp = ik_down(DROP_R, APPROACH_H)
_dk = ik_down(DROP_R, PLACE_H)

HOME      = [0.0, -1.0, 0.3, 0.7]
PRE_PICK  = [0.0] + list(_pp) if _pp else HOME
PICK      = [0.0] + list(_pk) if _pk else HOME
PRE_PLACE = [DROP_THETA] + list(_dp) if _dp else HOME
PLACE     = [DROP_THETA] + list(_dk) if _dk else HOME

GRIPPER_OPEN = 0.019
GRIPPER_CLOSED = -0.005


def quat_multiply(q1, q2):
    """Multiply two quaternions [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]


def quat_rotate(q, v):
    """Rotate vector v by quaternion q [x,y,z,w]."""
    qv = [v[0], v[1], v[2], 0.0]
    qc = [-q[0], -q[1], -q[2], q[3]]
    r = quat_multiply(quat_multiply(q, qv), qc)
    return [r[0], r[1], r[2]]


def quat_inverse(q):
    """Inverse of unit quaternion."""
    return [-q[0], -q[1], -q[2], q[3]]


class PickPlaceDirect(Node):
    def __init__(self):
        super().__init__("pick_place_direct")

        # Action clients
        self._arm_ac = ActionClient(self, FollowJointTrajectory, ARM_ACTION)
        self._gripper_ac = ActionClient(self, GripperCommand, GRIPPER_ACTION)

        # Gazebo state services
        self.get_state_cli = self.create_client(GetEntityState, '/get_entity_state')
        self.set_state_cli = self.create_client(SetEntityState, '/set_entity_state')

        # Attach thread control
        self._attach_active = False
        self._attach_thread = None
        self._grip_link = GRIPPER_LINK
        self._relative_pos = None
        self._relative_quat = None

        self.get_logger().info("Waiting for controllers...")
        if not self._arm_ac.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("arm_controller NOT found!")
            return
        if not self._gripper_ac.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("gripper_controller NOT found!")
            return
        self.get_logger().info("Controllers ready.")

        # Wait for Gazebo services
        self.get_logger().info("Waiting for Gazebo services...")
        if not self.get_state_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("get_entity_state service not found - attach won't work")
        if not self.set_state_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("set_entity_state service not found - attach won't work")
        self.get_logger().info("Gazebo services ready.")

    def get_entity_state(self, name, reference_frame="world"):
        """Get pose of an entity (model or link) from Gazebo."""
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = reference_frame
        future = self.get_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result and result.success:
            return result.state
        return None

    def set_entity_state(self, name, pose, reference_frame="world"):
        """Set pose of a model in Gazebo."""
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.pose = pose
        req.state.twist = Twist()  # zero velocity
        req.state.reference_frame = reference_frame
        future = self.set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        return future.result()

    def start_attach(self):
        """
        Compute the relative transform between gripper link and box,
        then start a thread that teleports the box to follow the gripper.
        """
        # Get current poses
        grip_state = self.get_entity_state(self._grip_link)
        if grip_state is None:
            # Try alternative link name
            self.get_logger().warn(f"Couldn't get {self._grip_link}, trying {GRIPPER_LINK_ALT}")
            self._grip_link = GRIPPER_LINK_ALT
            grip_state = self.get_entity_state(self._grip_link)
            if grip_state is None:
                self.get_logger().error("Cannot get gripper link state!")
                return False

        box_state = self.get_entity_state(BOX_MODEL)
        if box_state is None:
            self.get_logger().error(f"Cannot get {BOX_MODEL} state!")
            return False

        # Compute relative transform: box pose in gripper frame
        gp = grip_state.pose.position
        gq = grip_state.pose.orientation
        bp = box_state.pose.position
        bq = box_state.pose.orientation

        g_quat = [gq.x, gq.y, gq.z, gq.w]
        b_quat = [bq.x, bq.y, bq.z, bq.w]

        # Relative position: rotate (box_pos - grip_pos) by inverse of grip_quat
        diff = [bp.x - gp.x, bp.y - gp.y, bp.z - gp.z]
        g_inv = quat_inverse(g_quat)
        self._relative_pos = quat_rotate(g_inv, diff)

        # Relative orientation
        self._relative_quat = quat_multiply(g_inv, b_quat)

        self.get_logger().info(
            f"Attached! Offset: [{self._relative_pos[0]:.4f}, "
            f"{self._relative_pos[1]:.4f}, {self._relative_pos[2]:.4f}]"
        )

        # Start teleport thread
        self._attach_active = True
        self._attach_thread = threading.Thread(target=self._attach_loop, daemon=True)
        self._attach_thread.start()
        return True

    def _attach_loop(self):
        """Continuously teleport box to follow gripper."""
        rate = 0.02  # 50 Hz
        while self._attach_active:
            try:
                grip_state = self.get_entity_state(self._grip_link)
                if grip_state is None:
                    time.sleep(rate)
                    continue

                gp = grip_state.pose.position
                gq = grip_state.pose.orientation
                g_quat = [gq.x, gq.y, gq.z, gq.w]

                # Compute box world pose from relative transform
                world_offset = quat_rotate(g_quat, self._relative_pos)
                box_quat = quat_multiply(g_quat, self._relative_quat)

                pose = Pose()
                pose.position = Point(
                    x=gp.x + world_offset[0],
                    y=gp.y + world_offset[1],
                    z=gp.z + world_offset[2],
                )
                pose.orientation = Quaternion(
                    x=box_quat[0], y=box_quat[1],
                    z=box_quat[2], w=box_quat[3],
                )

                self.set_entity_state(BOX_MODEL, pose)
            except Exception as e:
                pass
            time.sleep(rate)

    def stop_attach(self):
        """Stop teleporting the box — release it."""
        self._attach_active = False
        if self._attach_thread:
            self._attach_thread.join(timeout=2.0)
            self._attach_thread = None
        self.get_logger().info("Detached box.")

    def move_arm(self, positions, duration_sec=3.0):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.velocities = [0.0] * 4
        sec = int(duration_sec)
        pt.time_from_start = Duration(sec=sec, nanosec=int((duration_sec - sec) * 1e9))
        traj.points = [pt]
        goal.trajectory = traj

        future = self._arm_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().error(f"Arm REJECTED: {positions}")
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=duration_sec + 10.0)
        self.get_logger().info(f"Arm -> [{', '.join(f'{p:.3f}' for p in positions)}]")
        return True

    def move_gripper(self, position, max_effort=20.0):
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        future = self._gripper_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().error("Gripper REJECTED!")
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=5.0)
        self.get_logger().info(f"Gripper {'OPEN' if position > 0 else 'CLOSED'}")
        return True

    def run(self):
        self.get_logger().info("=" * 55)
        self.get_logger().info("  PICK & PLACE (with virtual attach)")
        self.get_logger().info("=" * 55)
        time.sleep(1.0)

        # --- PICK PHASE ---
        self.get_logger().info("[1] HOME")
        self.move_arm(HOME, 3.0)
        time.sleep(0.5)

        self.get_logger().info("[2] Open gripper")
        self.move_gripper(GRIPPER_OPEN)
        time.sleep(0.5)

        self.get_logger().info("[3] PRE-PICK (above box)")
        self.move_arm(PRE_PICK, 3.0)
        time.sleep(0.5)

        self.get_logger().info("[4] PICK (descend to box)")
        self.move_arm(PICK, 3.0)
        time.sleep(1.0)

        self.get_logger().info("[5] Close gripper")
        self.move_gripper(GRIPPER_CLOSED, max_effort=30.0)
        time.sleep(1.0)

        # --- ATTACH (virtual grip) ---
        self.get_logger().info("[6] Attaching box to gripper (virtual)")
        attached = self.start_attach()
        if not attached:
            self.get_logger().error("Attach FAILED — aborting")
            return
        time.sleep(0.5)

        # --- TRANSPORT PHASE ---
        self.get_logger().info("[7] LIFT")
        self.move_arm(PRE_PICK, 4.0)
        time.sleep(0.5)

        self.get_logger().info("[8] PRE-PLACE (above drop)")
        self.move_arm(PRE_PLACE, 3.0)
        time.sleep(0.5)

        self.get_logger().info("[9] PLACE (descend)")
        self.move_arm(PLACE, 3.0)
        time.sleep(0.5)

        # --- RELEASE PHASE ---
        self.get_logger().info("[10] Detach box")
        self.stop_attach()
        time.sleep(0.3)

        self.get_logger().info("[11] Open gripper")
        self.move_gripper(GRIPPER_OPEN)
        time.sleep(0.5)

        self.get_logger().info("[12] RETREAT")
        self.move_arm(PRE_PLACE, 2.5)
        time.sleep(0.5)

        self.get_logger().info("[13] HOME")
        self.move_arm(HOME, 3.0)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  PICK & PLACE — Complete!")
        self.get_logger().info("=" * 55)


def main(args=None):
    rclpy.init(args=args)

    print("\n=== IK Check ===")
    for n, r in [("PRE_PICK", _pp), ("PICK", _pk), ("PRE_PLACE", _dp), ("PLACE", _dk)]:
        if r:
            print(f"  {n}: q2={r[0]:+.4f} q3={r[1]:+.4f} q4={r[2]:+.4f}")
        else:
            print(f"  {n}: FAILED!")
    print("================\n")

    node = PickPlaceDirect()
    try:
        node.run()
    except KeyboardInterrupt:
        node.stop_attach()
    finally:
        node.stop_attach()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()