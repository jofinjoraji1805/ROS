#!/usr/bin/env python3
import math
import time
import csv
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration

import tf2_ros

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import Twist

# =========================
# USER TUNABLES
# =========================
BOX_MODEL = "unit_box_0"
BOX_HALF_HEIGHT = 0.020  # box z-size=0.040 in your world

# IK clearances (arm-frame heights)
PREGRASP_CLEARANCE = 0.03
PICK_EXTRA_ABOVE_TOP = 0.008

# distance sweep range (meters)
SWEEP_R_MIN = 0.10
SWEEP_R_MAX = 0.25
SWEEP_R_STEP = 0.01

# print rate
PRINT_HZ = 2.0

# where to log snapshots
LOG_CSV = "tb3_pick_probe_log.csv"

# Topics/entities
ROBOT_ENTITY = "turtlebot3_manipulation_system"
BASE_ENTITY = f"{ROBOT_ENTITY}::base_footprint"

BASE_TF_CANDIDATES = ["base_footprint", "base_link"]
ARM_BASE_TF_CANDIDATES = [
    "arm_base_link",
    "manipulator_base_link",
    "open_manipulator_base_link",
    "link1",
    "open_manipulator_link1",
]

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]
ARM_ACTION = "/arm_controller/follow_joint_trajectory"
CMD_VEL_TOPIC = "/cmd_vel"

HOME = [0.0, -1.0, 0.3, 0.7]

# =========================
# OpenMANIPULATOR-X IK model (same as your pick script)
# =========================
BASE_TO_J2_R = 0.012
BASE_TO_J2_H = 0.0765
J2_J3_X = 0.024
J2_J3_Z = 0.128
LA = math.sqrt(J2_J3_X**2 + J2_J3_Z**2)
BETA = math.atan2(J2_J3_X, J2_J3_Z)
LB = 0.124
LC = 0.126
GRIPPER_OFFSET = 0.0817


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def quat_inverse(q):
    return [-q[0], -q[1], -q[2], q[3]]


def quat_rotate(q, v):
    qv = [v[0], v[1], v[2], 0.0]
    qc = [-q[0], -q[1], -q[2], q[3]]
    r = quat_multiply(quat_multiply(q, qv), qc)
    return [r[0], r[1], r[2]]


def ik_down(target_r, gripper_h):
    """Analytical IK for gripper pointing straight down (phi=pi/2). Returns (q2,q3,q4) or None."""
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
    cos_d = clamp(cos_d, -1.0, 1.0)
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


@dataclass
class ProbeState:
    r: float
    theta: float
    box_x: float
    box_y: float
    box_z: float
    box_top_h: float
    pregrasp_h: float
    pick_h: float
    ik_pre_ok: bool
    ik_pick_ok: bool
    pre_joints: Optional[List[float]]
    pick_joints: Optional[List[float]]


class TB3PickProbe(Node):
    def __init__(self):
        super().__init__("tb3_pick_probe")

        # Gazebo service
        self.get_state_cli = self.create_client(GetEntityState, "/get_entity_state")

        # Arm action
        self.arm_ac = ActionClient(self, FollowJointTrajectory, ARM_ACTION)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # cmd_vel pub (not used to drive, but handy to stop)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self.base_tf = None
        self.arm_tf = None
        self.tf_base_to_arm = None  # (txyz, qxyzw): base<-arm

        self.latest: Optional[ProbeState] = None
        self._stop = False

        self._wait_for()
        self._resolve_tf_frames()

        # periodic print
        self.last_print = 0.0
        self.timer = self.create_timer(1.0 / PRINT_HZ, self._tick)

        # command thread
        self.cmd_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self.cmd_thread.start()

        self._init_csv()

        self.get_logger().info(
            "\nCommands:\n"
            "  s  + Enter : save snapshot to CSV\n"
            "  w  + Enter : sweep distances and show IK-valid r\n"
            "  t  + Enter : test arm move to PREGRASP (no grasp)\n"
            "  h  + Enter : move arm HOME\n"
            "  q  + Enter : quit\n"
        )

    def _wait_for(self):
        self.get_logger().info("Waiting for /get_entity_state and arm action...")
        if not self.get_state_cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/get_entity_state not found (gazebo_ros_state plugin?)")
        if not self.arm_ac.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("arm_controller FollowJointTrajectory action not found")

    def _try_lookup_tf(self, target, source, timeout_sec=1.0):
        try:
            t = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(),
                timeout=RclpyDuration(seconds=timeout_sec)
            )
            tr = t.transform.translation
            rq = t.transform.rotation
            return [tr.x, tr.y, tr.z], [rq.x, rq.y, rq.z, rq.w]
        except Exception:
            return None

    def _resolve_tf_frames(self):
        # give TF time
        end = time.time() + 2.0
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        for base in BASE_TF_CANDIDATES:
            for arm in ARM_BASE_TF_CANDIDATES:
                res = self._try_lookup_tf(base, arm, timeout_sec=0.8)
                if res is not None:
                    self.base_tf = base
                    self.arm_tf = arm
                    self.tf_base_to_arm = res
                    t, _ = res
                    self.get_logger().info(
                        f"[TF] Using base='{base}' arm='{arm}'  base<-arm: t=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]"
                    )
                    return
        raise RuntimeError("Could not find TF transform base->arm (base_footprint->link1 etc)")

    def _init_csv(self):
        # create header if file doesn't exist or empty
        try:
            with open(LOG_CSV, "r", newline="") as f:
                has = f.read(1) != ""
            if has:
                return
        except FileNotFoundError:
            pass

        with open(LOG_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "stamp",
                "r", "theta_deg",
                "box_x_arm", "box_y_arm", "box_z_arm",
                "box_top_h", "pregrasp_h", "pick_h",
                "ik_pre_ok", "ik_pick_ok",
                "pre_q1","pre_q2","pre_q3","pre_q4",
                "pick_q1","pick_q2","pick_q3","pick_q4",
            ])

    # ---------------- Gazebo poses ----------------
    def get_entity_state(self, name, ref="world", timeout=1.0):
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = ref
        fut = self.get_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        res = fut.result()
        if res and res.success:
            return res.state
        return None

    def get_base_world_pose(self):
        st = self.get_entity_state(BASE_ENTITY, timeout=1.0)
        if st is None:
            return None
        p = st.pose.position
        q = st.pose.orientation
        return (p.x, p.y, p.z), [q.x, q.y, q.z, q.w]

    def get_box_world_pos(self):
        st = self.get_entity_state(BOX_MODEL, timeout=1.0)
        if st is None:
            return None
        p = st.pose.position
        return (p.x, p.y, p.z)

    # ---------------- Arm base world pose via TF ----------------
    def arm_base_world_pose_from_tf(self):
        base = self.get_base_world_pose()
        if base is None or self.tf_base_to_arm is None:
            return None
        (bx, by, bz), bq = base
        t_base_arm, q_base_arm = self.tf_base_to_arm

        t_world = quat_rotate(bq, t_base_arm)
        ax = bx + t_world[0]
        ay = by + t_world[1]
        az = bz + t_world[2]
        aq = quat_multiply(bq, q_base_arm)
        return (ax, ay, az), aq

    def world_to_arm(self, arm_world_pose, p_world):
        (ax, ay, az), aq = arm_world_pose
        diff_w = [p_world[0] - ax, p_world[1] - ay, p_world[2] - az]
        return quat_rotate(quat_inverse(aq), diff_w)

    # ---------------- Arm motion (for testing) ----------------
    def move_arm(self, positions, duration_sec=2.5):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        pt.velocities = [0.0] * len(ARM_JOINTS)
        sec = int(duration_sec)
        pt.time_from_start = Duration(sec=sec, nanosec=int((duration_sec - sec) * 1e9))
        traj.points = [pt]
        goal.trajectory = traj

        fut = self.arm_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error("Arm goal rejected")
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=duration_sec + 10.0)
        return True

    # ---------------- Probe computation ----------------
    def compute_probe(self) -> Optional[ProbeState]:
        arm_world = self.arm_base_world_pose_from_tf()
        box_w = self.get_box_world_pos()
        if arm_world is None or box_w is None:
            return None

        box_a = self.world_to_arm(arm_world, box_w)
        box_x, box_y, box_z = box_a

        r = math.hypot(box_x, box_y)
        theta = math.atan2(box_y, box_x)

        box_top_h = box_z + BOX_HALF_HEIGHT
        pregrasp_h = box_top_h + PREGRASP_CLEARANCE
        pick_h = box_top_h + PICK_EXTRA_ABOVE_TOP

        ik_pre = ik_down(r, pregrasp_h)
        ik_pick = ik_down(r, pick_h)

        pre_j = None
        pick_j = None
        if ik_pre is not None:
            pre_j = [theta, ik_pre[0], ik_pre[1], ik_pre[2]]
        if ik_pick is not None:
            pick_j = [theta, ik_pick[0], ik_pick[1], ik_pick[2]]

        return ProbeState(
            r=r,
            theta=theta,
            box_x=box_x, box_y=box_y, box_z=box_z,
            box_top_h=box_top_h,
            pregrasp_h=pregrasp_h,
            pick_h=pick_h,
            ik_pre_ok=ik_pre is not None,
            ik_pick_ok=ik_pick is not None,
            pre_joints=pre_j,
            pick_joints=pick_j,
        )

    def _tick(self):
        st = self.compute_probe()
        if st is None:
            return
        self.latest = st

        now = time.time()
        if now - self.last_print < (1.0 / PRINT_HZ):
            return
        self.last_print = now

        pre = "OK" if st.ik_pre_ok else "FAIL"
        pk = "OK" if st.ik_pick_ok else "FAIL"
        self.get_logger().info(
            f"[LIVE] r_arm={st.r:.3f}  theta={math.degrees(st.theta):+.1f}deg  "
            f"box_top_h={st.box_top_h:+.3f}  pre_h={st.pregrasp_h:+.3f} pick_h={st.pick_h:+.3f}  "
            f"IK(pre)={pre} IK(pick)={pk}"
        )
        if st.pre_joints:
            q = st.pre_joints
            self.get_logger().info(
                f"       PRE q=[{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f}]"
            )

    # ---------------- Commands ----------------
    def _stdin_loop(self):
        while not self._stop and rclpy.ok():
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                cmd = line.strip().lower()
                if cmd == "q":
                    self.get_logger().info("Quit requested.")
                    self._stop = True
                    break
                elif cmd == "s":
                    self.save_snapshot()
                elif cmd == "w":
                    self.sweep_distances()
                elif cmd == "h":
                    self.get_logger().info("Moving HOME...")
                    self.move_arm(HOME, 3.0)
                elif cmd == "t":
                    self.test_pregrasp()
                else:
                    self.get_logger().info("Commands: s / w / t / h / q")
            except Exception:
                time.sleep(0.1)

    def save_snapshot(self):
        st = self.latest or self.compute_probe()
        if st is None:
            self.get_logger().warn("No probe state to save.")
            return
        stamp = time.time()

        pre = st.pre_joints if st.pre_joints else [math.nan]*4
        pk = st.pick_joints if st.pick_joints else [math.nan]*4

        with open(LOG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                f"{stamp:.3f}",
                f"{st.r:.6f}", f"{math.degrees(st.theta):.3f}",
                f"{st.box_x:.6f}", f"{st.box_y:.6f}", f"{st.box_z:.6f}",
                f"{st.box_top_h:.6f}", f"{st.pregrasp_h:.6f}", f"{st.pick_h:.6f}",
                int(st.ik_pre_ok), int(st.ik_pick_ok),
                *[f"{v:.6f}" for v in pre],
                *[f"{v:.6f}" for v in pk],
            ])
        self.get_logger().info(f"Saved snapshot → {LOG_CSV}")

    def sweep_distances(self):
        st = self.latest or self.compute_probe()
        if st is None:
            self.get_logger().warn("No probe state for sweep.")
            return

        # Keep theta and heights; vary r as if you move base closer/farther
        rs = []
        ok_pre = []
        ok_pick = []
        for i in range(int((SWEEP_R_MAX - SWEEP_R_MIN) / SWEEP_R_STEP) + 1):
            r = SWEEP_R_MIN + i * SWEEP_R_STEP
            ik_pre = ik_down(r, st.pregrasp_h)
            ik_pick = ik_down(r, st.pick_h)
            rs.append(r)
            ok_pre.append(ik_pre is not None)
            ok_pick.append(ik_pick is not None)

        self.get_logger().info("Sweep results (r -> IK_pre / IK_pick):")
        line = []
        for r, a, b in zip(rs, ok_pre, ok_pick):
            line.append(f"{r:.2f}:{'P' if a else '-'}{'K' if b else '-'}")
            if len(line) >= 10:
                self.get_logger().info("  " + "  ".join(line))
                line = []
        if line:
            self.get_logger().info("  " + "  ".join(line))

        # Suggest a “good” r range where both are OK
        both = [r for r, a, b in zip(rs, ok_pre, ok_pick) if a and b]
        if both:
            self.get_logger().info(f"Suggested IK-valid r range: {min(both):.2f} .. {max(both):.2f} m")
        else:
            self.get_logger().warn("No r in sweep range gives BOTH IK(pre) and IK(pick). Try widening sweep or increase clearances.")

    def test_pregrasp(self):
        st = self.latest or self.compute_probe()
        if st is None:
            self.get_logger().warn("No probe state to test.")
            return
        if not st.pre_joints:
            self.get_logger().warn("IK(pregrasp) not available at current pose.")
            return

        self.get_logger().info("Testing PREGRASP pose (no grasp).")
        # go home -> pre -> home
        self.move_arm(HOME, 2.5)
        self.move_arm(st.pre_joints, 2.5)
        time.sleep(0.4)
        self.move_arm(HOME, 2.5)
        self.get_logger().info("PREGRASP test done.")

    def shutdown(self):
        self._stop = True
        try:
            self.cmd_thread.join(timeout=1.0)
        except Exception:
            pass


def main():
    rclpy.init()
    node = None
    try:
        node = TB3PickProbe()
        while rclpy.ok() and not node._stop:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()