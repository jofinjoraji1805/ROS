#!/usr/bin/env python3
"""
robot_node.py -- Main ROS 2 node for the two-phase pick-and-place system.

Phase 1 (MAPPING): Explorer drives waypoints, LIDAR builds map, saves it.
Phase 2 (MISSION): Navigator loads map, A* plans paths, FSM picks/places.
"""

import math
import threading
from typing import List, Optional, Tuple

import rclpy
import rclpy.time
import rclpy.duration

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import GripperCommand as GripperCommandAction
from gazebo_msgs.srv import SetEntityState, DeleteEntity, SpawnEntity
from gazebo_msgs.msg import EntityState
from linkattacher_msgs.srv import AttachLink, DetachLink
from gazebo_msgs.msg import ContactsState
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from .config import (
    YOLO_MODEL, CONF_THRESH,
    CAMERA_TOPIC, ODOM_TOPIC, CMD_VEL_TOPIC, SCAN_TOPIC,
    ARM_TRAJECTORY_TOPIC, GRIPPER_ACTION,
    TELEOP_LINEAR, TELEOP_ANGULAR,
    ST_SEARCH_DOCK, ST_ALIGN_DOCK, ST_APPROACH_DOCK, ST_DOCK_CREEP, ST_PARKED,
)
from .perception import PerceptionModule, Detection
from .motion import MotionController
from .arm_control import ArmController
from .lidar import LidarProcessor
from .explorer import Explorer
from .navigator import Navigator
from .state_machine import StateMachine


class RobotController(Node):
    """Central ROS 2 node with exploration, navigation, and pick-place."""

    # Phase constants
    PHASE_IDLE = "IDLE"
    PHASE_MAPPING = "MAPPING"
    PHASE_MISSION = "MISSION"
    PHASE_MANUAL = "MANUAL"

    def __init__(self):
        super().__init__("yolo_pick_place")
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ── Perception ────────────────────────────────────────────────
        self.get_logger().info(f"Loading YOLO model: {YOLO_MODEL}")
        self.perception = PerceptionModule(YOLO_MODEL, CONF_THRESH)
        self.get_logger().info(
            f"YOLO ready -- classes: {list(self.perception.names.values())}")

        # ── LIDAR ─────────────────────────────────────────────────────
        self.lidar = LidarProcessor()

        # ── ROS interfaces ────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        arm_pub = self.create_publisher(
            JointTrajectory, ARM_TRAJECTORY_TOPIC, 10)
        gripper_ac = ActionClient(
            self, GripperCommandAction, GRIPPER_ACTION)

        self.cam_sub = self.create_subscription(
            Image, CAMERA_TOPIC, self._cam_cb, 1)
        self.odom_sub = self.create_subscription(
            Odometry, ODOM_TOPIC, self._odom_cb, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, SCAN_TOPIC, self._scan_cb, 5)

        # ── Controllers ──────────────────────────────────────────────
        self.motion = MotionController(self._cmd_pub)
        self.arm = ArmController(arm_pub, gripper_ac)

        # ── Gazebo entity state (for cube teleport/drop) ─────────────
        self._set_entity_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        # ── Gazebo delete/spawn (for dock LED color change) ──────────
        self._delete_entity_cli = self.create_client(
            DeleteEntity, '/delete_entity')
        self._spawn_entity_cli = self.create_client(
            SpawnEntity, '/spawn_entity')

        # ── IFRA Link Attacher (for pick-and-place grasp) ────────────
        self._attach_cli = self.create_client(AttachLink, '/ATTACHLINK')
        self._detach_cli = self.create_client(DetachLink, '/DETACHLINK')

        # ── Contact sensors on gripper fingers ──────────────────────
        self._left_contact = False
        self._right_contact = False
        self._grasp_armed = False  # set True when we want to detect grasp
        self._grasp_attached = False  # True after attach is called
        self._grasp_target_color = ""
        self.create_subscription(
            ContactsState, '/gripper/left_finger_contact',
            self._left_contact_cb, 5)
        self.create_subscription(
            ContactsState, '/gripper/right_finger_contact',
            self._right_contact_cb, 5)

        self.get_logger().info("Waiting for gripper action server...")
        gripper_ac.wait_for_server(timeout_sec=10.0)
        self.get_logger().info("All services ready!")

        # ── Send arm to HOME after controllers are ready ─────────────
        self._arm_init_count = 0
        self._arm_init_timer = self.create_timer(1.0, self._init_arm_cb)

        # ── Explorer (Phase 1) ───────────────────────────────────────
        self.explorer = Explorer(
            motion=self.motion,
            lidar=self.lidar,
            get_odom_fn=self._get_odom,
            log_fn=lambda msg: self.get_logger().info(msg),
        )

        # ── Navigator (Phase 2) ─────────────────────────────────────
        self.navigator = Navigator(
            motion=self.motion,
            lidar=self.lidar,
            get_odom_fn=self._get_odom,
            log_fn=lambda msg: self.get_logger().info(msg),
        )

        # ── State machine (Phase 2) ─────────────────────────────────
        self.fsm = StateMachine(
            perception=self.perception,
            motion=self.motion,
            arm=self.arm,
            lidar=self.lidar,
            navigator=self.navigator,
            find_target_fn=self._find_target,
            get_odom_fn=self._get_odom,
            log_fn=lambda msg: self.get_logger().info(msg),
            robot_node=self,
        )

        # ── Phase tracking ───────────────────────────────────────────
        self.phase = self.PHASE_IDLE

        # ── Shared data ─────────────────────────────────────────────
        self.latest_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.detections: List[Detection] = []

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0

        self._manual_lx = 0.0
        self._manual_az = 0.0

        # ── CLI trigger services ─────────────────────────────────────
        self.create_service(Trigger, '~/start_mission', self._srv_start_mission)
        self.create_service(Trigger, '~/stop_mission', self._srv_stop_mission)

        # ── 10 Hz tick ───────────────────────────────────────────────
        self.create_timer(0.1, self._tick)

    # ── Arm init (delayed so controller subscribes first) ─────────

    def _init_arm_cb(self):
        self._arm_init_count += 1
        self.arm.home(dur=3.0)
        self.arm.open_gripper()
        self.get_logger().info(
            f"Arm HOME command sent (attempt {self._arm_init_count})")
        # Send 3 times over 3 seconds to ensure controller receives it
        if self._arm_init_count >= 3:
            self._arm_init_timer.cancel()

    # ── ROS callbacks ────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._odom_x = p.x
        self._odom_y = p.y
        self._odom_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z))

    def _scan_cb(self, msg: LaserScan):
        self.lidar.update_scan(
            msg.ranges, msg.angle_min, msg.angle_max,
            msg.angle_increment, msg.range_min, msg.range_max)
        self.lidar.update_map(self._odom_x, self._odom_y, self._odom_yaw)

    def _cam_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        dets = self.perception.detect(frame)

        idx = self.fsm.task_idx
        tasks = self.fsm.tasks
        _dock_states = (ST_SEARCH_DOCK, ST_ALIGN_DOCK, ST_APPROACH_DOCK, ST_DOCK_CREEP, ST_PARKED)
        if idx < len(tasks):
            lbl = tasks[idx].label
        elif self.fsm.state in (ST_SEARCH_DOCK, ST_ALIGN_DOCK, ST_APPROACH_DOCK, ST_DOCK_CREEP):
            lbl = "DOCKING"
        elif self.fsm.state == ST_PARKED:
            lbl = "PARKED"
        else:
            lbl = "DONE"
        front_d = self.lidar.get_front_distance()
        hud = f"[{lbl}] {self.fsm.state}  LIDAR:{front_d:.2f}m"

        vis = self.perception.annotate(frame, dets, hud)

        with self.lock:
            self.latest_frame = frame
            self.annotated_frame = vis
            self.detections = dets

    # ── Accessors ────────────────────────────────────────────────────

    def _get_odom(self) -> Tuple[float, float, float]:
        return self._odom_x, self._odom_y, self._odom_yaw

    def _find_target(self, target_cls: int):
        with self.lock:
            dets = list(self.detections)
            frame = self.latest_frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return self.perception.find_target(dets, target_cls, w, h)

    # ── Gazebo cube manipulation ───────────────────────────────────

    _CUBE_NAMES = {"RED": "red_cube", "BLUE": "blue_cube", "GREEN": "green_cube"}

    # ── Contact sensor callbacks ─────────────────────────────────

    def _left_contact_cb(self, msg: ContactsState):
        self._left_contact = len(msg.states) > 0
        self._check_grasp()

    def _right_contact_cb(self, msg: ContactsState):
        self._right_contact = len(msg.states) > 0
        self._check_grasp()

    def _check_grasp(self):
        """Auto-attach when both fingers detect contact and grasp is armed."""
        if self._grasp_armed and not self._grasp_attached:
            if self._left_contact or self._right_contact:
                self.get_logger().info(
                    f"Contact detected! L={self._left_contact} R={self._right_contact} — attaching...")
                self.attach_cube(self._grasp_target_color)
                self._grasp_attached = True
                self._grasp_armed = False

    def arm_grasp(self, color: str):
        """Arm the grasp detector — will auto-attach on finger contact."""
        self._grasp_armed = True
        self._grasp_attached = False
        self._grasp_target_color = color
        self._left_contact = False
        self._right_contact = False
        self.get_logger().info(f"Grasp armed for {color}")

    def is_grasp_attached(self) -> bool:
        return self._grasp_attached

    def disarm_grasp(self):
        self._grasp_armed = False
        self._grasp_attached = False

    def attach_cube(self, color: str):
        """Attach cube to gripper using IFRA LinkAttacher fixed joint."""
        cube_name = self._CUBE_NAMES.get(color)
        if not cube_name:
            return
        if not self._attach_cli.service_is_ready():
            self.get_logger().warn("ATTACHLINK service not ready! Waiting...")
            self._attach_cli.wait_for_service(timeout_sec=5.0)
        req = AttachLink.Request()
        req.model1_name = 'turtlebot3_manipulation_system'
        req.link1_name = 'gripper_left_link'
        req.model2_name = cube_name
        req.link2_name = 'link'
        future = self._attach_cli.call_async(req)
        future.add_done_callback(
            lambda f: self._log_attach_result(f, f"Attach {cube_name}"))
        self.get_logger().info(f"Attaching {cube_name} to gripper...")

    def detach_cube(self, color: str):
        """Detach cube from gripper using IFRA LinkAttacher."""
        cube_name = self._CUBE_NAMES.get(color)
        if not cube_name:
            return
        if not self._detach_cli.service_is_ready():
            self.get_logger().warn("DETACHLINK service not ready! Waiting...")
            self._detach_cli.wait_for_service(timeout_sec=5.0)
        req = DetachLink.Request()
        req.model1_name = 'turtlebot3_manipulation_system'
        req.link1_name = 'gripper_left_link'
        req.model2_name = cube_name
        req.link2_name = 'link'
        future = self._detach_cli.call_async(req)
        future.add_done_callback(
            lambda f: self._log_attach_result(f, f"Detach {cube_name}"))
        self.get_logger().info(f"Detaching {cube_name}...")

    def _log_attach_result(self, future, label: str):
        try:
            result = future.result()
            self.get_logger().info(f"LinkAttacher result: {result}")
        except Exception as e:
            self.get_logger().error(f"LinkAttacher exception: {e}")

    def carry_cube(self, color: str):
        """No-op: cube is physically attached via fixed joint."""
        pass

    def drop_cube(self, color: str, x: float, y: float, z: float = 0.05):
        """Place cube at drop location."""
        cube_name = self._CUBE_NAMES.get(color)
        if not cube_name:
            return
        req = SetEntityState.Request()
        state = EntityState()
        state.name = cube_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.reference_frame = 'world'
        req.state = state
        self._set_entity_cli.call_async(req)
        self.get_logger().info(f"Dropped {cube_name} at ({x:.2f}, {y:.2f})")

    # ── Dock LED color change ──────────────────────────────────────

    _DOCK_SDF_TEMPLATE = """\
<sdf version='1.6'>
  <model name='charging_dock'>
    <static>1</static>
    <pose>0.5 -3.1 0.0 0 0 0</pose>
    <link name='dock_link'>
      <visual name='platform_v'>
        <pose>0 0 0.001 0 0 0</pose>
        <geometry><box><size>0.45 0.40 0.002</size></box></geometry>
        <material>
          <ambient>0.15 0.15 0.15 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
      <visual name='charger_wall_v'>
        <pose>0 -0.18 0.175 0 0 0</pose>
        <geometry><box><size>0.35 0.04 0.35</size></box></geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.15 0.15 0.15 1</diffuse>
        </material>
      </visual>
      <collision name='charger_wall_c'>
        <pose>0 -0.18 0.175 0 0 0</pose>
        <geometry><box><size>0.35 0.04 0.35</size></box></geometry>
      </collision>
      <visual name='led_v'>
        <pose>0 -0.165 0.30 0 0 0</pose>
        <geometry><cylinder><radius>0.015</radius><length>0.01</length></cylinder></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{er} {eg} {eb} 1</emissive>
        </material>
      </visual>
      <visual name='contact_pad_v'>
        <pose>0 -0.16 0.06 0 0 0</pose>
        <geometry><box><size>0.20 0.02 0.06</size></box></geometry>
        <material>
          <ambient>1.0 0.5 0.0 1</ambient>
          <diffuse>1.0 0.5 0.0 1</diffuse>
          <emissive>0.5 0.25 0.0 1</emissive>
        </material>
      </visual>
      <visual name='rail_left_v'>
        <pose>0.19 -0.05 0.003 0 0 0</pose>
        <geometry><box><size>0.02 0.30 0.006</size></box></geometry>
        <material>
          <ambient>0.3 0.3 0.0 1</ambient>
          <diffuse>0.5 0.5 0.0 1</diffuse>
        </material>
      </visual>
      <visual name='rail_right_v'>
        <pose>-0.19 -0.05 0.003 0 0 0</pose>
        <geometry><box><size>0.02 0.30 0.006</size></box></geometry>
        <material>
          <ambient>0.3 0.3 0.0 1</ambient>
          <diffuse>0.5 0.5 0.0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def set_dock_led_color(self, r: float, g: float, b: float):
        """Delete charging_dock and respawn it with a new LED color."""
        # Delete existing dock
        del_req = DeleteEntity.Request()
        del_req.name = 'charging_dock'
        del_future = self._delete_entity_cli.call_async(del_req)
        del_future.add_done_callback(
            lambda f: self._respawn_dock(r, g, b))

    def _respawn_dock(self, r: float, g: float, b: float):
        """Callback: respawn dock with new LED color after delete completes."""
        sdf = self._DOCK_SDF_TEMPLATE.format(
            r=r, g=g, b=b,
            er=r * 0.8, eg=g * 0.8, eb=b * 0.8,
        )
        spawn_req = SpawnEntity.Request()
        spawn_req.name = 'charging_dock'
        spawn_req.xml = sdf
        spawn_req.initial_pose.position.x = 0.5
        spawn_req.initial_pose.position.y = -3.1
        spawn_req.initial_pose.position.z = 0.0
        self._spawn_entity_cli.call_async(spawn_req)
        self.get_logger().info(
            f"Dock LED changed to ({r}, {g}, {b})")

    # ── Timer ────────────────────────────────────────────────────────

    def _tick(self):
        if self.phase == self.PHASE_MAPPING:
            self.explorer.tick()
            if self.explorer.state == Explorer.ST_DONE:
                self.phase = self.PHASE_IDLE
                self.get_logger().info("Exploration complete!")
        elif self.phase == self.PHASE_MISSION:
            self.fsm.tick()
        elif self.phase == self.PHASE_MANUAL:
            msg = Twist()
            msg.linear.x = self._manual_lx
            msg.angular.z = self._manual_az
            self._cmd_pub.publish(msg)

    # ── Phase control (called from GUI) ──────────────────────────────

    def start_mapping(self):
        """Start Phase 1: autonomous exploration."""
        self.phase = self.PHASE_MAPPING
        self.explorer.start()

    def stop_mapping(self):
        self.explorer.stop()
        self.phase = self.PHASE_IDLE

    def start_mission(self):
        """Start Phase 2: load map and begin pick-and-place."""
        # Load saved map into navigator
        if not self.navigator.map_loaded:
            if not self.navigator.load_map():
                self.get_logger().warn(
                    "No map -- mission will run without navigation!")
        self.phase = self.PHASE_MISSION
        self.fsm.start_mission()

    def stop_mission(self):
        self.fsm.stop_mission()
        self.phase = self.PHASE_IDLE

    def _srv_start_mission(self, request, response):
        if self.phase == self.PHASE_IDLE:
            self.start_mission()
            response.success = True
            response.message = "Mission started"
        else:
            response.success = False
            response.message = f"Cannot start: phase={self.phase}"
        return response

    def _srv_stop_mission(self, request, response):
        self.stop_mission()
        response.success = True
        response.message = "Mission stopped"
        return response

    def set_manual_mode(self, enabled: bool):
        if enabled:
            if self.phase == self.PHASE_MAPPING:
                self.stop_mapping()
            elif self.phase == self.PHASE_MISSION:
                self.stop_mission()
            self.phase = self.PHASE_MANUAL
        else:
            self._manual_lx = 0.0
            self._manual_az = 0.0
            self._cmd_pub.publish(Twist())
            self.phase = self.PHASE_IDLE

    def manual_cmd(self, lx: float, az: float):
        self._manual_lx = lx
        self._manual_az = az
