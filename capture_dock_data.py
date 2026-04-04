#!/usr/bin/env python3
"""
capture_dock_data.py — Capture YOLO training data for the charging dock.

Teleports the robot to viewpoints around the charging dock, captures camera
frames, auto-labels the dock via the orange contact pad (HSV detection),
and appends to the existing YOLO dataset with class 6: charging_dock.

Run while Gazebo is already running (gzserver + world loaded).

Usage:
  ros2 run tb3_pick_place capture_dock_data
  OR
  python3 capture_dock_data.py
"""

import os
import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Point, Quaternion, Twist
from cv_bridge import CvBridge


# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/home/jofin/colcon_ws/yoloimages"
ROBOT_MODEL = "turtlebot3_manipulation_system"
CAMERA_TOPIC = "/pi_camera/image_raw"

# charging_dock = class 6 (appended after the existing 6 classes)
DOCK_CLASS_ID = 6
DOCK_CLASS_NAME = "charging_dock"

# All class names (original 6 + dock)
ALL_CLASS_NAMES = [
    "red_cube", "green_cube", "blue_cube",
    "red_drop_zone", "green_drop_zone", "blue_drop_zone",
    "charging_dock",
]

# ── HSV ranges ────────────────────────────────────────────────────────────
# Orange contact pad on the dock: H ~8-25, high S, high V
DOCK_HSV_RANGES = [
    (np.array([8, 100, 100]), np.array([25, 255, 255])),
]

# Also detect existing objects so we get multi-class labels in images
# that show both the dock and other objects
DETECT_CHANNELS = [
    ("red_cube", 0, [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([175, 100, 100]), np.array([180, 255, 255])),
    ]),
    ("green_cube", 1, [
        (np.array([50, 80, 80]), np.array([75, 255, 255])),
    ]),
    ("blue_cube", 2, [
        (np.array([105, 80, 80]), np.array([135, 255, 255])),
    ]),
    ("red_drop_zone", 3, [
        (np.array([150, 80, 80]), np.array([175, 255, 255])),
    ]),
    ("green_drop_zone", 4, [
        (np.array([25, 80, 80]), np.array([48, 255, 255])),
    ]),
    ("blue_drop_zone", 5, [
        (np.array([85, 80, 80]), np.array([103, 255, 255])),
    ]),
]

MIN_CONTOUR_AREA = 25
MIN_BBOX_SIDE = 4

# Dock position in world
DOCK_X, DOCK_Y = 0.5, -3.1

# Bounding box expansion: the orange pad is small, expand to cover full dock
BBOX_EXPAND_X = 1.8   # multiply width
BBOX_EXPAND_Y = 3.0   # multiply height (dock is taller than the pad)
BBOX_SHIFT_Y = -0.3   # shift bbox upward (pad is at bottom of dock)


# ── Viewpoint generation ─────────────────────────────────────────────────
def generate_dock_viewpoints():
    """Generate viewpoints around the charging dock from many angles/distances."""
    viewpoints = []

    # Close-up views from front (robot approaches from +Y direction)
    for dist in [0.35, 0.50, 0.70, 1.0, 1.3, 1.6]:
        for ang_deg in [-50, -35, -20, -10, 0, 10, 20, 35, 50]:
            ang = math.radians(ang_deg)
            # Dock is at (0.5, -3.1), robot approaches from north (+Y)
            rx = DOCK_X + dist * math.sin(ang)
            ry = DOCK_Y + dist  # north of dock
            ryaw = math.atan2(DOCK_Y - ry, DOCK_X - rx)
            viewpoints.append((rx, ry, ryaw))

    # Side views
    for dist in [0.5, 0.8, 1.2]:
        for side_offset in [-0.3, -0.6, 0.3, 0.6]:
            rx = DOCK_X + side_offset
            ry = DOCK_Y + dist
            ryaw = math.atan2(DOCK_Y - ry, DOCK_X - rx)
            viewpoints.append((rx, ry, ryaw))

    # Far views (robot sees dock from various room positions)
    far_positions = [
        (0.5, -2.0), (0.5, -1.5), (0.0, -2.0), (1.0, -2.0),
        (0.0, -1.5), (1.0, -1.5), (0.5, -1.0),
        (-0.3, -2.0), (1.3, -2.0),
    ]
    for (fx, fy) in far_positions:
        ryaw = math.atan2(DOCK_Y - fy, DOCK_X - fx)
        viewpoints.append((fx, fy, ryaw))
        # Slight angle variations
        for offset_deg in [-15, 15]:
            viewpoints.append((fx, fy, ryaw + math.radians(offset_deg)))

    return viewpoints


class DockDataCapture(Node):
    def __init__(self):
        super().__init__("dock_data_capture")
        self.bridge = CvBridge()
        self.latest_image = None
        self._img_stamp = 0

        self.create_subscription(Image, CAMERA_TOPIC, self._img_cb, 10)
        self.set_state_cli = self.create_client(
            SetEntityState, "/set_entity_state"
        )

        self.get_logger().info("Waiting for Gazebo set_entity_state service...")
        self.set_state_cli.wait_for_service(timeout_sec=30.0)

        self.get_logger().info(f"Waiting for camera on {CAMERA_TOPIC}...")
        t0 = time.time()
        while self.latest_image is None and time.time() - t0 < 30.0:
            rclpy.spin_once(self, timeout_sec=0.2)
        if self.latest_image is None:
            raise RuntimeError(f"No image on {CAMERA_TOPIC} after 30s")
        self.get_logger().info("Camera ready.")

    def _img_cb(self, msg):
        self.latest_image = msg
        self._img_stamp = time.time()

    def _spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def set_entity_pose(self, name, x, y, z, yaw=0.0):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = name
        state.pose.position = Point(x=float(x), y=float(y), z=float(z))
        state.pose.orientation = Quaternion(
            x=0.0, y=0.0,
            z=float(math.sin(yaw / 2)),
            w=float(math.cos(yaw / 2)),
        )
        state.twist = Twist()
        state.reference_frame = "world"
        req.state = state
        fut = self.set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

    def grab_fresh_image(self):
        deadline = time.time() + 2.0
        old_stamp = self._img_stamp
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._img_stamp > old_stamp:
                break
        if self.latest_image is None:
            return None
        try:
            return self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return None

    def detect_dock(self, img):
        """Detect the charging dock via orange contact pad HSV."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h, img_w = img.shape[:2]
        kernel = np.ones((3, 3), np.uint8)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in DOCK_HSV_RANGES:
            mask |= cv2.inRange(hsv, low, high)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
                continue

            # Expand bbox to cover full dock (not just the orange pad)
            cx = x + w / 2.0
            cy = y + h / 2.0
            new_w = w * BBOX_EXPAND_X
            new_h = h * BBOX_EXPAND_Y
            cy = cy + BBOX_SHIFT_Y * new_h  # shift up

            # Clamp to image bounds
            x1 = max(0, cx - new_w / 2.0)
            y1 = max(0, cy - new_h / 2.0)
            x2 = min(img_w, cx + new_w / 2.0)
            y2 = min(img_h, cy + new_h / 2.0)

            final_w = x2 - x1
            final_h = y2 - y1
            if final_w < MIN_BBOX_SIDE or final_h < MIN_BBOX_SIDE:
                continue

            x_center = (x1 + final_w / 2.0) / img_w
            y_center = (y1 + final_h / 2.0) / img_h
            w_norm = final_w / img_w
            h_norm = final_h / img_h

            detections.append((DOCK_CLASS_ID, x_center, y_center, w_norm, h_norm))

        return detections

    def detect_other_objects(self, img):
        """Detect cubes and drop zones (same as main capture script)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h, img_w = img.shape[:2]
        kernel = np.ones((3, 3), np.uint8)
        detections = []

        for name, cls_id, hsv_ranges in DETECT_CHANNELS:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low, high in hsv_ranges:
                mask |= cv2.inRange(hsv, low, high)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
                    continue
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                detections.append((cls_id, x_center, y_center, w_norm, h_norm))

        return detections

    def run(self):
        img_dir = os.path.join(OUTPUT_DIR, "images")
        lbl_dir = os.path.join(OUTPUT_DIR, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Find next frame index (append to existing dataset)
        existing = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        start_idx = len(existing)
        self.get_logger().info(
            f"Existing dataset: {start_idx} images. "
            f"Appending dock data starting at frame_{start_idx:05d}"
        )

        viewpoints = generate_dock_viewpoints()
        total = len(viewpoints)
        self.get_logger().info(f"Capturing {total} dock viewpoints...")

        saved = 0
        skipped = 0

        for idx, (rx, ry, ryaw) in enumerate(viewpoints):
            self.set_entity_pose(ROBOT_MODEL, rx, ry, 0.0, ryaw)
            self._spin_for(0.4)

            img = self.grab_fresh_image()
            if img is None:
                skipped += 1
                continue

            dock_dets = self.detect_dock(img)
            if not dock_dets:
                skipped += 1
                continue

            # Also detect other objects visible in the frame
            other_dets = self.detect_other_objects(img)
            all_dets = dock_dets + other_dets

            # Save
            frame_idx = start_idx + saved
            fname = f"frame_{frame_idx:05d}"
            cv2.imwrite(
                os.path.join(img_dir, f"{fname}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                for (cls, xc, yc, wn, hn) in all_dets:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            saved += 1
            if saved % 20 == 0 or idx == total - 1:
                self.get_logger().info(
                    f"  [{idx+1}/{total}] Saved {saved} dock images "
                    f"({skipped} skipped)"
                )

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  DONE: {saved} dock images added to dataset")
        self.get_logger().info(f"  Total dataset: {start_idx + saved} images")
        self.get_logger().info(f"  Images: {img_dir}")
        self.get_logger().info(f"  Labels: {lbl_dir}")
        self.get_logger().info("=" * 60)

        # Update classes.txt and dataset.yaml with 7 classes
        with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
            for name in ALL_CLASS_NAMES:
                f.write(f"{name}\n")

        with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
            f.write(f"path: {OUTPUT_DIR}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            f.write(f"nc: {len(ALL_CLASS_NAMES)}\n")
            f.write(f"names: {ALL_CLASS_NAMES}\n")

        self.get_logger().info(
            "Updated classes.txt and dataset.yaml (7 classes including charging_dock)"
        )

        # Reset robot to spawn position
        self.set_entity_pose(ROBOT_MODEL, 0.5, -2.0, 0.0, math.pi / 2)
        self.get_logger().info("Robot reset to spawn position.")


def main():
    rclpy.init()
    node = DockDataCapture()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
