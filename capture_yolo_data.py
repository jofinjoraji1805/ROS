#!/usr/bin/env python3
"""
capture_yolo_data.py — Capture multi-class YOLO training data from TurtleBot3's camera.

Teleports the robot to many viewpoints around the room, captures camera frames,
auto-labels all visible objects (cubes + drop zones) via HSV color detection,
saves in YOLO format.

Classes:
  0: red_cube        1: green_cube       2: blue_cube
  3: red_drop_zone   4: green_drop_zone  5: blue_drop_zone

Output: ~/colcon_ws/yoloimages/images/*.jpg  +  yoloimages/labels/*.txt
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

# Class names
CLASS_NAMES = [
    "red_cube", "green_cube", "blue_cube",
    "red_drop_zone", "green_drop_zone", "blue_drop_zone",
]

# ── HSV color ranges ──────────────────────────────────────────────────────
# Cubes are pure red/green/blue; drop zone walls are magenta/lime/cyan.
# Each object type gets its own HSV range → no area-based guessing needed.

# 6 detection channels: (name, class_id, hsv_ranges)
# hsv_ranges is a list of (low, high) tuples (red wraps around H=0/180)
DETECT_CHANNELS = [
    # ── Cubes (pure colors) ──
    ("red_cube",   0, [
        (np.array([0,   100, 100]), np.array([10,  255, 255])),
        (np.array([175, 100, 100]), np.array([180, 255, 255])),
    ]),
    ("green_cube", 1, [
        (np.array([50,  80, 80]),  np.array([75,  255, 255])),
    ]),
    ("blue_cube",  2, [
        (np.array([105, 80, 80]),  np.array([135, 255, 255])),
    ]),
    # ── Drop zones (shifted colors: magenta, lime, cyan) ──
    ("red_drop_zone",   3, [
        (np.array([150, 80, 80]),  np.array([175, 255, 255])),   # magenta/pink
    ]),
    ("green_drop_zone", 4, [
        (np.array([25,  80, 80]),  np.array([48,  255, 255])),   # yellow-lime
    ]),
    ("blue_drop_zone",  5, [
        (np.array([85,  80, 80]),  np.array([103, 255, 255])),   # cyan/turquoise
    ]),
]

# Detection thresholds
MIN_CONTOUR_AREA = 25       # min pixels to count as detection
MIN_BBOX_SIDE    = 4        # min bbox side in pixels


# ── Known object positions (from world file) ─────────────────────────────
# Tables with cubes (left side)
CUBE_POSITIONS = {
    "green_cube": (-0.91,  1.5, 0.175),
    "blue_cube":  (-0.91,  0.0, 0.175),
    "red_cube":   (-0.91, -1.5, 0.175),
}

# Drop zones (right side)
DROP_POSITIONS = {
    "red_drop_zone":   (2.0,  1.5, 0.0),
    "green_drop_zone": (2.0,  0.0, 0.0),
    "blue_drop_zone":  (2.0, -1.5, 0.0),
}

# ── Viewpoint generation ─────────────────────────────────────────────────
def generate_viewpoints():
    """Generate (x, y, yaw) viewpoints for the robot throughout the room."""
    viewpoints = []

    # --- Viewpoints near each cube/table (close-up from multiple angles) ---
    for name, (cx, cy, _) in CUBE_POSITIONS.items():
        for dist in [0.30, 0.45, 0.65, 0.90]:
            for ang_deg in [-60, -30, -10, 0, 10, 30, 60]:
                ang = math.radians(ang_deg)
                rx = cx - dist * math.cos(ang)
                ry = cy - dist * math.sin(ang)
                ryaw = math.atan2(cy - ry, cx - rx)
                viewpoints.append((rx, ry, ryaw))

    # --- Viewpoints near each drop zone (close-up from multiple angles) ---
    for name, (dx, dy, _) in DROP_POSITIONS.items():
        for dist in [0.40, 0.60, 0.90, 1.20]:
            for ang_deg in [-50, -25, 0, 25, 50]:
                ang = math.radians(ang_deg)
                rx = dx - dist * math.cos(ang)
                ry = dy - dist * math.sin(ang)
                ryaw = math.atan2(dy - ry, dx - rx)
                viewpoints.append((rx, ry, ryaw))

    # --- Mid-room viewpoints (see multiple objects) ---
    mid_positions = [
        (0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.5, -1.0),
        (1.0, 0.0), (1.0, 1.0), (1.0, -1.0),
        (0.0, 1.0), (0.0, -1.0), (0.0, 1.5), (0.0, -1.5),
        (-0.3, 0.5), (-0.3, -0.5), (1.5, 0.5), (1.5, -0.5),
    ]
    yaw_angles = [0, math.pi/4, math.pi/2, 3*math.pi/4,
                  math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4]

    for (mx, my) in mid_positions:
        for yaw in yaw_angles:
            viewpoints.append((mx, my, yaw))

    # --- Sweep along center corridor ---
    for x in np.arange(-0.5, 2.0, 0.4):
        for yaw_deg in [-90, -45, 0, 45, 90, 135, 180, -135]:
            viewpoints.append((x, 0.0, math.radians(yaw_deg)))

    return viewpoints


class YoloDataCapture(Node):
    def __init__(self):
        super().__init__("yolo_data_capture")
        self.bridge = CvBridge()
        self.latest_image = None
        self._img_stamp = 0

        self.create_subscription(Image, CAMERA_TOPIC, self._img_cb, 10)

        self.set_state_cli = self.create_client(
            SetEntityState, "/set_entity_state"
        )

        self.get_logger().info("Waiting for Gazebo set_entity_state service...")
        self.set_state_cli.wait_for_service(timeout_sec=30.0)

        # Wait for first camera frame
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
        """Wait for a NEW camera frame (not a stale one)."""
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

    def detect_objects(self, img):
        """
        Detect all colored cubes and drop zones in the image.
        Each object type has its own HSV range — no area guessing needed.
        Returns list of (class_id, x_center, y_center, w_norm, h_norm).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h, img_w = img.shape[:2]
        kernel = np.ones((3, 3), np.uint8)

        detections = []

        for name, cls_id, hsv_ranges in DETECT_CHANNELS:
            # Build mask from all HSV ranges for this channel
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

        viewpoints = generate_viewpoints()
        total = len(viewpoints)
        self.get_logger().info(f"Starting capture: {total} viewpoints")

        saved = 0
        skipped = 0

        for idx, (rx, ry, ryaw) in enumerate(viewpoints):
            # Teleport robot
            self.set_entity_pose(ROBOT_MODEL, rx, ry, 0.0, ryaw)
            self._spin_for(0.4)  # let camera settle

            img = self.grab_fresh_image()
            if img is None:
                skipped += 1
                continue

            detections = self.detect_objects(img)
            if not detections:
                skipped += 1
                continue

            # Save image + labels
            fname = f"frame_{saved:05d}"
            cv2.imwrite(
                os.path.join(img_dir, f"{fname}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                for (cls, xc, yc, wn, hn) in detections:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            saved += 1
            if saved % 50 == 0 or idx == total - 1:
                self.get_logger().info(
                    f"  [{idx+1}/{total}] Saved {saved} images "
                    f"({skipped} skipped, "
                    f"{sum(1 for d in detections if d[0]<3)} cubes, "
                    f"{sum(1 for d in detections if d[0]>=3)} zones in last frame)"
                )

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  DONE: {saved} images saved, {skipped} skipped")
        self.get_logger().info(f"  Images: {img_dir}")
        self.get_logger().info(f"  Labels: {lbl_dir}")
        self.get_logger().info("=" * 60)

        # Write classes file
        with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
            for name in CLASS_NAMES:
                f.write(f"{name}\n")

        # Write dataset YAML for YOLO training
        with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
            f.write(f"path: {OUTPUT_DIR}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            f.write(f"nc: {len(CLASS_NAMES)}\n")
            f.write(f"names: {CLASS_NAMES}\n")

        self.get_logger().info("Wrote classes.txt and dataset.yaml")


def main():
    rclpy.init()
    node = YoloDataCapture()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
