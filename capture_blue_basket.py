#!/usr/bin/env python3
"""
Capture YOLO training images of the blue basket using robot's own camera.
Teleports robot via 'gz model' with settling delays between moves.
"""

import os
import math
import time
import subprocess

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

BASKET_POS = (-1.0, 1.0)
ROBOT_MODEL = "turtlebot3_manipulation_system"
IMAGE_TOPIC = "/pi_camera/image_raw"
IMG_W, IMG_H = 640, 480

OUTPUT_DIR = "/home/jofin/colcon_ws/yoloimages"
CLASS_ID = 1  # 0=red_box, 1=blue_box


def teleport(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", ROBOT_MODEL,
         "-x", f"{x:.4f}", "-y", f"{y:.4f}", "-z", "0.01",
         "-Y", f"{yaw:.4f}"],
        capture_output=True, timeout=5
    )


class BasketCapture(Node):
    def __init__(self):
        super().__init__("basket_capture")
        self.bridge = CvBridge()
        self.latest_image = None
        self.sub = self.create_subscription(Image, IMAGE_TOPIC, self._cb, 1)

    def _cb(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def flush_and_grab(self):
        """Long settle, flush stale frames, grab fresh one."""
        time.sleep(0.8)
        self.latest_image = None
        # Flush stale
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)
        # Grab fresh
        self.latest_image = None
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_image is not None:
                return self.latest_image.copy()
        return None

    @staticmethod
    def detect_blue(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 40, 30]), np.array([140, 255, 255]))
        kern = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None
        x, y, w, h = cv2.boundingRect(largest)
        cx = (x + w / 2.0) / IMG_W
        cy = (y + h / 2.0) / IMG_H
        nw = w / IMG_W
        nh = h / IMG_H
        return (cx, cy, nw, nh)

    def run(self):
        # Wait for camera
        self.get_logger().info("Waiting for camera topic...")
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.latest_image is not None:
                break
        if self.latest_image is None:
            self.get_logger().error("No camera images. Exiting.")
            return

        img_dir = os.path.join(OUTPUT_DIR, "images")
        lbl_dir = os.path.join(OUTPUT_DIR, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        existing = [f for f in os.listdir(img_dir) if f.startswith("bluebox_")]
        idx = len(existing)

        # Positions around basket — fewer but with good variety
        distances = [0.40, 0.55, 0.70, 0.90, 1.1, 1.4]
        angle_steps = list(range(0, 360, 20))       # every 20°
        yaw_offsets = [0, -12, 12]                   # 3 offsets

        positions = []
        for d in distances:
            for a_deg in angle_steps:
                for yo in yaw_offsets:
                    a = math.radians(a_deg)
                    rx = BASKET_POS[0] + d * math.cos(a)
                    ry = BASKET_POS[1] + d * math.sin(a)
                    if abs(rx) > 2.0 or abs(ry) > 2.0:
                        continue
                    face = math.atan2(BASKET_POS[1] - ry, BASKET_POS[0] - rx)
                    face += math.radians(yo)
                    positions.append((rx, ry, face))

        total = len(positions)
        self.get_logger().info(f"Positions to try: {total}")

        saved = 0
        for i, (rx, ry, yaw) in enumerate(positions):
            teleport(rx, ry, yaw)

            img = self.flush_and_grab()
            if img is None:
                continue

            bbox = self.detect_blue(img)
            if bbox is None:
                continue

            bcx, bcy, bnw, bnh = bbox
            if bnw < 0.02 or bnh < 0.02:
                continue
            if bnw > 0.97 and bnh > 0.97:
                continue

            fname = f"bluebox_{idx:04d}"
            cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), img)
            with open(os.path.join(lbl_dir, f"{fname}.txt"), "w") as f:
                f.write(f"{CLASS_ID} {bcx:.6f} {bcy:.6f} {bnw:.6f} {bnh:.6f}\n")
            idx += 1
            saved += 1

            if saved % 25 == 0:
                self.get_logger().info(f"  ... {saved} images saved ({i+1}/{total} positions)")

        self.get_logger().info(f"DONE — {saved} blue-basket images saved to {OUTPUT_DIR}")
        # Move robot back to origin
        teleport(0.0, 0.0, 0.0)


def main():
    rclpy.init()
    node = BasketCapture()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
