#!/usr/bin/env python3
"""
main.py -- Entry point for the autonomous YOLO pick-and-place system.

Initialises ROS 2, starts the robot controller node in a background
thread, then launches the PyQt5 GUI on the main thread.

Usage:
  ros2 run tb3_pick_place yolo_pick_place
  # or directly:
  python3 -m tb3_pick_place.main
"""

import sys
import os
import threading

# Ensure Qt platform plugins are found
os.environ.setdefault(
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
)

import rclpy
from PyQt5.QtWidgets import QApplication

from .robot_node import RobotController
from .gui import PickPlaceGUI


def main():
    rclpy.init()
    controller = RobotController()

    # Spin ROS in a daemon thread so the GUI can run on main thread
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(controller,), daemon=True)
    spin_thread.start()

    # Launch PyQt5 GUI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    gui = PickPlaceGUI(controller)
    gui.show()

    exit_code = app.exec_()

    # Cleanup
    controller.motion.stop()
    controller.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
