#!/usr/bin/env python3
"""
autonomous_pick_place.py -- Standalone launcher for the TurtleBot3
autonomous YOLO pick-and-place system.

This script can be run directly:
    python3 autonomous_pick_place.py

Or installed via colcon and run as:
    ros2 run tb3_pick_place yolo_pick_place

The system operates in two phases:
  Phase 1 (MAPPING): Robot autonomously explores the room using predefined
      waypoints, building a SLAM map from LIDAR data, and saves it to disk.
  Phase 2 (MISSION): Robot loads the saved map, uses A* path planning to
      navigate to each cube (RED -> BLUE -> GREEN), picks it using YOLO
      visual servo + manipulator control, navigates to the corresponding
      drop zone, and places the cube. Repeats for all three cubes.

The PyQt5 GUI provides:
  - Live camera feed with YOLO detection overlay
  - SLAM / navigation map with robot position and planned path
  - Mission control buttons (Build Map, Start Mission, Stop, Manual Mode)
  - Manual WASD teleop
  - Real-time state and log display
"""

import sys
import os

# Ensure Qt platform plugins are found
os.environ.setdefault(
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
)

# Add the package to the path if running standalone (not installed)
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from tb3_pick_place.main import main

if __name__ == "__main__":
    main()
