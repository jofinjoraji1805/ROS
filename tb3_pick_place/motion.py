#!/usr/bin/env python3
"""
motion.py -- Smooth base velocity controller for TurtleBot3.

Publishes Twist messages to /cmd_vel with velocity ramping to prevent
jerky motion and wheel slip.
"""

from geometry_msgs.msg import Twist

from .config import MAX_ANG_VEL, VEL_RAMP


class MotionController:
    """Publishes velocity commands with acceleration ramping."""

    LX_MIN = -0.10
    LX_MAX = 0.12

    def __init__(self, publisher):
        self._pub = publisher
        self._prev_lx = 0.0
        self._prev_az = 0.0

    def publish(self, lx: float = 0.0, az: float = 0.0):
        """
        Publish a velocity command with ramping applied to linear.x.
        Angular is clamped but not ramped (rotation needs responsiveness).
        """
        lx = max(self.LX_MIN, min(self.LX_MAX, float(lx)))
        az = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, float(az)))

        # Ramp linear velocity
        dlx = lx - self._prev_lx
        if abs(dlx) > VEL_RAMP:
            lx = self._prev_lx + VEL_RAMP * (1.0 if dlx > 0 else -1.0)

        self._prev_lx = lx
        self._prev_az = az

        msg = Twist()
        msg.linear.x = lx
        msg.angular.z = az
        self._pub.publish(msg)

    def stop(self):
        """Immediately stop all motion."""
        self._prev_lx = 0.0
        self._prev_az = 0.0
        self._pub.publish(Twist())
