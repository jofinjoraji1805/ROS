#!/usr/bin/env python3
"""
perception.py -- YOLO-based object detection for cubes and drop zones.

Loads a YOLOv8 model and provides:
  - detect(frame)      -> list of Detection
  - find_target(...)   -> (cx, cy, area) normalised or None
  - annotate(frame, detections, hud) -> annotated BGR image
  - build_tasks()      -> ordered list of TaskDef
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .config import YOLO_MODEL, CONF_THRESH


@dataclass
class Detection:
    """Single bounding-box detection."""
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


@dataclass
class TaskDef:
    """One pick-and-place task: pick cube_cls, place at zone_cls."""
    cube_cls: int
    zone_cls: int
    label: str          # e.g. "RED"


class PerceptionModule:
    """YOLO detection wrapper with drawing and target-finding helpers."""

    def __init__(self, model_path: str = YOLO_MODEL, conf: float = CONF_THRESH):
        self.model = YOLO(model_path)
        self.conf = conf
        self.names: Dict[int, str] = self.model.names

        # Reverse lookup: name -> class id
        _n2i = {v: k for k, v in self.names.items()}
        self.CLS_RED_CUBE = _n2i["red_cube"]
        self.CLS_RED_ZONE = _n2i["red_drop_zone"]
        self.CLS_BLUE_CUBE = _n2i["blue_cube"]
        self.CLS_BLUE_ZONE = _n2i["blue_drop_zone"]
        self.CLS_GREEN_CUBE = _n2i["green_cube"]
        self.CLS_GREEN_ZONE = _n2i["green_drop_zone"]
        self.CLS_DOCK = _n2i.get("charging_dock", -1)
        self.CUBE_IDS = {self.CLS_RED_CUBE, self.CLS_BLUE_CUBE, self.CLS_GREEN_CUBE}

        # Drawing colours (BGR)
        _cm = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 100, 0),
               "charging": (0, 165, 255)}
        self.draw_colors: Dict[int, Tuple[int, ...]] = {}
        for cid, name in self.names.items():
            for key, bgr in _cm.items():
                if key in name:
                    self.draw_colors[cid] = bgr
                    break

    # ── Detection ─────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO inference on a BGR frame, return list of Detection."""
        results = self.model(frame, conf=self.conf, verbose=False)
        dets: List[Detection] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                dets.append(Detection(
                    int(box.cls[0]), x1, y1, x2, y2, float(box.conf[0])
                ))
        return dets

    # ── Target finding ────────────────────────────────────────────────

    def find_target(
        self,
        dets: List[Detection],
        target_cls: int,
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find highest-confidence detection of target_cls.
        Returns (cx, cy, area) all normalised to [0, 1], or None.
        Filters out impossibly large cubes and impossibly tiny zones.
        """
        img_area = img_w * img_h
        is_cube = target_cls in self.CUBE_IDS
        is_dock = target_cls == self.CLS_DOCK
        best: Optional[Detection] = None
        best_conf = 0.0

        for d in dets:
            if d.cls_id != target_cls or d.conf <= best_conf:
                continue
            area = ((d.x2 - d.x1) * (d.y2 - d.y1)) / img_area
            if not is_dock:
                if is_cube and area > 0.15:
                    continue
                if not is_cube and area < 0.001:
                    continue
            best = d
            best_conf = d.conf

        if best is None:
            return None

        cx = ((best.x1 + best.x2) / 2.0) / img_w
        cy = ((best.y1 + best.y2) / 2.0) / img_h
        area = ((best.x2 - best.x1) * (best.y2 - best.y1)) / img_area
        return cx, cy, area

    @staticmethod
    def area_to_distance(area: float) -> str:
        """Human-readable distance estimate from normalised bbox area."""
        if area > 0.03:
            return "At Table"
        if area > 0.01:
            return "Very Close"
        if area > 0.005:
            return "Close"
        if area > 0.002:
            return "Medium"
        if area > 0.001:
            return "Far"
        return "Very Far"

    # ── Annotation ────────────────────────────────────────────────────

    def annotate(
        self,
        frame: np.ndarray,
        dets: List[Detection],
        hud: str = "",
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and optional HUD text on frame."""
        vis = frame.copy()
        for d in dets:
            c = self.draw_colors.get(d.cls_id, (255, 255, 255))
            cv2.rectangle(vis, (int(d.x1), int(d.y1)),
                          (int(d.x2), int(d.y2)), c, 2)
            lbl = f"{self.names[d.cls_id]} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (int(d.x1), int(d.y1) - th - 6),
                          (int(d.x1) + tw, int(d.y1)), c, -1)
            cv2.putText(vis, lbl, (int(d.x1), int(d.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if hud:
            cv2.putText(vis, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return vis

    def annotate_dock(
        self,
        frame: np.ndarray,
        dets: List[Detection],
        hud: str = "",
    ) -> np.ndarray:
        """Replace all detections with a single 'charging_dock' label."""
        vis = frame.copy()
        # Use bounding box of the largest detection as the dock region
        best = max(dets, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
        c = (0, 200, 0)  # green
        cv2.rectangle(vis, (int(best.x1), int(best.y1)),
                      (int(best.x2), int(best.y2)), c, 2)
        lbl = "charging_dock"
        (tw, th), _ = cv2.getTextSize(
            lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (int(best.x1), int(best.y1) - th - 6),
                      (int(best.x1) + tw, int(best.y1)), c, -1)
        cv2.putText(vis, lbl, (int(best.x1), int(best.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if hud:
            cv2.putText(vis, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return vis

    # ── Task list ─────────────────────────────────────────────────────

    def build_tasks(self) -> List[TaskDef]:
        """Return ordered list of pick-and-place tasks: RED -> BLUE -> GREEN."""
        return [
            TaskDef(self.CLS_RED_CUBE, self.CLS_RED_ZONE, "RED"),
            TaskDef(self.CLS_BLUE_CUBE, self.CLS_BLUE_ZONE, "BLUE"),
            TaskDef(self.CLS_GREEN_CUBE, self.CLS_GREEN_ZONE, "GREEN"),
        ]
