import io
import math
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

from app.schemas.response import DetectedObject


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) for [ymin, xmin, ymax, xmax]."""
    y1_a, x1_a, y2_a, x2_a = box_a
    y1_b, x1_b, y2_b, x2_b = box_b

    inter_y1 = max(y1_a, y1_b)
    inter_x1 = max(x1_a, x1_b)
    inter_y2 = min(y2_a, y2_b)
    inter_x2 = min(x2_a, x2_b)

    if inter_y2 <= inter_y1 or inter_x2 <= inter_x1:
        return 0.0

    inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
    area_a = (y2_a - y1_a) * (x2_a - x1_a)
    area_b = (y2_b - y1_b) * (x2_b - x1_b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def apply_nms(boxes: List[Dict[str, Any]], iou_threshold: float = 0.35) -> List[Dict[str, Any]]:
    """Applies Non-Maximum Suppression to eliminate duplicate overlapping boxes."""
    if not boxes:
        return []

    # Sort boxes by score descending
    sorted_boxes = sorted(boxes, key=lambda b: b.get("score", 0.0), reverse=True)
    kept_boxes = []

    while sorted_boxes:
        best = sorted_boxes.pop(0)
        kept_boxes.append(best)
        sorted_boxes = [
            b for b in sorted_boxes
            if compute_iou(best["box"], b["box"]) < iou_threshold
        ]

    return kept_boxes


class GroundingEngine:
    """
    Query-specific visual grounding engine for remote-sensing imagery.
    Extracts discrete, localized regions matching specific query concepts.
    Prevents giant generic bounding boxes by enforcing footprint geometry constraints.
    """

    def __init__(self):
        pass

    def ground_target_concept(
        self,
        image: Image.Image,
        target_concept: str,
        vlm_affirmative: bool = True,
        max_detections: int = 20
    ) -> List[DetectedObject]:
        """
        Locates target objects/features in the image matching the query.
        Returns discrete [ymin, xmin, ymax, xmax] bounding boxes.
        """
        if not vlm_affirmative:
            # If VLM confirmed absence, do not fabricate visual detections
            return []

        width, height = image.size
        img_np = np.array(image.convert("RGB"))
        candidates: List[Dict[str, Any]] = []

        target = target_concept.lower()

        if target in ["building", "structure", "facility", "house"]:
            candidates = self._detect_building_footprints(img_np, width, height)
        elif target in ["road", "highway", "runway", "track"]:
            candidates = self._detect_road_corridors(img_np, width, height)
        elif target in ["ship", "vessel", "boat", "watercraft"]:
            candidates = self._detect_marine_vessels(img_np, width, height)
        elif target in ["vegetation", "crop", "forest", "field"]:
            candidates = self._detect_vegetation_zones(img_np, width, height)
        elif target in ["water", "river", "lake", "ocean"]:
            candidates = self._detect_water_bodies(img_np, width, height)
        elif target in ["vehicle", "car", "truck"]:
            candidates = self._detect_small_vehicles(img_np, width, height)
        else:
            # Default to high-contrast spatial feature grounding
            candidates = self._detect_salient_regions(img_np, width, height)

        # Apply Non-Maximum Suppression to eliminate duplicates
        filtered = apply_nms(candidates, iou_threshold=0.35)

        # Limit to top detections
        filtered = filtered[:max_detections]

        # Convert to schema
        detections: List[DetectedObject] = []
        for c in filtered:
            detections.append(
                DetectedObject(
                    label=c.get("label", target_concept),
                    score=round(c.get("score", 0.85), 3),
                    confidence=round(c.get("score", 0.85), 3),
                    box=c["box"]
                )
            )

        return detections

    def _detect_building_footprints(
        self,
        img_np: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict[str, Any]]:
        """
        Identifies discrete structure footprints using morphological gradient & contrast analysis.
        Enforces realistic building footprint aspect ratios and size constraints.
        """
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        total_area = float(width * height)

        # Sobel edge gradient magnitude
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        candidates = []
        tile_w = max(40, width // 10)
        tile_h = max(40, height // 10)

        for y in range(0, height - tile_h, tile_h // 2):
            for x in range(0, width - tile_w, tile_w // 2):
                patch_gray = gray[y:y+tile_h, x:x+tile_w]
                patch_grad = grad_mag[y:y+tile_h, x:x+tile_w]

                if patch_gray.size == 0:
                    continue

                local_std = float(np.std(patch_gray))
                max_edge = float(np.max(patch_grad))

                # Structure footprints have distinct contrast against local background and sharp edges
                if local_std > 12.0 and max_edge > 18.0:
                    edge_mask = patch_grad > max(12.0, max_edge * 0.40)
                    active_y, active_x = np.where(edge_mask)

                    if len(active_y) >= 8:
                        sub_ymin = y + int(np.min(active_y))
                        sub_ymax = y + int(np.max(active_y))
                        sub_xmin = x + int(np.min(active_x))
                        sub_xmax = x + int(np.max(active_x))

                        bw = max(10, sub_xmax - sub_xmin)
                        bh = max(10, sub_ymax - sub_ymin)
                        b_area = bw * bh

                        # Filter: area between 0.05% and 15% of image, aspect ratio between 0.2 and 5.0
                        if (0.0005 * total_area) <= b_area <= (0.15 * total_area):
                            aspect = float(bw) / float(bh)
                            if 0.20 <= aspect <= 5.0:
                                score = min(0.95, 0.72 + (local_std / 120.0) * 0.22)
                                candidates.append({
                                    "label": "structure",
                                    "box": [float(sub_ymin), float(sub_xmin), float(sub_ymax), float(sub_xmax)],
                                    "score": round(score, 3)
                                })

        return candidates

    def _detect_road_corridors(
        self,
        img_np: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict[str, Any]]:
        """
        Identifies linear transportation infrastructure (roads, runways, highways)
        characterized by continuous linear corridors and flanking edge gradients.
        """
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        candidates = []

        # Horizontal corridors
        for y in range(10, height - 30, 4):
            for bw in range(25, 75, 10):
                if y + bw < height:
                    edge_top = float(np.mean(grad_mag[y, :] > 14.0))
                    edge_bot = float(np.mean(grad_mag[y + bw, :] > 14.0))
                    if edge_top > 0.25 and edge_bot > 0.25:
                        score = min(0.94, 0.75 + (edge_top + edge_bot) * 0.15)
                        candidates.append({
                            "label": "road corridor",
                            "box": [float(y), 0.0, float(y + bw), float(width)],
                            "score": round(score, 3)
                        })

        # Vertical corridors
        for x in range(10, width - 30, 4):
            for bw in range(25, 75, 10):
                if x + bw < width:
                    edge_left = float(np.mean(grad_mag[:, x] > 14.0))
                    edge_right = float(np.mean(grad_mag[:, x + bw] > 14.0))
                    if edge_left > 0.25 and edge_right > 0.25:
                        score = min(0.94, 0.75 + (edge_left + edge_right) * 0.15)
                        candidates.append({
                            "label": "road corridor",
                            "box": [0.0, float(x), float(height), float(x + bw)],
                            "score": round(score, 3)
                        })

        # Localized linear segments fallback if full-span roads are absent
        if not candidates:
            step = min(width, height) // 5
            for y in range(0, height - step, step):
                for x in range(0, width - step, step):
                    patch_grad = grad_mag[y:y+step, x:x+step]
                    if np.mean(patch_grad > 14.0) > 0.18:
                        candidates.append({
                            "label": "road segment",
                            "box": [float(y), float(x), float(y + step), float(x + step)],
                            "score": 0.81
                        })

        return candidates

    def _detect_marine_vessels(
        self,
        img_np: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict[str, Any]]:
        """
        Identifies vessels/ships as high-contrast compact anomalies over water backgrounds.
        """
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        med = np.median(gray)
        # Reflective hull anomaly
        anomaly = np.abs(gray - med)
        thresh = np.percentile(anomaly, 95)

        candidates = []
        box_size = max(16, min(width, height) // 10)

        for y in range(0, height - box_size, box_size):
            for x in range(0, width - box_size, box_size):
                patch = anomaly[y:y+box_size, x:x+box_size]
                if np.mean(patch > thresh) > 0.15:
                    candidates.append({
                        "label": "vessel",
                        "box": [float(y), float(x), float(y + box_size), float(x + box_size)],
                        "score": 0.84
                    })

        return candidates

    def _detect_vegetation_zones(
        self,
        img_np: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict[str, Any]]:
        """
        Isolates vegetation / canopy / crop zones using Visible Green Difference Index (VGDI).
        """
        r = img_np[..., 0].astype(float)
        g = img_np[..., 1].astype(float)
        b = img_np[..., 2].astype(float)

        vgdi = (2 * g - r - b) / (2 * g + r + b + 1e-6)
        veg_mask = vgdi > 0.08

        candidates = []
        step_w = max(32, width // 4)
        step_h = max(32, height // 4)

        for y in range(0, height - step_h, step_h):
            for x in range(0, width - step_w, step_w):
                veg_ratio = np.mean(veg_mask[y:y+step_h, x:x+step_w])
                if veg_ratio > 0.35:
                    candidates.append({
                        "label": "vegetation zone",
                        "box": [float(y), float(x), float(y + step_h), float(x + step_w)],
                        "score": min(0.95, 0.75 + float(veg_ratio) * 0.2)
                    })

        return candidates

    def _detect_water_bodies(
        self,
        img_np: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict[str, Any]]:
        """
        Detects low-radiance, smooth surface water bodies.
        """
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        dark_thresh = np.percentile(gray, 25)
        candidates = []
        step = max(32, min(width, height) // 4)

        for y in range(0, height - step, step):
            for x in range(0, width - step, step):
                patch = gray[y:y+step, x:x+step]
                if np.mean(patch) < dark_thresh and np.std(patch) < 15.0:
                    candidates.append({
                        "label": "water body",
                        "box": [float(y), float(x), float(y + step), float(x + step)],
                        "score": 0.87
                    })

        return candidates

    def _detect_small_vehicles(
        self,
        img_np: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict[str, Any]]:
        """Detects sub-30px vehicle footprints."""
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx**2 + gy**2)
        thresh = np.percentile(mag, 92)

        candidates = []
        v_size = max(12, min(width, height) // 25)

        for y in range(0, height - v_size, v_size):
            for x in range(0, width - v_size, v_size):
                patch = mag[y:y+v_size, x:x+v_size]
                if np.mean(patch > thresh) > 0.25:
                    candidates.append({
                        "label": "vehicle",
                        "box": [float(y), float(x), float(y + v_size), float(x + v_size)],
                        "score": 0.81
                    })

        return candidates

    def _detect_salient_regions(
        self,
        img_np: np.ndarray,
        width: int,
        height: int
    ) -> List[Dict[str, Any]]:
        """Fallback: captures distinct visual feature clusters without giant boxes."""
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx**2 + gy**2)
        thresh = np.percentile(mag, 85)

        candidates = []
        step = max(32, min(width, height) // 5)

        for y in range(0, height - step, step):
            for x in range(0, width - step, step):
                patch = mag[y:y+step, x:x+step]
                density = np.mean(patch > thresh)
                if density > 0.20:
                    candidates.append({
                        "label": "identified feature",
                        "box": [float(y), float(x), float(y + step), float(x + step)],
                        "score": 0.83
                    })

        return candidates


grounding_engine = GroundingEngine()
