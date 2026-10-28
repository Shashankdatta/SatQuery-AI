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


def apply_nms(boxes: List[Dict[str, Any]], iou_threshold: float = 0.18) -> List[Dict[str, Any]]:
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
        filtered = apply_nms(candidates, iou_threshold=0.18)

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
        Identifies discrete structure footprints using connected-component analysis,
        negative water/vegetation masking, and geometric aspect ratio constraints.
        Enforces verified instance separation and merges overlapping duplicate boxes.
        """
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        total_pixels = float(width * height)

        # 1. Negative Mask Filters
        # Water mask (high blue relative to red, dark surface; cast to int to prevent overflow)
        is_water = (img_np[..., 2].astype(int) > img_np[..., 0].astype(int) + 15) & (gray < 110) & (img_np[..., 1].astype(int) < 135)

        # Vegetation mask (VGDI > 0.08)
        r = img_np[..., 0].astype(float)
        g = img_np[..., 1].astype(float)
        b = img_np[..., 2].astype(float)
        vgdi = (2 * g - r - b) / (2 * g + r + b + 1e-6)
        is_veg = vgdi > 0.08

        # Background median
        bg_med = np.median(gray)
        contrast = np.abs(gray - bg_med)

        # Structure mask: distinct contrast against ground, not water, not vegetation
        structure_mask = (contrast > 18.0) & (~is_water) & (~is_veg)

        # 2. Connected Component Instance Extraction via BFS
        step = 2
        mask_sub = structure_mask[::step, ::step]
        sub_h, sub_w = mask_sub.shape
        sub_visited = np.zeros((sub_h, sub_w), dtype=bool)
        candidate_instances: List[Dict[str, Any]] = []

        for sy in range(sub_h):
            for sx in range(sub_w):
                if mask_sub[sy, sx] and not sub_visited[sy, sx]:
                    queue = [(sy, sx)]
                    sub_visited[sy, sx] = True
                    pts = []
                    while queue:
                        cy, cx = queue.pop()
                        pts.append((cy * step, cx * step))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < sub_h and 0 <= nx < sub_w:
                                if mask_sub[ny, nx] and not sub_visited[ny, nx]:
                                    sub_visited[ny, nx] = True
                                    queue.append((ny, nx))

                    if len(pts) >= 12:
                        ys = [p[0] for p in pts]
                        xs = [p[1] for p in pts]
                        ymin, ymax = min(ys), max(ys)
                        xmin, xmax = min(xs), max(xs)
                        bw = xmax - xmin
                        bh = ymax - ymin
                        area = bw * bh
                        aspect = max(bw, bh) / max(1, min(bw, bh))
                        pixel_count = len(pts) * (step * step)
                        fill = pixel_count / max(1.0, float(area))

                        # Enforce building instance constraints:
                        # - Minimum footprint dimensions
                        # - Aspect ratio <= 3.2 (excludes linear infrastructure like roads)
                        # - Fill factor >= 0.35 (excludes sparse background textures)
                        # - Area <= 12% of scene (excludes broad regional terrain)
                        if (
                            18 <= bw <= 320 and
                            18 <= bh <= 280 and
                            aspect <= 3.2 and
                            fill >= 0.35 and
                            area <= (0.12 * total_pixels)
                        ):
                            candidate_instances.append({
                                "label": "building",
                                "box": [float(ymin), float(xmin), float(ymax), float(xmax)],
                                "score": 0.88,
                                "area": area
                            })

        # 3. Aggressive Duplicate Removal & Envelope Merging (IoU > 0.15)
        merged: List[Dict[str, Any]] = []
        for cand in candidate_instances:
            matched = False
            for existing in merged:
                y1_a, x1_a, y2_a, x2_a = cand["box"]
                y1_b, x1_b, y2_b, x2_b = existing["box"]

                inter_y1 = max(y1_a, y1_b)
                inter_x1 = max(x1_a, x1_b)
                inter_y2 = min(y2_a, y2_b)
                inter_x2 = min(x2_a, x2_b)

                if inter_y2 > inter_y1 and inter_x2 > inter_x1:
                    inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                    union_area = cand["area"] + existing["area"] - inter_area
                    iou = inter_area / max(1.0, union_area)
                    if iou > 0.15:
                        existing["box"] = [
                            min(y1_a, y1_b), min(x1_a, x1_b),
                            max(y2_a, y2_b), max(x2_a, x2_b)
                        ]
                        existing["area"] = (existing["box"][2] - existing["box"][0]) * (existing["box"][3] - existing["box"][1])
                        matched = True
                        break

            if not matched:
                merged.append(cand)

        return merged

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
        Strictly requires verified water bodies and rejects false detections over land.
        """
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        # Water zone: high blue relative to red, dark surface (cast to int to prevent overflow)
        is_water = (img_np[..., 2].astype(int) > img_np[..., 0].astype(int) + 15) & (gray < 110) & (img_np[..., 1].astype(int) < 135)
        water_ratio = float(np.mean(is_water))

        # If water body is absent or negligible, no marine vessels can be present
        if water_ratio < 0.015:
            return []

        # Look for compact anomalous bright/hull pixels strictly inside water zones
        water_gray = gray[is_water]
        med_water = np.median(water_gray)
        ship_contrast = np.abs(gray - med_water)
        ship_mask = is_water & (ship_contrast > 25.0) & (gray > med_water + 15)

        if np.sum(ship_mask) < 6:
            return []

        step = 2
        mask_sub = ship_mask[::step, ::step]
        sub_h, sub_w = mask_sub.shape
        sub_visited = np.zeros((sub_h, sub_w), dtype=bool)
        candidates = []

        for sy in range(sub_h):
            for sx in range(sub_w):
                if mask_sub[sy, sx] and not sub_visited[sy, sx]:
                    queue = [(sy, sx)]
                    sub_visited[sy, sx] = True
                    pts = []
                    while queue:
                        cy, cx = queue.pop()
                        pts.append((cy * step, cx * step))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < sub_h and 0 <= nx < sub_w:
                                if mask_sub[ny, nx] and not sub_visited[ny, nx]:
                                    sub_visited[ny, nx] = True
                                    queue.append((ny, nx))

                    if len(pts) >= 6:
                        ys = [p[0] for p in pts]
                        xs = [p[1] for p in pts]
                        ymin, ymax = min(ys), max(ys)
                        xmin, xmax = min(xs), max(xs)
                        bw = xmax - xmin
                        bh = ymax - ymin
                        if 8 <= bw <= 140 and 8 <= bh <= 140:
                            candidates.append({
                                "label": "vessel",
                                "box": [float(ymin), float(xmin), float(ymax), float(xmax)],
                                "score": 0.86
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
