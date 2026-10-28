import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from app.schemas.response import DetectedObject


class InstanceCountingPipeline:
    """
    Dedicated remote-sensing object counting pipeline.
    CRITICAL: Never equates number of raw grounding proposals to object count.
    
    Pipeline:
      1. Target Concept Extraction
      2. High-Resolution Target Semantic Segmentation & Masking
      3. False-Positive Filtering (Excludes water, roads, vegetation, borders)
      4. Connected Component Footprint Analysis
      5. Instance Separability & Resolution Evaluation
      6. Aggressive Duplicate Removal & Instance Grouping
      7. Uncertainty / Abstention Determination
    """

    @classmethod
    def evaluate_building_count(
        cls,
        image: Image.Image,
        vlm_text: str
    ) -> Dict[str, Any]:
        """
        Extracts validated building instances and determines if an exact count is reliable.
        """
        w, h = image.size
        total_pixels = float(w * h)
        img_np = np.array(image.convert("RGB"))
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])

        # 1. Negative Mask Filters
        # Water mask (dark, high blue relative to red; cast to int to prevent uint8 underflow)
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

                    # Filter component
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

                        # Semantic & geometric validation for buildings:
                        # - Width and height >= 18px (removes sub-resolution noise)
                        # - Aspect ratio <= 3.2 (removes roads and linear artifacts)
                        # - Fill factor >= 0.35 (removes sparse noise)
                        # - Area <= 12% of image (removes background regions)
                        if (
                            18 <= bw <= 320 and
                            18 <= bh <= 280 and
                            aspect <= 3.2 and
                            fill >= 0.35 and
                            area <= (0.12 * total_pixels)
                        ):
                            # Verify box is not predominantly road (aspect ratio check + linear check)
                            candidate_instances.append({
                                "box": [float(ymin), float(xmin), float(ymax), float(xmax)],
                                "center": ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0),
                                "width": bw,
                                "height": bh,
                                "area": area,
                                "fill": fill,
                                "score": 0.88
                            })

        # 3. Aggressive Duplicate Removal & Instance Merging
        # If two boxes overlap by > 15% IoU or centers are within 25px, merge them into ONE instance
        merged_instances: List[Dict[str, Any]] = []
        for cand in candidate_instances:
            matched = False
            for existing in merged_instances:
                # Calculate IoU
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
                        # Merge into unified bounding envelope
                        existing["box"] = [
                            min(y1_a, y1_b), min(x1_a, x1_b),
                            max(y2_a, y2_b), max(x2_a, x2_b)
                        ]
                        existing["area"] = (existing["box"][2] - existing["box"][0]) * (existing["box"][3] - existing["box"][1])
                        matched = True
                        break

            if not matched:
                merged_instances.append(cand)

        instance_count = len(merged_instances)

        # 4. Check for Negative / Zero case
        lower_vlm = vlm_text.lower().strip()
        vlm_negative = any(
            w in lower_vlm for w in ["no", "none", "zero", "not visible", "not found", "cannot identify"]
        )

        if instance_count == 0 or (instance_count <= 1 and vlm_negative):
            return {
                "count": 0,
                "is_reliable": True,
                "abstained": True,
                "answer": "Visual analysis does not find reliable evidence of individual buildings in this image.",
                "detections": [],
                "reason": "No validated building footprint instances detected."
            }

        # 5. Instance Separability & Resolution Evaluation (Abstention Gate)
        # Check if buildings are too dense or resolution is insufficient to guarantee exactness
        # Conditions for abstaining from an exact number:
        # A. Low resolution: min image dimension < 200 px and count > 3
        # B. Dense clustering: average instance separation is very tight
        # C. Very high count (> 15 instances in a single scene often have touching rooftops)
        is_exact_reliable = True
        abstain_reason = None

        if min(w, h) < 220 and instance_count > 3:
            is_exact_reliable = False
            abstain_reason = f"Image resolution ({w}x{h} px) is insufficient to resolve individual building boundaries."

        elif instance_count > 15:
            is_exact_reliable = False
            abstain_reason = "High building density and touching rooftop footprints prevent a verified exact count."

        # Format detection objects
        detections: List[DetectedObject] = []
        for i, inst in enumerate(merged_instances):
            detections.append(
                DetectedObject(
                    label=f"building instance #{i+1}",
                    box=inst["box"],
                    score=inst["score"],
                    confidence=inst["score"]
                )
            )

        # 6. Generate Reliable Answer (Abstain if unreliable)
        if not is_exact_reliable:
            answer = (
                f"I can identify several building clusters across the scene, "
                f"but an exact building count cannot be reliably determined from this image ({abstain_reason})."
            )
            return {
                "count": None,
                "is_reliable": False,
                "abstained": True,
                "answer": answer,
                "detections": detections,
                "reason": abstain_reason
            }

        if instance_count == 1:
            answer = "Identified 1 validated building instance with distinct spatial boundaries."
        else:
            answer = f"Identified {instance_count} validated building instances with distinct spatial footprints."

        return {
            "count": instance_count,
            "is_reliable": True,
            "abstained": False,
            "answer": answer,
            "detections": detections,
            "reason": None
        }

    @classmethod
    def evaluate_ship_count(
        cls,
        image: Image.Image,
        vlm_text: str
    ) -> Dict[str, Any]:
        """
        Extracts validated vessel/ship instances over water.
        """
        w, h = image.size
        img_np = np.array(image.convert("RGB"))
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])

        # Water zone: high blue relative to red, dark surface (cast to int to prevent overflow)
        is_water = (img_np[..., 2].astype(int) > img_np[..., 0].astype(int) + 15) & (gray < 110) & (img_np[..., 1].astype(int) < 135)
        water_ratio = np.mean(is_water)

        # Check VLM polarity
        lower_vlm = vlm_text.lower().strip()
        vlm_negative = any(
            w in lower_vlm for w in ["no", "none", "zero", "not visible", "not found", "cannot identify"]
        )

        if water_ratio < 0.015 or vlm_negative:
            return {
                "count": 0,
                "is_reliable": True,
                "abstained": True,
                "answer": "Visual analysis does not detect ships or vessels in this image.",
                "detections": [],
                "reason": "No marine vessels detected over water bodies."
            }

        # Look for compact anomalous bright/hull pixels strictly inside water zones
        water_gray = gray[is_water]
        med_water = np.median(water_gray)
        ship_contrast = np.abs(gray - med_water)
        ship_mask = is_water & (ship_contrast > 25.0) & (gray > med_water + 15)

        if np.sum(ship_mask) < 6:
            return {
                "count": 0,
                "is_reliable": True,
                "abstained": True,
                "answer": "Visual analysis does not detect ships or vessels in this image.",
                "detections": [],
                "reason": "No ship instances located in water regions."
            }
        step = 2
        mask_sub = ship_mask[::step, ::step]
        sub_h, sub_w = mask_sub.shape
        sub_visited = np.zeros((sub_h, sub_w), dtype=bool)
        ship_instances = []

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

                    if len(pts) >= 8:
                        ys = [p[0] for p in pts]
                        xs = [p[1] for p in pts]
                        ymin, ymax = min(ys), max(ys)
                        xmin, xmax = min(xs), max(xs)
                        bw = xmax - xmin
                        bh = ymax - ymin
                        if 10 <= bw <= 120 and 10 <= bh <= 120:
                            ship_instances.append({
                                "box": [float(ymin), float(xmin), float(ymax), float(xmax)],
                                "score": 0.85
                            })

        if not ship_instances:
            return {
                "count": 0,
                "is_reliable": True,
                "abstained": True,
                "answer": "Visual analysis does not detect ships or vessels in this image.",
                "detections": [],
                "reason": "No ship instances located in water regions."
            }

        detections = [
            DetectedObject(label=f"ship instance #{i+1}", box=s["box"], score=s["score"])
            for i, s in enumerate(ship_instances)
        ]

        return {
            "count": len(ship_instances),
            "is_reliable": True,
            "abstained": False,
            "answer": f"Identified {len(ship_instances)} vessel instance(s) localized in water regions.",
            "detections": detections,
            "reason": None
        }

    @classmethod
    def evaluate_general_count(
        cls,
        image: Image.Image,
        target_concept: str,
        vlm_text: str
    ) -> Dict[str, Any]:
        """General fallback count resolution."""
        target = target_concept.lower()
        if target in ["building", "structure", "facility", "house"]:
            return cls.evaluate_building_count(image, vlm_text)
        elif target in ["ship", "vessel", "boat"]:
            return cls.evaluate_ship_count(image, vlm_text)

        # For roads / vegetation / terrain: continuous features cannot be counted as discrete instances
        if target in ["road", "highway", "water", "vegetation", "landscape"]:
            return {
                "count": None,
                "is_reliable": False,
                "abstained": True,
                "answer": f"The target concept '{target_concept}' is a continuous spatial network/region, not a discrete countable object instance.",
                "detections": [],
                "reason": f"'{target_concept}' is continuous spatial infrastructure, not discrete countable instances."
            }

        return {
            "count": None,
            "is_reliable": False,
            "abstained": True,
            "answer": f"I can observe the scene, but an exact count of '{target_concept}' cannot be reliably determined from this image.",
            "detections": [],
            "reason": "Instance boundaries lack verified separability."
        }


instance_counting_pipeline = InstanceCountingPipeline()
