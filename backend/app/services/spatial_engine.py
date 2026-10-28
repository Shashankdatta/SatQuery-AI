import math
from typing import List, Dict, Any, Optional, Tuple
from app.schemas.response import DetectedObject


class SpatialEngine:
    """
    Computes rigorous coordinate-based spatial relationships from grounded bounding boxes.
    Satellite convention: Top=North, Bottom=South, Left=West, Right=East.
    Bounding box format: [ymin, xmin, ymax, xmax]
    """

    @staticmethod
    def get_box_center(box: List[float]) -> Tuple[float, float]:
        ymin, xmin, ymax, xmax = [float(c) for c in box[:4]]
        return ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)

    @staticmethod
    def get_box_dimensions(box: List[float]) -> Tuple[float, float, float]:
        ymin, xmin, ymax, xmax = [float(c) for c in box[:4]]
        h = max(1.0, ymax - ymin)
        w = max(1.0, xmax - xmin)
        area = h * w
        return (w, h, area)

    @staticmethod
    def get_sector_name(cy: float, cx: float, img_w: int, img_h: int) -> str:
        """Determines compass cardinal sector from normalized image coordinates."""
        ny = cy / max(1.0, float(img_h))
        nx = cx / max(1.0, float(img_w))

        ns = "central"
        if ny < 0.38:
            ns = "northern"
        elif ny > 0.62:
            ns = "southern"

        ew = "central"
        if nx < 0.38:
            ew = "western"
        elif nx > 0.62:
            ew = "eastern"

        if ns == "central" and ew == "central":
            return "central region"
        if ns == "central":
            return f"{ew} sector"
        if ew == "central":
            return f"{ns} sector"
        return f"{ns}-{ew} sector"

    @classmethod
    def analyze_spatial_distribution(
        cls,
        detections: List[DetectedObject],
        img_w: int,
        img_h: int
    ) -> Dict[str, Any]:
        """
        Calculates cluster density, quadrant distributions, and spatial spread.
        """
        if not detections:
            return {
                "count": 0,
                "summary": "No visual grounding targets identified for spatial analysis.",
                "quadrants": {},
                "primary_sector": "scene",
                "distribution_desc": "not localized",
                "densest_quadrant": "unobserved",
                "densest_count": 0,
                "is_clustered": False
            }

        quadrant_counts = {
            "northern": 0,
            "southern": 0,
            "eastern": 0,
            "western": 0,
            "central": 0
        }

        sectors: List[str] = []
        centers: List[Tuple[float, float]] = []

        for d in detections:
            if not d.box or len(d.box) < 4:
                continue
            cy, cx = cls.get_box_center(d.box)
            centers.append((cy, cx))
            sector = cls.get_sector_name(cy, cx, img_w, img_h)
            sectors.append(sector)

            # Assign to quadrant counts
            ny = cy / max(1.0, float(img_h))
            nx = cx / max(1.0, float(img_w))
            if ny < 0.5:
                quadrant_counts["northern"] += 1
            else:
                quadrant_counts["southern"] += 1

            if nx < 0.5:
                quadrant_counts["western"] += 1
            else:
                quadrant_counts["eastern"] += 1

            if 0.25 <= ny <= 0.75 and 0.25 <= nx <= 0.75:
                quadrant_counts["central"] += 1

        # Determine densest quadrant
        densest_cardinal = max(
            [("northern", quadrant_counts["northern"]),
             ("southern", quadrant_counts["southern"]),
             ("eastern", quadrant_counts["eastern"]),
             ("western", quadrant_counts["western"])],
            key=lambda x: x[1]
        )

        # Calculate clustering via average nearest-neighbor distance
        is_clustered = False
        avg_nn_dist = 0.0
        diag = math.sqrt(img_w**2 + img_h**2)

        if len(centers) >= 2:
            nn_dists = []
            for i, c1 in enumerate(centers):
                dists = [
                    math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    for j, c2 in enumerate(centers) if i != j
                ]
                if dists:
                    nn_dists.append(min(dists))
            if nn_dists:
                avg_nn_dist = sum(nn_dists) / len(nn_dists)
                # If average neighbor distance is < 15% of image diagonal, targets are clustered
                is_clustered = (avg_nn_dist / diag) < 0.15

        # Format natural-language spatial description
        sector_counts: Dict[str, int] = {}
        for s in sectors:
            sector_counts[s] = sector_counts.get(s, 0) + 1

        top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        primary_sector = top_sectors[0][0] if top_sectors else "scene"

        if is_clustered:
            distribution_desc = f"clustered primarily in the {primary_sector}"
        else:
            distribution_desc = f"distributed across the {primary_sector}"

        return {
            "count": len(detections),
            "sectors": sectors,
            "primary_sector": primary_sector,
            "distribution_desc": distribution_desc,
            "quadrants": quadrant_counts,
            "densest_quadrant": densest_cardinal[0],
            "densest_count": densest_cardinal[1],
            "is_clustered": is_clustered,
            "avg_nn_dist_px": round(avg_nn_dist, 1)
        }

    @classmethod
    def evaluate_relative_proximity(
        cls,
        box_a: List[float],
        box_b: List[float],
        img_w: int,
        img_h: int
    ) -> Dict[str, Any]:
        """Calculates directional offset and proximity between two grounded boxes."""
        cy_a, cx_a = cls.get_box_center(box_a)
        cy_b, cx_b = cls.get_box_center(box_b)

        dy = cy_b - cy_a
        dx = cx_b - cx_a
        dist = math.sqrt(dx**2 + dy**2)
        diag = math.sqrt(img_w**2 + img_h**2)
        norm_dist = dist / diag

        direction = []
        if dy < -0.05 * img_h:
            direction.append("north")
        elif dy > 0.05 * img_h:
            direction.append("south")

        if dx > 0.05 * img_w:
            direction.append("east")
        elif dx < -0.05 * img_w:
            direction.append("west")

        rel_dir = "-".join(direction) if direction else "adjacent"
        is_near = norm_dist < 0.20

        return {
            "distance_px": round(dist, 1),
            "normalized_dist": round(norm_dist, 3),
            "is_near": is_near,
            "relative_direction": rel_dir
        }


spatial_engine = SpatialEngine()
