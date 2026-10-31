import io
import re
import base64
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from app.schemas.response import DetectedObject


class EvidenceVerifier:
    """
    Cross-verifies textual VLM assertions with actual visual bounding box evidence.
    Calibrates confidence and eliminates hallucinations when visual evidence is missing.
    Follows zero-fabrication confidence rules.
    """

    @staticmethod
    def verify_and_calibrate(
        vlm_text: str,
        target_concept: str,
        detections: List[DetectedObject],
        intent: str
    ) -> Tuple[bool, bool, Optional[float], str]:
        """
        Returns:
          evidence_found (bool)
          abstained (bool)
          calibrated_confidence (Optional[float])
          final_reason (str)
        """
        lower_vlm = vlm_text.lower().strip()

        # Check for explicit negative or abstention signals in VLM answer
        negative_phrases = [
            "no", "none", "nothing", "not visible", "not found",
            "cannot tell", "unclear", "insufficient", "not present",
            "cannot determine", "cannot see", "does not appear"
        ]
        is_negative = bool(lower_vlm) and any(
            re.search(r'\b' + re.escape(p) + r'\b', lower_vlm)
            for p in negative_phrases
        )

        # 1. Negative result: visual evidence is absent or negative
        if is_negative or len(detections) == 0:
            if is_negative:
                return (False, True, None, f"Visual evidence is negative: target '{target_concept}' is not discernible.")
            else:
                return (False, True, None, f"Visual features for '{target_concept}' lack sufficient boundary resolution.")

        # 2. Positive result: visual grounding boxes exist
        # Calculate actual mean detection score if calibrated scores are present
        scores = [d.score for d in detections if d.score is not None]
        avg_score = round(sum(scores) / len(scores), 2) if scores else None

        return (True, False, avg_score, "Visual evidence verified with spatial grounding.")

    @staticmethod
    def render_annotated_evidence(
        image: Image.Image,
        detections: List[DetectedObject],
        target_concept: str
    ) -> Optional[str]:
        """
        Renders crisp visual grounding annotations directly on the native image.
        Returns base64 PNG string.
        """
        if not detections:
            return None

        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size

        # High-visibility remote-sensing green
        box_color = (0, 230, 138)

        for i, det in enumerate(detections):
            if not det.box or len(det.box) < 4:
                continue
            ymin, xmin, ymax, xmax = [float(c) for c in det.box]

            # Clamp coordinates to image boundaries
            ymin = max(0.0, min(float(height - 1), ymin))
            ymax = max(0.0, min(float(height - 1), ymax))
            xmin = max(0.0, min(float(width - 1), xmin))
            xmax = max(0.0, min(float(width - 1), xmax))

            if ymax <= ymin or xmax <= xmin:
                continue

            # Draw outer rectangle
            line_w = max(2, min(width, height) // 300)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=line_w)

        buffered = io.BytesIO()
        annotated.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


evidence_verifier = EvidenceVerifier()
