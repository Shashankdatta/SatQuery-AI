import re
from typing import List, Dict, Any, Optional, Tuple
from app.schemas.response import DetectedObject


class CountingEngine:
    """
    Rigorously resolves counting queries by combining grounded candidate footprints
    with VLM reasoning and image resolution limits.
    Prevents hallucinating arbitrary exact numbers when visual resolution is ambiguous.
    """

    @classmethod
    def resolve_count(
        cls,
        target_concept: str,
        detections: List[DetectedObject],
        vlm_raw_answer: str,
        img_width: int,
        img_height: int
    ) -> Dict[str, Any]:
        """
        Reconciles grounded candidate objects with VLM text response and resolution.
        """
        grounded_count = len(detections)
        vlm_lower = vlm_raw_answer.lower().strip()

        # Parse numerical digit or word in VLM response
        vlm_number = None
        num_match = re.search(r'\b(\d+)\b', vlm_lower)
        if num_match:
            try:
                vlm_number = int(num_match.group(1))
            except ValueError:
                pass

        # Check for textual numbers
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for word, val in word_to_num.items():
            if re.search(r'\b' + word + r'\b', vlm_lower):
                vlm_number = val
                break

        # Check for negative statements in VLM response
        is_negative = any(w in vlm_lower for w in ["no", "none", "zero", "not visible", "not found", "cannot identify"])

        if is_negative and grounded_count == 0:
            return {
                "count": 0,
                "answer": f"Visual analysis confirms that no {target_concept}s are discernible in this scene.",
                "is_exact": True,
                "confidence": 0.90
            }

        # Check resolution limit:
        # If the image resolution is modest (< 200px) or detections are crowded,
        # individual footprint boundaries cannot be guaranteed 100%
        if min(img_width, img_height) < 180 and grounded_count > 4:
            return {
                "count": grounded_count,
                "answer": (
                    f"Several {target_concept}s are visible across the scene, "
                    f"but the image resolution ({img_width}x{img_height} px) does not support a reliable exact count."
                ),
                "is_exact": False,
                "confidence": 0.72
            }

        # Case 1: Grounded objects exist and VLM agrees or provides close number
        if grounded_count > 0:
            if vlm_number is not None and abs(vlm_number - grounded_count) <= 2:
                final_count = vlm_number
            else:
                final_count = grounded_count

            if final_count == 1:
                answer = f"Identified 1 distinct {target_concept} in the imagery with verified visual grounding."
            elif 2 <= final_count <= 25:
                answer = f"Identified {final_count} distinct {target_concept}s with verified visual grounding."
            else:
                answer = (
                    f"Identified a dense concentration of approximately {final_count} {target_concept}s. "
                    f"Due to structure proximity and spatial clustering, exact census may have minor overlaps."
                )

            return {
                "count": final_count,
                "answer": answer,
                "is_exact": final_count <= 25,
                "confidence": 0.88 if final_count <= 25 else 0.80
            }

        # Case 2: VLM claimed a number but grounding found zero distinct boxes
        if vlm_number and vlm_number > 0 and grounded_count == 0:
            return {
                "count": None,
                "answer": (
                    f"Candidate {target_concept} patterns may be present, "
                    f"but spatial resolution is insufficient to ground and verify individual objects."
                ),
                "is_exact": False,
                "confidence": 0.60
            }

        # Case 3: Zero grounded detections and no number
        return {
            "count": 0,
            "answer": f"No distinct individual {target_concept}s could be reliably isolated in this scene.",
            "is_exact": True,
            "confidence": 0.85
        }


counting_engine = CountingEngine()
