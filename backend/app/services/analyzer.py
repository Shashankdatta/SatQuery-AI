import io
import time
from typing import Optional, List
from PIL import Image

from app.schemas.response import AnalysisResponse, DetectedObject, QueryInspectorData


class RemoteSensingAnalyzer:
    """
    Modular vision-language and grounding analyzer service.
    Currently runs in MOCK mode to support UI foundation development.
    Designed for seamless drop-in replacement with Qwen2.5-VL-3B-Instruct
    and remote-sensing detection backends.
    """

    EVIDENCE_THRESHOLD = 0.50
    MOCK_MODEL_ID = "Qwen2.5-VL-3B-Instruct (Mock Foundation)"

    async def analyze(
        self,
        image_bytes: bytes,
        filename: str,
        question: str,
        simulate_abstain: bool = False
    ) -> AnalysisResponse:
        start_time = time.perf_counter()

        # Parse image properties safely
        img_width, img_height = 800, 600
        image_size_kb = len(image_bytes) / 1024.0
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img_width, img_height = img.size
        except Exception:
            pass

        q_lower = question.strip().lower()

        # Determine if query warrants abstention
        # Checks for unresolvable sub-meter queries, occlusions, or explicit simulation
        abstain_keywords = [
            "license plate", "face", "underground", "sub-resolution", 
            "blurry", "cloud cover", "obscured", "unclear", "abstain", 
            "who is inside", "driver"
        ]
        
        should_abstain = simulate_abstain or any(kw in q_lower for kw in abstain_keywords)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if should_abstain:
            return AnalysisResponse(
                status="abstained",
                answer=None,
                evidence_score=0.24,
                objects=[],
                reason="Insufficient visual evidence to answer reliably.",
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent="Unresolvable / Low-Confidence Feature",
                    image_resolution=[img_width, img_height],
                    image_size_kb=round(image_size_kb, 2),
                    latency_ms=elapsed_ms,
                    model_id=self.MOCK_MODEL_ID,
                    evidence_threshold=self.EVIDENCE_THRESHOLD,
                    abstention_triggered=True,
                    mock_mode=True,
                    system_log=[
                        f"Image decoded: {img_width}x{img_height} px ({image_size_kb:.1f} KB)",
                        f"Query parsed: '{question}'",
                        "Evidence evaluation: 0.24 < threshold 0.50",
                        "Abstention policy triggered: insufficient visual ground truth"
                    ]
                )
            )

        # Context-aware mock responses based on question type
        detected_objects: List[DetectedObject] = []
        
        # Scale coordinates relative to uploaded image dimensions
        # format: [ymin, xmin, ymax, xmax] in absolute pixels
        scale_x = img_width / 1000.0
        scale_y = img_height / 800.0

        if any(w in q_lower for kw in ["how many", "count", "number of"] for w in [kw]):
            intent = "Object Counting / Spatial Census"
            # 3 sample structures/facilities
            b1 = [int(150 * scale_y), int(180 * scale_x), int(260 * scale_y), int(310 * scale_x)]
            b2 = [int(220 * scale_y), int(420 * scale_x), int(340 * scale_y), int(560 * scale_x)]
            b3 = [int(400 * scale_y), int(250 * scale_x), int(520 * scale_y), int(390 * scale_x)]
            
            detected_objects = [
                DetectedObject(label="industrial structure #1", box=b1, confidence=0.94),
                DetectedObject(label="industrial structure #2", box=b2, confidence=0.89),
                DetectedObject(label="storage facility #3", box=b3, confidence=0.91),
            ]
            answer = f"Found 3 primary target structures in this sector. Visual grounding confirmed for each facility."
            evidence_score = 0.92

        elif any(w in q_lower for kw in ["runway", "airport", "airstrip", "aircraft", "plane"] for w in [kw]):
            intent = "Aerodrome / Transportation Infrastructure"
            r1 = [int(180 * scale_y), int(80 * scale_x), int(240 * scale_y), int(880 * scale_x)]
            a1 = [int(310 * scale_y), int(450 * scale_x), int(390 * scale_y), int(530 * scale_x)]
            detected_objects = [
                DetectedObject(label="runway tarmac", box=r1, confidence=0.96),
                DetectedObject(label="commercial transport aircraft", box=a1, confidence=0.88),
            ]
            answer = "Active runway tarmac identified with 1 commercial-class transport aircraft located near the central apron."
            evidence_score = 0.94

        elif any(w in q_lower for kw in ["water", "river", "lake", "ocean", "reservoir"] for w in [kw]):
            intent = "Hydrological Feature Detection"
            w1 = [int(350 * scale_y), int(100 * scale_x), int(680 * scale_y), int(750 * scale_x)]
            detected_objects = [
                DetectedObject(label="water body / reservoir boundary", box=w1, confidence=0.91)
            ]
            answer = "Distinct hydrological reservoir boundary identified in the south-central quad of the scene."
            evidence_score = 0.89

        elif any(w in q_lower for kw in ["ship", "vessel", "boat", "port", "dock", "harbor"] for w in [kw]):
            intent = "Maritime / Port Assessment"
            s1 = [int(210 * scale_y), int(310 * scale_x), int(290 * scale_y), int(450 * scale_x)]
            s2 = [int(360 * scale_y), int(520 * scale_x), int(430 * scale_y), int(640 * scale_x)]
            detected_objects = [
                DetectedObject(label="cargo vessel", box=s1, confidence=0.93),
                DetectedObject(label="moored service vessel", box=s2, confidence=0.87),
            ]
            answer = "Identified 2 marine vessels docked along the coastline channel."
            evidence_score = 0.90

        else:
            intent = "General Remote-Sensing Grounded Analysis"
            # Default mock response matching requested format:
            # label: "building", box: [120, 80, 190, 150]
            b0 = [
                int(120 * scale_y),
                int(80 * scale_x),
                int(190 * scale_y),
                int(150 * scale_x)
            ]
            detected_objects = [
                DetectedObject(label="building", box=b0, confidence=0.90)
            ]
            answer = "Identified prominent building complex situated within the designated quadrant of the satellite imagery."
            evidence_score = 0.90

        return AnalysisResponse(
            status="answered",
            answer=answer,
            evidence_score=evidence_score,
            objects=detected_objects,
            reason=None,
            inspector=QueryInspectorData(
                raw_query=question,
                normalized_intent=intent,
                image_resolution=[img_width, img_height],
                image_size_kb=round(image_size_kb, 2),
                latency_ms=elapsed_ms,
                model_id=self.MOCK_MODEL_ID,
                evidence_threshold=self.EVIDENCE_THRESHOLD,
                abstention_triggered=False,
                mock_mode=True,
                system_log=[
                    f"Image decoded: {img_width}x{img_height} px ({image_size_kb:.1f} KB)",
                    f"Query parsed: '{question}'",
                    f"Classified intent: {intent}",
                    f"Grounding boxes generated: {len(detected_objects)}",
                    f"Confidence: {evidence_score:.2f} >= threshold {self.EVIDENCE_THRESHOLD}"
                ]
            )
        )


# Global singleton instance
analyzer_service = RemoteSensingAnalyzer()
