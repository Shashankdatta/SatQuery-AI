from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class DetectedObject(BaseModel):
    label: str
    box: Optional[List[float]] = Field(
        None,
        description="Bounding box [ymin, xmin, ymax, xmax] or [x1, y1, x2, y2]"
    )
    score: Optional[float] = Field(None, description="Detection confidence score")
    confidence: Optional[float] = Field(None, description="Backward-compatible score")

class QueryInspectorData(BaseModel):
    raw_query: Optional[str] = None
    normalized_intent: Optional[str] = None
    image_resolution: Optional[List[int]] = None
    image_size_kb: Optional[float] = None
    latency_ms: Optional[float] = None
    model_id: Optional[str] = None
    evidence_threshold: float = 0.50
    abstention_triggered: bool = False
    mock_mode: bool = False
    system_log: List[str] = []

class AnalysisResponse(BaseModel):
    question: Optional[str] = Field(None, description="Natural-language question")
    answer: Optional[str] = Field(None, description="Model generated answer")
    targets: List[str] = Field(default_factory=list, description="Target object categories")
    evidence_found: bool = Field(True, description="Whether visual evidence was verified")
    detections: List[DetectedObject] = Field(default_factory=list, description="Grounded detection regions")
    abstained: bool = Field(False, description="Whether system abstained due to insufficient evidence")
    evidence_image: Optional[str] = Field(None, description="Base64 encoded visual evidence PNG")

    # Backward-compatible fields
    status: Optional[str] = Field(None, description="'answered', 'loading', 'abstained', or 'error'")
    evidence_score: Optional[float] = Field(None, description="Evidence score")
    objects: List[DetectedObject] = Field(default_factory=list, description="Alias for detections")
    reason: Optional[str] = Field(None, description="Reason if abstained")
    inspector: Optional[QueryInspectorData] = Field(None, description="Technical metadata")
