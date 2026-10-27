from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class DetectedObject(BaseModel):
    label: str
    box: List[int] = Field(
        ...,
        description="Bounding box [ymin, xmin, ymax, xmax] or [x1, y1, x2, y2]"
    )
    confidence: Optional[float] = Field(None, description="Detection confidence score")

class QueryInspectorData(BaseModel):
    raw_query: str
    normalized_intent: str
    image_resolution: Optional[List[int]] = None
    image_size_kb: Optional[float] = None
    latency_ms: float
    model_id: str
    evidence_threshold: float = 0.50
    abstention_triggered: bool = False
    mock_mode: bool = False
    system_log: List[str] = []

class AnalysisResponse(BaseModel):
    status: str = Field(..., description="'answered', 'loading', 'abstained', or 'error'")
    answer: Optional[str] = Field(None, description="Natural language grounded answer from Qwen2.5-VL")
    evidence_score: Optional[float] = Field(None, description="Evidence score (null for pretrained model)")
    objects: List[DetectedObject] = Field(default_factory=list, description="Grounding bounding boxes")
    reason: Optional[str] = Field(None, description="Reason if abstained or error")
    inspector: Optional[QueryInspectorData] = Field(None, description="Technical inspection metadata for Query Inspector")
