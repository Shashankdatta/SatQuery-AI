import re
from enum import Enum
from typing import List, Optional, Set
from pydantic import BaseModel


class QueryIntent(str, Enum):
    PRESENCE_ABSENCE = "presence_absence"
    COUNTING = "counting"
    SPATIAL_RELATIONSHIP = "spatial_relationship"
    OBJECT_ATTRIBUTES = "object_attributes"
    CHANGE_DETECTION = "change_detection"
    SCENE_UNDERSTANDING = "scene_understanding"
    COMPARATIVE = "comparative"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    GENERAL_VQA = "general_vqa"


class QueryClassification(BaseModel):
    intent: QueryIntent
    target_concepts: List[str]
    spatial_keywords: List[str] = []
    requires_multitemporal: bool = False
    requires_non_rgb: bool = False
    normalized_intent: str


# Remote-sensing target concept synonyms mapping
TARGET_LEXICON = {
    "building": [
        "building", "buildings", "structure", "structures", "house", "houses", 
        "rooftop", "rooftops", "facility", "facilities", "settlement", "settlements",
        "factory", "shed", "barn", "warehouse", "residential", "compound"
    ],
    "road": [
        "road", "roads", "highway", "highways", "street", "streets", "avenue", 
        "lane", "path", "runway", "runways", "taxiway", "track", "freeway"
    ],
    "ship": [
        "ship", "ships", "vessel", "vessels", "boat", "boats", "barge", 
        "tanker", "craft", "watercraft"
    ],
    "port": [
        "port", "harbor", "harbour", "dock", "docks", "pier", "marina", "quay"
    ],
    "vehicle": [
        "vehicle", "vehicles", "car", "cars", "truck", "trucks", "automobile"
    ],
    "aircraft": [
        "aircraft", "airplane", "airplanes", "plane", "planes", "jet", "helicopter"
    ],
    "water": [
        "water", "river", "lake", "ocean", "sea", "canal", "reservoir", "coast", "coastline"
    ],
    "vegetation": [
        "vegetation", "forest", "tree", "trees", "crop", "crops", "field", "fields",
        "agriculture", "agricultural", "greenery", "canopy", "grass"
    ],
    "landscape": [
        "landscape", "terrain", "urban", "rural", "land-use", "land use", "area", "scene"
    ]
}

SPATIAL_KEYWORDS = [
    "where", "location", "located", "north", "northern", "south", "southern",
    "east", "eastern", "west", "western", "left", "right", "center", "central",
    "above", "below", "near", "nearby", "next to", "close", "far", "distance",
    "adjacent", "clustered", "cluster", "surrounding", "between", "densest", "denser"
]

CHANGE_KEYWORDS = [
    "change", "changed", "changes", "difference", "differences", "expanded",
    "expansion", "construction", "new", "demolished", "before", "after",
    "temporal", "timeline", "over time"
]

SPECTRAL_KEYWORDS = [
    "nir", "swir", "sar", "thermal", "infrared", "ndvi", "multispectral",
    "hyperspectral", "band", "bands", "radar", "microwave"
]


def classify_query(question: str) -> QueryClassification:
    """
    Parses user query into remote-sensing intent, target concepts, and constraints.
    """
    q = question.strip().lower()

    # 1. Check for Multispectral / Non-RGB query
    requires_non_rgb = any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in SPECTRAL_KEYWORDS)
    if requires_non_rgb:
        return QueryClassification(
            intent=QueryIntent.SPECTRAL_ANALYSIS,
            target_concepts=["multispectral_band"],
            requires_non_rgb=True,
            normalized_intent="Multispectral / Non-RGB Spectral Analysis"
        )

    # 2. Check for Change Detection query
    requires_multitemporal = any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in CHANGE_KEYWORDS)
    if requires_multitemporal:
        return QueryClassification(
            intent=QueryIntent.CHANGE_DETECTION,
            target_concepts=["temporal_change"],
            requires_multitemporal=True,
            normalized_intent="Multi-Temporal Change Detection"
        )

    # Extract target concepts
    targets: Set[str] = set()
    for cat, synonyms in TARGET_LEXICON.items():
        for syn in synonyms:
            if re.search(r'\b' + re.escape(syn) + r'\b', q):
                targets.add(cat)
                break

    target_list = list(targets) if targets else ["general_feature"]

    # Extract spatial keywords
    found_spatial = [kw for kw in SPATIAL_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', q)]

    # 3. Intent Determination
    # Counting
    if any(re.search(r'\b' + re.escape(p) + r'\b', q) for p in ["how many", "count", "number of", "how much"]):
        intent = QueryIntent.COUNTING
        normalized = f"Object Counting ({', '.join(target_list)})"

    # Comparative questions
    elif any(re.search(r'\b' + re.escape(p) + r'\b', q) for p in ["which area", "which region", "denser", "more", "less", "compared to"]):
        intent = QueryIntent.COMPARATIVE
        normalized = f"Spatial Comparative Analysis ({', '.join(target_list)})"

    # Spatial relationships
    elif found_spatial or q.startswith("where") or "near" in q or "located" in q:
        intent = QueryIntent.SPATIAL_RELATIONSHIP
        normalized = f"Spatial Relationship & Localization ({', '.join(target_list)})"

    # Scene Understanding
    elif any(re.search(r'\b' + re.escape(p) + r'\b', q) for p in ["urban or rural", "type of landscape", "land-use", "environment", "rural or urban", "what kind of terrain"]):
        intent = QueryIntent.SCENE_UNDERSTANDING
        normalized = "Macro Scene / Land-Use Understanding"

    # Object attributes
    elif any(re.search(r'\b' + re.escape(p) + r'\b', q) for p in ["what shape", "what type of structure", "large or small", "color of", "arrangement"]):
        intent = QueryIntent.OBJECT_ATTRIBUTES
        normalized = f"Target Attribute Analysis ({', '.join(target_list)})"

    # Presence / Absence
    elif any(q.startswith(prefix) for prefix in ["is there", "are there", "do you see", "can you see", "is any", "are any", "does it have"]):
        intent = QueryIntent.PRESENCE_ABSENCE
        normalized = f"Presence / Absence Verification ({', '.join(target_list)})"

    else:
        intent = QueryIntent.GENERAL_VQA
        normalized = f"Remote-Sensing Vision-Language Query ({', '.join(target_list)})"

    return QueryClassification(
        intent=intent,
        target_concepts=target_list,
        spatial_keywords=found_spatial,
        requires_multitemporal=requires_multitemporal,
        requires_non_rgb=requires_non_rgb,
        normalized_intent=normalized
    )
