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
    UNOBSERVABLE_METADATA = "unobservable_metadata"
    GENERAL_VQA = "general_vqa"


class QueryClassification(BaseModel):
    intent: QueryIntent
    target_concepts: List[str]
    spatial_keywords: List[str] = []
    requires_multitemporal: bool = False
    requires_non_rgb: bool = False
    requires_unobservable: bool = False
    unobservable_reason: Optional[str] = None
    evidence_policy: str = "A"  # "A" (Answerable), "B" (Ambiguous), "C" (Not Answerable)
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
    "airport": [
        "airport", "airports", "airfield", "airfields", "aerodrome", "airstrip"
    ],
    "bridge": [
        "bridge", "bridges", "overpass", "viaduct"
    ],
    "water": [
        "water", "river", "lake", "ocean", "sea", "canal", "reservoir", "coast", "coastline"
    ],
    "vegetation": [
        "vegetation", "forest", "tree", "trees", "crop", "crops", "field", "fields",
        "agriculture", "agricultural", "greenery", "canopy", "grass"
    ],
    "parking": [
        "parking", "parking lot", "parking lots"
    ],
    "stadium": [
        "stadium", "sports field", "arena"
    ],
    "railway": [
        "railway", "railroad", "train track", "trains", "tracks"
    ],
    "storage_tank": [
        "storage tank", "storage tanks", "tank", "tanks", "silo", "silos"
    ],
    "solar_panel": [
        "solar panel", "solar panels", "photovoltaic"
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
    "expansion", "construction", "constructed", "build", "built", "new",
    "demolished", "demolition", "destroyed", "before", "after", "temporal",
    "timeline", "over time", "last year", "next year", "past year", "previous year",
    "yesterday", "last month", "last decade", "historical", "history", "years ago",
    "growth over", "rate of change", "prior to"
]

SPECTRAL_KEYWORDS = [
    "nir", "swir", "sar", "thermal", "infrared", "ndvi", "multispectral",
    "hyperspectral", "band", "bands", "radar", "microwave", "backscatter",
    "reflectance", "radiance", "elevation", "dem", "lidar"
]

POPULATION_PATTERNS = [
    r'\bpopulation\b', r'\bhow many people live\b', r'\bnumber of people\b',
    r'\bhow many residents\b', r'\bhow many inhabitants\b', r'\bcensus\b',
    r'\bdemographic\b', r'\bexact population\b'
]

GEOGRAPHIC_NAME_PATTERNS = [
    r'\bwhat is the name of this (?:city|town|place|area|region|village|state|country|island|neighborhood)\b',
    r'\bwhat (?:city|town|country|state|place) is this\b',
    r'\bwhich (?:city|town|country|state|place) is this\b',
    r'\bname of this (?:city|town|place|area|region|airport|harbor|port)\b',
    r'\bwhat is this place called\b',
    r'\bwhere on (?:earth|the map) is this\b',
    r'\bwhat is the address\b',
    r'\bexact coordinates\b',
    r'\bwhat street is this\b',
    r'\bstreet name\b'
]

OWNERSHIP_PATTERNS = [
    r'\bbelongs? to\b', r'\bwho owns\b', r'\bwhich company owns\b',
    r'\bcompany [a-z0-9]+\b', r'\bowner of\b', r'\bowned by\b',
    r'\bprivate property\b', r'\bproperty owner\b', r'\bproperty boundaries?\b',
    r'\bcorporate owner\b', r'\bwho built this\b'
]

NON_VISUAL_PATTERNS = [
    r'\binside the\b', r'\binterior of\b', r'\bwhat is inside\b',
    r'\bdepth of the water\b', r'\bhow deep is the\b', r'\bwater depth\b',
    r'\bspeed of\b', r'\bhow fast\b', r'\bcost of\b', r'\bhow much does it cost\b',
    r'\bprice of\b', r'\bhow much is it worth\b'
]


def classify_query(question: str) -> QueryClassification:
    """
    Parses user query into remote-sensing intent, target concepts, and constraints.
    Enforces Evidence-First Answer Policy categories:
      A. ANSWERABLE FROM IMAGE
      B. AMBIGUOUS / INSUFFICIENT VISUAL EVIDENCE
      C. NOT ANSWERABLE FROM THIS IMAGE
    """
    q = question.strip().lower()

    # 1. Gatekeeper: Multispectral / Non-RGB query on standard RGB (Category C)
    requires_non_rgb = any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in SPECTRAL_KEYWORDS)
    if requires_non_rgb:
        return QueryClassification(
            intent=QueryIntent.SPECTRAL_ANALYSIS,
            target_concepts=["multispectral_band"],
            requires_non_rgb=True,
            evidence_policy="C",
            normalized_intent="Multispectral / Non-RGB Spectral Analysis"
        )

    # 2. Gatekeeper: Change Detection / Temporal query on single image (Category C)
    has_temporal_kw = any(re.search(r'\b' + re.escape(kw) + r'\b', q) for kw in CHANGE_KEYWORDS)
    has_temporal_regex = bool(
        re.search(r'\b(in|since|during)\s+(?:19\d\d|20\d\d)\b', q) or
        re.search(r'\b\d+\s+years?\s+ago\b', q) or
        re.search(r'\bconstructed\s+(?:last|past|in|during)\b', q) or
        re.search(r'\bhow many (?:.*) were (?:built|constructed|added|demolished|created)\b', q)
    )
    if has_temporal_kw or has_temporal_regex:
        return QueryClassification(
            intent=QueryIntent.CHANGE_DETECTION,
            target_concepts=["temporal_change"],
            requires_multitemporal=True,
            evidence_policy="C",
            normalized_intent="Multi-Temporal Change Detection"
        )

    # 3. Gatekeeper: Unobservable Metadata Inquiries (Category C)
    # 3A. Population / Demographic statistics
    if any(re.search(p, q) for p in POPULATION_PATTERNS):
        return QueryClassification(
            intent=QueryIntent.UNOBSERVABLE_METADATA,
            target_concepts=["population_statistics"],
            requires_unobservable=True,
            unobservable_reason="I can't determine population or demographic statistics reliably from a visual satellite image.",
            evidence_policy="C",
            normalized_intent="Unobservable Demographic Inquiry"
        )

    # 3B. Geographic proper names / addresses
    if any(re.search(p, q) for p in GEOGRAPHIC_NAME_PATTERNS):
        return QueryClassification(
            intent=QueryIntent.UNOBSERVABLE_METADATA,
            target_concepts=["geographic_location"],
            requires_unobservable=True,
            unobservable_reason="I can't determine geographic names or exact locations reliably from the provided visual image alone.",
            evidence_policy="C",
            normalized_intent="Unobservable Geographic Name Inquiry"
        )

    # 3C. Ownership / Corporate Affiliation
    if any(re.search(p, q) for p in OWNERSHIP_PATTERNS):
        return QueryClassification(
            intent=QueryIntent.UNOBSERVABLE_METADATA,
            target_concepts=["property_ownership"],
            requires_unobservable=True,
            unobservable_reason="Ownership, organizational affiliation, and property records cannot be determined from visual appearance in a satellite image.",
            evidence_policy="C",
            normalized_intent="Unobservable Ownership Inquiry"
        )

    # 3D. Internal / Non-Visual properties
    if any(re.search(p, q) for p in NON_VISUAL_PATTERNS):
        return QueryClassification(
            intent=QueryIntent.UNOBSERVABLE_METADATA,
            target_concepts=["non_visual_attribute"],
            requires_unobservable=True,
            unobservable_reason="Internal structural details and non-visual physical properties cannot be determined reliably from this satellite image.",
            evidence_policy="C",
            normalized_intent="Non-Visual Property Inquiry"
        )

    # Extract target concepts
    targets: Set[str] = set()
    for cat, synonyms in TARGET_LEXICON.items():
        for syn in synonyms:
            if re.search(r'\b' + re.escape(syn) + r'\b', q):
                targets.add(cat)
                break

    # If no target found in lexicon, attempt to extract explicit noun from question
    if not targets:
        noun_match = re.search(r'(?:how many|number of|are there|is there|count(?: of)?)\s+([a-z0-9_-]+)', q)
        if not noun_match:
            noun_match = re.search(r'(?:north|south|east|west|near|around)\s+(?:of\s+)?(?:the\s+)?([a-z0-9_-]+)', q)
        if noun_match:
            extracted = noun_match.group(1).strip()
            if extracted not in ["any", "the", "a", "an", "this", "that", "these", "those"]:
                targets.add(extracted)

    target_list = list(targets) if targets else ["feature"]

    # Extract spatial keywords
    found_spatial = [kw for kw in SPATIAL_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', q)]

    # 4. Intent Determination (Answerable / Grounding candidates)
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
        requires_multitemporal=False,
        requires_non_rgb=False,
        requires_unobservable=False,
        evidence_policy="A",
        normalized_intent=normalized
    )
