import io
import os
import time
import base64
import re
import threading
from typing import Optional, Dict, Any, List
from PIL import Image

import torch
from app.schemas.response import AnalysisResponse, DetectedObject, QueryInspectorData
from app.services.query_intent import classify_query, QueryIntent, QueryClassification
from app.services.grounding_engine import grounding_engine
from app.services.spatial_engine import spatial_engine
from app.services.counting_engine import counting_engine
from app.services.counting_pipeline import instance_counting_pipeline
from app.services.evidence_verifier import evidence_verifier


class VisionService:
    """
    Query-Aware Remote-Sensing Vision-Language Assistant.
    Enforces Evidence-First Answer Policy:
      A. ANSWERABLE FROM IMAGE -> Answer based strictly on verified visual evidence.
      B. AMBIGUOUS / INSUFFICIENT EVIDENCE -> Refuse / abstain cleanly.
      C. NOT ANSWERABLE FROM THIS IMAGE -> Gatekeeper refusal (temporal, spectral, unobservable).
    """

    def __init__(self):
        self.status = "unloaded"
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_progress = "Awaiting initialization"
        self.error_message: Optional[str] = None
        self._lock = threading.Lock()
        self._gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

    def start_background_loading(self):
        with self._lock:
            if self.status in ("loading", "ready"):
                return
            self.status = "loading"
            self.load_progress = "Initializing Vision-Language engine..."

        if self._gemini_api_key:
            with self._lock:
                self.status = "ready"
                self.load_progress = "Cloud Gemini 1.5 Flash Vision engine active"
            print("[VisionService] Using Google Gemini 1.5 Flash Vision engine.")
            return

        def _loader():
            try:
                print(f"[VisionService] Loading BLIP VQA model on {self.device}...")
                from transformers import BlipProcessor, BlipForQuestionAnswering
                self.load_progress = "Loading Salesforce/blip-vqa-base..."
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
                self.model.to(self.device)
                self.model.eval()
                with self._lock:
                    self.status = "ready"
                    self.load_progress = f"BLIP VQA ready on {self.device.upper()}"
                print(f"[VisionService] Vision model loaded successfully on {self.device}!")
            except Exception as exc:
                print(f"[VisionService] Error loading local model: {exc}")
                with self._lock:
                    self.status = "ready"
                    self.load_progress = f"Rule-based VLM fallback active ({str(exc)})"

        thread = threading.Thread(target=_loader, daemon=True)
        thread.start()

    def get_status(self) -> Dict[str, Any]:
        engine_name = "Gemini 1.5 Flash" if self._gemini_api_key else "BLIP VQA (Salesforce)"
        return {
            "status": self.status,
            "engine": engine_name,
            "device": self.device,
            "progress": self.load_progress,
            "error": self.error_message,
        }

    async def analyze(
        self,
        image_bytes: bytes,
        filename: str,
        question: str,
        api_key: Optional[str] = None,
    ) -> AnalysisResponse:
        start_time = time.perf_counter()

        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded image file is empty.")
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Decode image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_width, img_height = image.size
            image_size_kb = len(image_bytes) / 1024.0
        except Exception as exc:
            raise ValueError(f"Failed to decode image file '{filename}': {str(exc)}")

        # Check for insufficient image geometry threshold (< 32px cannot support visual features)
        if img_width < 32 or img_height < 32:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResponse(
                question=question,
                answer="I can't determine that reliably from the provided image because the image resolution is too low (< 32px) to detect spatial features.",
                targets=["feature"],
                evidence_found=False,
                detections=[],
                abstained=True,
                status="abstained",
                reason="Image resolution is too low to provide reliable visual evidence.",
            )

        try:
            # 1. Query Intent Classification & Evidence-First Policy
            classification: QueryClassification = classify_query(question)
            target_concept = classification.target_concepts[0] if classification.target_concepts else "feature"
            system_logs = [
                f"Image decoded: {img_width}x{img_height} px ({image_size_kb:.1f} KB)",
                f"Query parsed: intent='{classification.intent.value}', target='{target_concept}'",
                f"Evidence policy category: '{classification.evidence_policy}'",
            ]

            # 2. Gatekeeper: Unobservable Metadata / Non-Visual Inquiries (Category C)
            if classification.requires_unobservable:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                reason_msg = classification.unobservable_reason or "I can't determine that reliably from the provided image."
                system_logs.append(f"Gatekeeper triggered: {classification.normalized_intent}.")
                return AnalysisResponse(
                    question=question,
                    answer=reason_msg,
                    targets=classification.target_concepts,
                    evidence_found=False,
                    detections=[],
                    abstained=True,
                    status="abstained",
                    reason=reason_msg,
                    inspector=QueryInspectorData(
                        raw_query=question,
                        normalized_intent=classification.normalized_intent,
                        image_resolution=[img_width, img_height],
                        image_size_kb=round(image_size_kb, 2),
                        latency_ms=elapsed_ms,
                        model_id="UnobservableMetadataGatekeeper",
                        evidence_threshold=0.50,
                        abstention_triggered=True,
                        system_log=system_logs
                    )
                )

            # 3. Gatekeeper: Change Detection / Temporal on single image (Category C)
            if classification.requires_multitemporal:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                reason_msg = (
                    "The provided input is a single satellite image. Historical construction, growth, "
                    "or temporal changes over time cannot be determined reliably without multi-date imagery."
                )
                system_logs.append("Gatekeeper triggered: Single temporal acquisition provided for change detection query.")
                return AnalysisResponse(
                    question=question,
                    answer=reason_msg,
                    targets=["temporal_change"],
                    evidence_found=False,
                    detections=[],
                    abstained=True,
                    status="abstained",
                    reason="Single temporal image provided. Historical or temporal change queries require multi-date imagery.",
                    inspector=QueryInspectorData(
                        raw_query=question,
                        normalized_intent=classification.normalized_intent,
                        image_resolution=[img_width, img_height],
                        image_size_kb=round(image_size_kb, 2),
                        latency_ms=elapsed_ms,
                        model_id="ChangeDetectionGatekeeper",
                        evidence_threshold=0.50,
                        abstention_triggered=True,
                        system_log=system_logs
                    )
                )

            # 4. Gatekeeper: Non-RGB / Multispectral inquiry on standard RGB (Category C)
            if classification.requires_non_rgb:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                reason_msg = (
                    "The provided input is standard 3-channel RGB imagery. True multispectral or radar "
                    "measurements (such as SAR backscatter, NIR reflectance, or thermal anomalies) "
                    "cannot be determined reliably from this image."
                )
                system_logs.append("Gatekeeper triggered: Non-RGB spectral band requested on standard 3-channel RGB image.")
                return AnalysisResponse(
                    question=question,
                    answer=reason_msg,
                    targets=["multispectral_band"],
                    evidence_found=False,
                    detections=[],
                    abstained=True,
                    status="abstained",
                    reason="Spectral or radar band unavailable in standard 3-channel RGB imagery.",
                    inspector=QueryInspectorData(
                        raw_query=question,
                        normalized_intent=classification.normalized_intent,
                        image_resolution=[img_width, img_height],
                        image_size_kb=round(image_size_kb, 2),
                        latency_ms=elapsed_ms,
                        model_id="SpectralGatekeeper",
                        evidence_threshold=0.50,
                        abstention_triggered=True,
                        system_log=system_logs
                    )
                )

            # 5. Remote-Sensing VLM Semantic Reasoning
            raw_vlm_text = ""
            model_name = "BLIP VQA (Salesforce)"

            # 5A. Check Cloud Vision API (Gemini) if API key is present
            gemini_key = (api_key or "").strip() or self._gemini_api_key
            if gemini_key:
                try:
                    import requests
                    b64_img = base64.b64encode(image_bytes).decode("utf-8")
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                    prompt = (
                        f"You are SatQuery AI, an expert remote-sensing vision-language assistant for satellite imagery. "
                        f"Carefully analyze this satellite/aerial image and answer the user question:\n'{question.strip()}'\n"
                        f"Provide an accurate, concise, grounded answer based strictly on what is visible in the image. "
                        f"If the image does not provide sufficient visual evidence, state that clearly."
                    )
                    payload = {
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
                            ]
                        }]
                    }
                    resp = requests.post(api_url, json=payload, timeout=20)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_vlm_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        model_name = "Gemini 1.5 Flash Vision"
                        system_logs.append("VLM inference completed using Gemini 1.5 Flash Vision.")
                except Exception as e:
                    print(f"[VisionService] Gemini API call error: {e}, falling back to local engine.")

            # 5B. Local Open-Source Vision-Language Inference (BLIP VQA)
            if not raw_vlm_text and self.model is not None and self.processor is not None:
                try:
                    inputs = self.processor(image, question, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=100)
                        raw_vlm_text = self.processor.decode(out[0], skip_special_tokens=True).strip()
                    model_name = "BLIP VQA (Salesforce)"
                    system_logs.append(f"VLM inference completed using Salesforce/blip-vqa-base on {self.device}.")
                except Exception as exc:
                    print(f"[VisionService] Inference error: {exc}")
                    system_logs.append(f"Local inference warning: {exc}")

            # Check if model is still loading
            if not raw_vlm_text and self.status == "loading":
                return AnalysisResponse(
                    question=question,
                    answer=f"Vision model weights are currently downloading and initializing ({self.load_progress}). Please try again in a few moments.",
                    targets=["feature"],
                    evidence_found=False,
                    detections=[],
                    abstained=False,
                    status="loading",
                )

            # Evaluate VLM polarity if explicit VLM text is available
            has_vlm_output = bool(raw_vlm_text and raw_vlm_text.strip())
            if has_vlm_output:
                lower_vlm = raw_vlm_text.lower().strip()
                vlm_negative = any(
                    re.search(r'\b' + re.escape(w) + r'\b', lower_vlm)
                    for w in ["no", "none", "nothing", "not visible", "not found", "cannot tell", "unclear", "cannot identify", "not present"]
                )
            else:
                raw_vlm_text = ""
                vlm_negative = False

            # 6. Query-Aware Visual Grounding
            detections: List[DetectedObject] = []
            if not vlm_negative:
                detections = grounding_engine.ground_target_concept(
                    image=image,
                    target_concept=target_concept,
                    vlm_affirmative=True,
                    max_detections=20
                )
                system_logs.append(f"Grounding engine located {len(detections)} discrete region footprints for '{target_concept}'.")
            else:
                system_logs.append("VLM negative response: visual grounding bypassed to prevent false-positive hallucination.")

            # 7. Task-Specific Engine Resolution
            final_answer = ""
            is_abstained = False
            reason_str = None

            # 7A. COUNTING RESOLUTION (Instance-Level Dedicated Counting Pipeline)
            if classification.intent == QueryIntent.COUNTING:
                count_res = instance_counting_pipeline.evaluate_general_count(
                    image=image,
                    target_concept=target_concept,
                    vlm_text=raw_vlm_text
                )
                final_answer = count_res["answer"]
                is_abstained = count_res["abstained"]
                reason_str = count_res.get("reason")
                if not is_abstained and count_res.get("detections"):
                    detections = count_res["detections"]
                else:
                    detections = []  # Purge proposals when abstaining

                system_logs.append(
                    f"Instance counting pipeline: count={count_res.get('count')}, reliable={count_res.get('is_reliable')}, abstained={is_abstained}."
                )

            # 7B. SPATIAL & COMPARATIVE RESOLUTION
            elif classification.intent in (QueryIntent.SPATIAL_RELATIONSHIP, QueryIntent.COMPARATIVE):
                if not detections:
                    display_target = target_concept if target_concept != "feature" else "features"
                    final_answer = f"Visual analysis does not detect reliable evidence of {display_target} in this image to determine spatial relationships."
                    is_abstained = True
                    reason_str = f"No validated footprints detected for {display_target}."
                    detections = []
                else:
                    spatial_res = spatial_engine.analyze_spatial_distribution(detections, img_width, img_height)

                    if any(w in question.lower() for w in ["densest", "denser", "highest concentration", "which area", "which region"]):
                        final_answer = (
                            f"Based on spatial coordinate analysis of the {len(detections)} grounded {target_concept} footprints: "
                            f"The {spatial_res['densest_quadrant']} sector contains the highest concentration with {spatial_res['densest_count']} identified features. "
                            f"Features are {spatial_res['distribution_desc']}."
                        )
                    elif classification.intent == QueryIntent.SPATIAL_RELATIONSHIP and "where" in question.lower():
                        final_answer = (
                            f"The {len(detections)} grounded {target_concept} features are located primarily in the "
                            f"{spatial_res['primary_sector']}, with structures {spatial_res['distribution_desc']}."
                        )
                    elif "near" in question.lower():
                        final_answer = (
                            f"Spatial analysis of verified {target_concept} features: features are localized in the "
                            f"{spatial_res['primary_sector']} ({spatial_res['distribution_desc']})."
                        )
                    else:
                        if raw_vlm_text and raw_vlm_text.strip():
                            final_answer = f"Based on spatial analysis of the imagery: {raw_vlm_text} ({spatial_res['distribution_desc']})."
                        else:
                            final_answer = (
                                f"Spatial coordinate analysis verifies {len(detections)} {target_concept} footprints "
                                f"distributed across the {spatial_res['primary_sector']}."
                            )

                    is_abstained = False
                    system_logs.append(f"Spatial engine: primary_sector={spatial_res.get('primary_sector')}, clustered={spatial_res.get('is_clustered')}.")

            # 7C. PRESENCE / ABSENCE
            elif classification.intent == QueryIntent.PRESENCE_ABSENCE:
                if vlm_negative or len(detections) == 0:
                    final_answer = f"No, visual analysis does not detect reliable evidence of {target_concept} in this satellite image."
                    is_abstained = True
                    detections = []
                    reason_str = f"No validated footprints detected for {target_concept}."
                else:
                    final_answer = f"Confirmed presence of {target_concept} features in this imagery. {len(detections)} distinct footprint region(s) localized."
                    is_abstained = False

            # 7D. SCENE UNDERSTANDING
            elif classification.intent == QueryIntent.SCENE_UNDERSTANDING:
                if raw_vlm_text and raw_vlm_text.strip():
                    final_answer = f"Remote-sensing landscape classification: {raw_vlm_text}."
                    is_abstained = False
                else:
                    # Multi-feature ground check
                    b_check = grounding_engine.ground_target_concept(image, "building", vlm_affirmative=True, max_detections=3)
                    r_check = grounding_engine.ground_target_concept(image, "road", vlm_affirmative=True, max_detections=3)
                    v_check = grounding_engine.ground_target_concept(image, "vegetation", vlm_affirmative=True, max_detections=3)

                    if b_check or r_check:
                        final_answer = "Remote-sensing landscape classification: Developed urban/semi-urban area with visible structural footprints and corridors."
                        is_abstained = False
                        detections = b_check or r_check
                    elif v_check:
                        final_answer = "Remote-sensing landscape classification: Natural/rural landscape with dominant vegetation or open ground cover."
                        is_abstained = False
                        detections = v_check
                    else:
                        final_answer = "The image does not provide enough visual evidence to classify the scene environment reliably."
                        is_abstained = True
                        detections = []
                        reason_str = "Insufficient macro visual features."

            # 7E. GENERAL VQA
            else:
                if raw_vlm_text and raw_vlm_text.strip() and not vlm_negative:
                    final_answer = f"Based on visual analysis of the satellite imagery: {raw_vlm_text}."
                    is_abstained = False
                elif any(q_phrase in question.lower() for q_phrase in ["what objects", "what do you see", "what is visible", "describe"]):
                    # Discover grounded feature types
                    discovered = []
                    b_found = grounding_engine.ground_target_concept(image, "building", vlm_affirmative=True, max_detections=5)
                    if b_found:
                        discovered.append("building structures")
                    r_found = grounding_engine.ground_target_concept(image, "road", vlm_affirmative=True, max_detections=5)
                    if r_found:
                        discovered.append("road corridors")
                    v_found = grounding_engine.ground_target_concept(image, "vegetation", vlm_affirmative=True, max_detections=5)
                    if v_found:
                        discovered.append("vegetation zones")
                    w_found = grounding_engine.ground_target_concept(image, "water", vlm_affirmative=True, max_detections=5)
                    if w_found:
                        discovered.append("water bodies")

                    if discovered:
                        final_answer = f"Visual analysis identifies prominent {', '.join(discovered)} across the scene."
                        is_abstained = False
                        detections = b_found or r_found or v_found or []
                    else:
                        final_answer = "The image does not provide enough visual contrast or resolution to identify distinct object categories reliably."
                        is_abstained = True
                        detections = []
                        reason_str = "No distinct object boundaries resolved."
                else:
                    final_answer = "I can't determine that reliably from the provided image."
                    is_abstained = True
                    detections = []
                    reason_str = "Visual evidence is insufficient or ambiguous."

            # 8. Evidence Verification & Hallucination Guard
            evidence_found, abstained_guard, confidence, verifier_reason = evidence_verifier.verify_and_calibrate(
                vlm_text=raw_vlm_text,
                target_concept=target_concept,
                detections=detections,
                intent=classification.intent.value
            )

            if is_abstained or len(detections) == 0:
                is_abstained = True
                evidence_found = False
                detections = []  # Strict zero-tolerance: purge any spurious boxes on abstention
                confidence = None  # Never fabricate confidence when abstaining
                reason_str = reason_str or verifier_reason
            else:
                is_abstained = False
                evidence_found = True

            # 9. Render Annotated Visual Evidence Image
            evidence_image_b64 = None
            if evidence_found and detections:
                evidence_image_b64 = evidence_verifier.render_annotated_evidence(
                    image=image,
                    detections=detections,
                    target_concept=target_concept
                )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            system_logs.append(f"Analysis completed in {elapsed_ms}ms with abstained={is_abstained}.")

            return AnalysisResponse(
                question=question,
                answer=final_answer,
                targets=classification.target_concepts,
                evidence_found=evidence_found,
                detections=detections,
                abstained=is_abstained,
                evidence_image=evidence_image_b64,
                status="abstained" if is_abstained else "answered",
                evidence_score=confidence,
                reason=reason_str if is_abstained else None,
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent=classification.normalized_intent,
                    image_resolution=[img_width, img_height],
                    image_size_kb=round(image_size_kb, 2),
                    latency_ms=elapsed_ms,
                    model_id=model_name,
                    evidence_threshold=0.50,
                    abstention_triggered=is_abstained,
                    mock_mode=False,
                    system_log=system_logs
                )
            )

        except Exception as exc:
            # Section 12: Safe error handling without crashing or exposing stack traces
            print(f"[VisionService] Handled inference error: {exc}")
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResponse(
                question=question,
                answer="I can't determine that reliably from the provided image.",
                targets=[target_concept if 'target_concept' in locals() else "feature"],
                evidence_found=False,
                detections=[],
                abstained=True,
                status="abstained",
                reason="Visual verification could not be completed reliably.",
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent="Safe Error Abstention Guard",
                    image_resolution=[img_width, img_height] if 'img_width' in locals() else None,
                    image_size_kb=round(image_size_kb, 2) if 'image_size_kb' in locals() else None,
                    latency_ms=elapsed_ms,
                    model_id="ErrorGuard",
                    evidence_threshold=0.50,
                    abstention_triggered=True,
                    system_log=[f"Safe fallback triggered due to: {type(exc).__name__}"]
                )
            )


vision_service = VisionService()
