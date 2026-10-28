import io
import time
import threading
import re
import json
from typing import Optional, Dict, Any, List
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from app.schemas.response import AnalysisResponse, DetectedObject, QueryInspectorData


def parse_qwen_bounding_boxes(text: str, img_width: int, img_height: int) -> List[DetectedObject]:
    """
    Parses Qwen2.5-VL bounding box representations if present in the generated answer.
    Supports formats:
      1. {"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}
      2. [ymin, xmin, ymax, xmax] normalized to 1000
    If no usable coordinates are present, returns [] without fabricating results.
    """
    objects: List[DetectedObject] = []

    # Check for JSON objects with box_2d
    json_candidates = re.findall(r'\{[^{}]*box_2d[^{}]*\}', text, re.IGNORECASE)
    for jc in json_candidates:
        try:
            parsed = json.loads(jc)
            if "box_2d" in parsed and len(parsed["box_2d"]) == 4:
                b = parsed["box_2d"]
                if max(b) <= 1000:
                    abs_b = [
                        int(b[0] * img_height / 1000),
                        int(b[1] * img_width / 1000),
                        int(b[2] * img_height / 1000),
                        int(b[3] * img_width / 1000),
                    ]
                else:
                    abs_b = [int(v) for v in b]
                label = parsed.get("label", "building")
                objects.append(DetectedObject(label=label, box=abs_b))
        except Exception:
            pass

    # Check for [ymin, xmin, ymax, xmax] coordinate list
    if not objects:
        box_pattern = re.findall(r'\[(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4})\]', text)
        for bp in box_pattern:
            try:
                coords = [int(x) for x in bp]
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    if max(coords) <= 1000:
                        abs_b = [
                            int(coords[0] * img_height / 1000),
                            int(coords[1] * img_width / 1000),
                            int(coords[2] * img_height / 1000),
                            int(coords[3] * img_width / 1000),
                        ]
                    else:
                        abs_b = coords
                    objects.append(DetectedObject(label="building", box=abs_b))
            except Exception:
                pass

    return objects


class QwenVLService:
    """
    Official Vision-Language service for Qwen/Qwen2.5-VL-3B-Instruct.
    Implements the official Hugging Face Transformers approach:
      - Qwen2_5_VLForConditionalGeneration
      - AutoProcessor
      - qwen_vl_utils.process_vision_info
    """

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(self):
        self.model = None
        self.processor = None
        self.status = "unloaded"  # "unloaded" | "loading" | "ready" | "error"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.error_message: Optional[str] = None
        self.load_progress: str = "Awaiting initialization"
        self._lock = threading.Lock()
        self._load_thread: Optional[threading.Thread] = None

    def start_background_loading(self):
        """Asynchronously load model weights without blocking the web server."""
        with self._lock:
            if self.status in ("loading", "ready"):
                return
            self.status = "loading"
            self.load_progress = "Initializing AutoProcessor and downloading model weights..."
            self.error_message = None

        def _loader():
            try:
                print(f"[QwenService] Device target: {self.device.upper()} (dtype: {self.torch_dtype})")

                print(f"[QwenService] Loading AutoProcessor for {self.MODEL_ID}...")
                self.load_progress = "Loading tokenizer and vision processor..."
                self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
                print("[QwenService] AutoProcessor loaded successfully.")

                print(f"[QwenService] Loading model weights from {self.MODEL_ID}...")
                self.load_progress = "Downloading and mapping Qwen2.5-VL-3B-Instruct weights..."

                device_map = "auto" if torch.cuda.is_available() else None

                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )

                if not torch.cuda.is_available():
                    self.model = self.model.to(self.device)

                self.model.eval()

                with self._lock:
                    self.status = "ready"
                    self.load_progress = f"Model loaded successfully on {self.device.upper()}"
                print(f"[QwenService] Model {self.MODEL_ID} successfully ready on {self.device.upper()}!")

            except Exception as exc:
                import traceback
                traceback.print_exc()
                with self._lock:
                    self.status = "error"
                    self.error_message = str(exc)
                    self.load_progress = f"Load failed: {str(exc)}"
                print(f"[QwenService] Failed to load model: {exc}")

        self._load_thread = threading.Thread(target=_loader, daemon=True)
        self._load_thread.start()

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "model_id": self.MODEL_ID,
            "progress": self.load_progress,
            "error": self.error_message,
        }

    async def analyze(
        self,
        image_bytes: bytes,
        filename: str,
        question: str,
    ) -> AnalysisResponse:
        start_time = time.perf_counter()

        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded image file is empty.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Decode image to RGB format
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_width, img_height = image.size
            image_size_kb = len(image_bytes) / 1024.0
        except Exception as exc:
            raise ValueError(f"Failed to decode image file '{filename}': {str(exc)}")

        # Check for insufficient image geometry threshold
        if img_width < 32 or img_height < 32:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResponse(
                status="abstained",
                answer=None,
                evidence_score=None,
                objects=[],
                reason="Image resolution is too low (< 32px) to provide reliable visual evidence for remote-sensing analysis.",
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent="Resolution Check (Abstention)",
                    image_resolution=[img_width, img_height],
                    image_size_kb=round(image_size_kb, 2),
                    latency_ms=elapsed_ms,
                    model_id=self.MODEL_ID,
                    evidence_threshold=0.50,
                    abstention_triggered=True,
                    mock_mode=False,
                    system_log=["Abstention triggered: image resolution below minimal threshold for remote-sensing feature resolution."]
                )
            )

        # Handle loading state
        if self.status == "loading":
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResponse(
                status="loading",
                answer=None,
                evidence_score=None,
                objects=[],
                reason=f"Qwen2.5-VL-3B-Instruct is currently loading ({self.load_progress}). Please try again shortly.",
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent="Model Initializing",
                    image_resolution=[img_width, img_height],
                    image_size_kb=round(image_size_kb, 2),
                    latency_ms=elapsed_ms,
                    model_id=self.MODEL_ID,
                    evidence_threshold=0.50,
                    abstention_triggered=False,
                    mock_mode=False,
                    system_log=[
                        f"Image: {img_width}x{img_height} px ({image_size_kb:.1f} KB)",
                        f"Query: '{question}'",
                        f"Status: {self.load_progress}"
                    ]
                )
            )

        # Handle error state
        if self.status == "error":
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResponse(
                status="error",
                answer=None,
                evidence_score=None,
                objects=[],
                reason=f"Qwen2.5-VL-3B-Instruct model failed to load: {self.error_message}",
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent="Model Load Failure",
                    image_resolution=[img_width, img_height],
                    image_size_kb=round(image_size_kb, 2),
                    latency_ms=elapsed_ms,
                    model_id=self.MODEL_ID,
                    evidence_threshold=0.50,
                    abstention_triggered=True,
                    mock_mode=False,
                    system_log=[f"Error detail: {self.error_message}"]
                )
            )

        # Handle unloaded state (trigger background loading)
        if self.status == "unloaded":
            self.start_background_loading()
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AnalysisResponse(
                status="loading",
                answer=None,
                evidence_score=None,
                objects=[],
                reason="Qwen2.5-VL-3B-Instruct load initiated. Please wait for model weights to initialize.",
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent="Model Load Initiated",
                    image_resolution=[img_width, img_height],
                    image_size_kb=round(image_size_kb, 2),
                    latency_ms=elapsed_ms,
                    model_id=self.MODEL_ID,
                    evidence_threshold=0.50,
                    abstention_triggered=False,
                    mock_mode=False,
                    system_log=["Initiated background model loading"]
                )
            )

        # Execute official Qwen2.5-VL inference
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question.strip()},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                answer = output_text[0].strip()

            detected_objects = parse_qwen_bounding_boxes(answer, img_width, img_height)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Check if model answer indicates insufficient evidence or refusal
            lower_ans = answer.lower()
            abstention_indicators = [
                "insufficient visual evidence",
                "cannot be determined from this image",
                "cannot be determined from the provided image",
                "not enough visual evidence",
                "cannot determine from the image",
                "not visible in this image",
                "not visible in the provided image",
                "cannot confirm from this image",
                "cannot reliably answer",
                "unable to determine from this image",
                "no visual evidence of",
            ]
            is_abstained = any(indicator in lower_ans for indicator in abstention_indicators)

            if is_abstained:
                return AnalysisResponse(
                    question=question,
                    answer=answer,
                    targets=["structure", "object"],
                    evidence_found=False,
                    detections=detected_objects,
                    abstained=True,
                    status="abstained",
                    evidence_score=None,
                    objects=detected_objects,
                    reason="The model determined that the image does not provide sufficient visual evidence to answer this question reliably.",
                    inspector=QueryInspectorData(
                        raw_query=question,
                        normalized_intent="Abstention / Refusal Triggered",
                        image_resolution=[img_width, img_height],
                        image_size_kb=round(image_size_kb, 2),
                        latency_ms=elapsed_ms,
                        model_id=self.MODEL_ID,
                        evidence_threshold=0.50,
                        abstention_triggered=True,
                        mock_mode=False,
                        system_log=[
                            f"Image: {img_width}x{img_height} px ({image_size_kb:.1f} KB)",
                            f"Query: '{question}'",
                            f"Device: {self.model.device}",
                            "Abstention triggered based on visual evidence insufficiency signals.",
                            f"Generation completed in {elapsed_ms} ms",
                        ],
                    ),
                )

            return AnalysisResponse(
                question=question,
                answer=answer,
                targets=["structure", "object"],
                evidence_found=len(detected_objects) > 0,
                detections=detected_objects,
                abstained=False,
                status="answered",
                evidence_score=None,
                objects=detected_objects,
                reason=None,
                inspector=QueryInspectorData(
                    raw_query=question,
                    normalized_intent="Qwen2.5-VL-3B-Instruct Inference",
                    image_resolution=[img_width, img_height],
                    image_size_kb=round(image_size_kb, 2),
                    latency_ms=elapsed_ms,
                    model_id=self.MODEL_ID,
                    evidence_threshold=0.50,
                    abstention_triggered=False,
                    mock_mode=False,
                    system_log=[
                        f"Image: {img_width}x{img_height} px ({image_size_kb:.1f} KB)",
                        f"Query: '{question}'",
                        f"Device: {self.model.device}",
                        f"Detected grounding boxes: {len(detected_objects)}",
                        f"Generation completed in {elapsed_ms} ms",
                    ],
                ),
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            raise RuntimeError(f"Qwen2.5-VL inference failed: {str(exc)}")


# Global singleton instance
qwen_service = QwenVLService()

