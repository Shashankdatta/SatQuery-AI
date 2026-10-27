# Detection & Grounding Module

## Visual Evidence Grounding

This directory is designated for spatial grounding and object detection models supporting the remote-sensing assistant.

### Features
- **Object Grounding**: Generates pixel-accurate or normalized bounding box coordinates for identified objects (structures, vessels, aircraft, runways, water reservoirs, etc.).
- **Evidence Verification**: Computes spatial intersection-over-ground-truth and feature alignment scores to support the refusal/abstention mechanism.
- **Candidate Architectures**:
  - Grounding DINO / GLIP for open-vocabulary detection.
  - Native spatial grounding tokens from Qwen2.5-VL (`<|box_start|>(y1,x1,y2,x2)<|box_end|>`).
  - Oriented Bounding Box (OBB) detector for aerial objects.
