---
title: SatQuery AI Backend
emoji: 🛰️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# SatQuery AI — Remote-Sensing Vision-Language Assistant Backend

High-performance FastAPI inference backend for Problem Statement **ST-09** (ISRO Remote-Sensing Vision-Language Assistant).

### Features
- Remote-Sensing VQA inference (Salesforce BLIP / ZoomEarth architecture)
- Query Intent Classification (Presence, Counting, Spatial, Attributes, Change)
- Multi-scale Visual Grounding with connected-component instance extraction
- Cross-modal Evidence Verification and Abstention Engine
- OpenAPI / Swagger documentation at `/docs`
- Health probe at `/health`
