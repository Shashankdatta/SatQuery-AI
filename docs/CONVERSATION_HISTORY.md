# SatQuery AI — Conversation & Project Record
**Project**: SatQuery AI — Remote-Sensing Vision-Language Assistant  
**Problem Statement**: ST-09 (ISRO / Space-Technology Hackathon)  
**Session Date**: October 27, 2026  
**Status**: Initial Project Setup & Frontend Foundation Verified (Local Servers Tested)

---

## 1. Executive Summary & Objectives

The goal of this session was to establish the initial project foundation and geospatial analysis interface for **SatQuery AI**, a remote-sensing vision-language assistant built for the **ST-09 problem statement**.

### Core Requirements Addressed:
1. **Remote-Sensing Vision-Language Workstation**: A modern, high-contrast geospatial analysis UI (not a generic chatbot).
2. **Visual Evidence Grounding**: An interactive optical viewport where bounding boxes are rendered with pixel-accurate coordinates, corner reticles, labels, and confidence tags.
3. **Abstention & Refusal Safety Layer**: Deterministic handling for queries requiring sub-resolution features or obscured imagery, preventing model hallucinations by returning an explicit abstention state.
4. **Query Inspector**: A specialized telemetry drawer displaying normalized query intent, image geometry, latency, evidence threshold evaluation, and raw JSON payload inspection.
5. **FastAPI Backend Foundation**: A modular API exposing `POST /analyze` and `GET /health` with Pydantic schemas, ready for drop-in replacement with Qwen2.5-VL-3B-Instruct weights.
6. **Local Execution & Verification**: Both frontend and backend running locally and verified via automated tests and live end-to-end HTTP requests.

---

## 2. Conversation & Development Chronology

### Step 1: Environment Discovery & Workspace Initialization
- Inspected the local environment: Node.js `v26.10.0`, npm `12.2.0`, Python `3.14.0`.
- Established project root at `C:\Users\SHASHANK DATTA\.gemini\antigravity\scratch\satquery-ai`.
- Created isolated Python virtual environment at `backend/.venv`.

### Step 2: Backend Architecture & Implementation
- Installed backend dependencies: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `pillow`, `pytest`, `httpx`.
- Created Pydantic response models in `backend/app/schemas/response.py` matching the exact required schema:
  - Answered: `status`, `answer`, `evidence_score`, `objects` with `[ymin, xmin, ymax, xmax]`, `reason: null`.
  - Abstained: `status: "abstained"`, `answer: null`, `evidence_score: 0.24`, `objects: []`, `reason: "Insufficient visual evidence to answer reliably."`.
  - Added `QueryInspectorData` schema for ST-09 telemetry.
- Implemented `RemoteSensingAnalyzer` in `backend/app/services/analyzer.py` with mock contextual inference for:
  - Aerodrome / runway & aircraft detection
  - Industrial structure counting
  - Maritime vessel & port assessment
  - Hydrological boundary detection
  - Sub-resolution / cloud occlusion abstention triggers
- Configured FastAPI app in `backend/app/main.py` with permissive CORS and route mounting at both `/` and `/api`.

### Step 3: Frontend Workstation Implementation
- Scaffolded modern React 18 + Vite 5 + Tailwind CSS application in `frontend/`.
- Designed custom geospatial color palette and HUD styles in `frontend/src/index.css`.
- Built modular UI components:
  - `Header.jsx`: Technical branding with live status pill (`READY`, `ANALYZING`, `ANSWERED`, `INSUFFICIENT EVIDENCE`).
  - `ImageBoundingCanvas.jsx`: Optical viewport with coordinate readouts, bounding boxes, reticle corners, and hover highlighting.
  - `ImageUploader.jsx`: Drag-and-drop file upload, file telemetry, and 1-click evaluation preset scenes.
  - `sampleImages.js`: Synthetic SVG test scenes for instant evaluation (Airport, Industrial, Marine Port, Cloud Occlusion).
  - `QueryChat.jsx`: Natural-language question input, query suggestions, and abstention testing toggle.
  - `AnalysisDisplay.jsx`: Grounded answer display, evidence confidence gauge with 50% threshold marker, and abstention alert card.
  - `GroundingPanel.jsx`: Synchronized object list with coordinates and confidence.
  - `QueryInspector.jsx`: ST-09 query inspector with metrics and raw JSON copy utility.
  - `api.js`: Communication service with reverse proxy fallback.

### Step 4: Verification & Testing
- Executed unit tests in `tests/test_api.py` covering health check, standard grounded response, abstention response, and validation errors (4 tests passing).
- Executed Vite production build (`npm run build`), verifying zero packaging errors.
- Launched backend daemon on `http://127.0.0.1:8000`.
- Launched frontend dev server on `http://localhost:5173`.
- Tested end-to-end communication via Python HTTP client against both direct backend and frontend reverse proxy.

---

## 3. Project Directory Map

```
satquery-ai/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── ImageBoundingCanvas.jsx
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── QueryChat.jsx
│   │   │   ├── AnalysisDisplay.jsx
│   │   │   ├── GroundingPanel.jsx
│   │   │   ├── QueryInspector.jsx
│   │   │   └── Icons.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── utils/
│   │   │   └── sampleImages.js
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── backend/
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── schemas/response.py
│   │   ├── services/analyzer.py
│   │   └── main.py
│   ├── requirements.txt
│   └── .venv/
├── models/
│   ├── vlm/README.md
│   └── detector/README.md
├── data/
│   └── samples/README.md
├── tests/
│   └── test_api.py
├── docs/
│   └── CONVERSATION_HISTORY.md
├── README.md
└── .gitignore
```

---

## 4. API Response Specifications

### Standard Grounded Response (`status: answered`)
```json
{
  "status": "answered",
  "answer": "Active runway tarmac identified with 1 commercial-class transport aircraft located near the central apron.",
  "evidence_score": 0.94,
  "objects": [
    {
      "label": "runway tarmac",
      "box": [135, 64, 180, 704],
      "confidence": 0.96
    },
    {
      "label": "commercial transport aircraft",
      "box": [232, 360, 292, 424],
      "confidence": 0.88
    }
  ],
  "reason": null,
  "inspector": {
    "raw_query": "Identify runway tarmac and detect any aircraft.",
    "normalized_intent": "Aerodrome / Transportation Infrastructure",
    "image_resolution": [800, 600],
    "image_size_kb": 2.45,
    "latency_ms": 1.2,
    "model_id": "Qwen2.5-VL-3B-Instruct (Mock Foundation)",
    "evidence_threshold": 0.50,
    "abstention_triggered": false,
    "mock_mode": true
  }
}
```

### Abstention Response (`status: abstained`)
```json
{
  "status": "abstained",
  "answer": null,
  "evidence_score": 0.24,
  "objects": [],
  "reason": "Insufficient visual evidence to answer reliably.",
  "inspector": {
    "raw_query": "Identify vehicle license plate in blurry imagery.",
    "normalized_intent": "Unresolvable / Low-Confidence Feature",
    "image_resolution": [800, 600],
    "image_size_kb": 2.1,
    "latency_ms": 0.9,
    "model_id": "Qwen2.5-VL-3B-Instruct (Mock Foundation)",
    "evidence_threshold": 0.50,
    "abstention_triggered": true,
    "mock_mode": true
  }
}
```

---

## 5. Instructions for Running Locally

### Backend
```powershell
cd "C:\Users\SHASHANK DATTA\.gemini\antigravity\scratch\satquery-ai\backend"
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### Frontend
```powershell
cd "C:\Users\SHASHANK DATTA\.gemini\antigravity\scratch\satquery-ai\frontend"
npm run dev
```
- Web Application: `http://localhost:5173`

### Tests
```powershell
cd "C:\Users\SHASHANK DATTA\.gemini\antigravity\scratch\satquery-ai"
.\backend\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

---

## 6. Next Steps for Phase 2

1. **Qwen2.5-VL-3B-Instruct Integration**:
   - Implement checkpoint loader with 4-bit/8-bit quantization in `models/vlm/loader.py`.
   - Implement visual coordinate parsing token decoder.
2. **LoRA Fine-Tuning Pipeline**:
   - Train adapter weights on remote-sensing VQA benchmarks (RSVQA / RSIVQA).
3. **Detector Integration**:
   - Integrate Grounding DINO / GLIP or native spatial tokens for open-vocabulary grounding.
4. **Calibrated Refusal Mechanism**:
   - Implement confidence thresholding on attention map entropy and visual feature similarity.
