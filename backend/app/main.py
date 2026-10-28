from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.services.qwen_service import qwen_service

app = FastAPI(
    title="SatQuery AI Backend",
    description="Remote-Sensing Vision-Language Assistant Backend (ST-09)",
    version="0.3.0",
)

# CORS Configuration allowing frontend access from localhost and GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://shashankdatta.github.io",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start background model loading when server begins
@app.on_event("startup")
async def startup_event():
    print("[SatQuery AI] Initializing background model loader...")
    try:
        qwen_service.start_background_loading()
    except Exception as e:
        print(f"[SatQuery AI] Model load note: {e}")

# Mount endpoints at both root and /api
app.include_router(api_router, tags=["Analysis"])
app.include_router(api_router, prefix="/api", tags=["Analysis (API)"])

@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health_check():
    model_status = qwen_service.get_status()
    return {
        "status": "healthy",
        "zoomearth": True,
        "grounding": True,
        "service": "SatQuery AI Remote-Sensing Assistant",
        "version": "0.3.0",
        "engine": "ZoomEarth-3B / Qwen2.5-VL-3B-Instruct",
        "model_status": model_status.get("status", "ready"),
        "device": model_status.get("device", "cpu"),
        "progress": model_status.get("progress", "Operational")
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "SatQuery AI",
        "description": "Remote-Sensing Vision-Language Assistant (Problem Statement ST-09)",
        "docs_url": "/docs",
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health"
        }
    }
