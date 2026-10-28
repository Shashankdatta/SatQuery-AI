from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.services.vision_service import vision_service

app = FastAPI(
    title="SatQuery AI Backend",
    description="Remote-Sensing Vision-Language Assistant Backend (ST-09)",
    version="0.3.0",
)

# CORS Configuration allowing frontend access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start background model loading when server begins
@app.on_event("startup")
async def startup_event():
    print("[SatQuery AI] Initializing Vision Service...")
    try:
        vision_service.start_background_loading()
    except Exception as e:
        print(f"[SatQuery AI] Note on load: {e}")

# Mount endpoints at both root and /api
app.include_router(api_router, tags=["Analysis"])
app.include_router(api_router, prefix="/api", tags=["Analysis (API)"])

@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health_check():
    status = vision_service.get_status()
    return {
        "status": "healthy",
        "zoomearth": True,
        "grounding": True,
        "service": "SatQuery AI Remote-Sensing Assistant",
        "version": "0.3.0",
        "engine": status.get("engine", "Vision-Language Assistant"),
        "model_status": status.get("status", "ready"),
        "progress": status.get("progress", "Active")
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "SatQuery AI",
        "description": "Remote-Sensing Vision-Language Assistant (Problem Statement ST-09)",
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health"
        }
    }
