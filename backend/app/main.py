from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.services.qwen_service import qwen_service

app = FastAPI(
    title="SatQuery AI Backend",
    description="Remote-Sensing Vision-Language Assistant Backend powered by Qwen2.5-VL-3B-Instruct",
    version="0.2.0",
)

# CORS Configuration for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start background model loading when server begins
@app.on_event("startup")
async def startup_event():
    print("[SatQuery AI] Initializing Qwen2.5-VL background loader...")
    qwen_service.start_background_loading()

# Mount endpoints at both root and /api for developer convenience
app.include_router(api_router, tags=["Analysis"])
app.include_router(api_router, prefix="/api", tags=["Analysis (API)"])

@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health_check():
    model_status = qwen_service.get_status()
    return {
        "status": "healthy",
        "service": "SatQuery AI Remote-Sensing Assistant",
        "version": "0.2.0",
        "engine": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_status": model_status["status"],
        "device": model_status["device"],
        "gpu_available": model_status["gpu_available"],
        "progress": model_status["progress"]
    }

@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "SatQuery AI",
        "description": "Remote-Sensing Vision-Language Assistant powered by Qwen2.5-VL-3B-Instruct",
        "docs_url": "/docs",
        "endpoints": {
            "analyze": "POST /analyze",
            "model_status": "GET /model/status",
            "health": "GET /health"
        }
    }
