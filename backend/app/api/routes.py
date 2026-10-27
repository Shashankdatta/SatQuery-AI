from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional, Dict, Any

from app.schemas.response import AnalysisResponse
from app.services.qwen_service import qwen_service

router = APIRouter()

@router.get("/model/status", summary="Get Qwen2.5-VL-3B-Instruct model loading status")
async def get_model_status() -> Dict[str, Any]:
    """
    Returns the real-time loading/ready state of Qwen2.5-VL-3B-Instruct.
    States: 'unloaded', 'loading', 'ready', 'error'.
    """
    return qwen_service.get_status()

@router.post("/analyze", response_model=AnalysisResponse, summary="Analyze image with Qwen2.5-VL-3B-Instruct")
async def analyze_image(
    image: UploadFile = File(..., description="Uploaded image file (JPG, PNG, TIFF)"),
    question: str = Form(..., description="Natural language question about the image")
) -> AnalysisResponse:
    """
    Core vision-language query endpoint powered by Qwen2.5-VL-3B-Instruct.
    Sends the user's uploaded image and question to the model and returns the generated answer.
    """
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An image file is required."
        )

    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A non-empty question must be provided."
        )

    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty."
            )

        response = await qwen_service.analyze(
            image_bytes=image_bytes,
            filename=image.filename or "image.png",
            question=question.strip()
        )
        return response

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )
