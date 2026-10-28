from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, status
from typing import Dict, Any, Optional

from app.schemas.response import AnalysisResponse
from app.services.vision_service import vision_service

router = APIRouter()

@router.get("/model/status", summary="Get model loading status")
async def get_model_status() -> Dict[str, Any]:
    return vision_service.get_status()

@router.post("/analyze", response_model=AnalysisResponse, summary="Analyze image with Vision-Language Assistant")
async def analyze_image(
    image: UploadFile = File(..., description="Uploaded satellite image file (JPG, PNG, TIFF)"),
    question: str = Form(..., description="Natural language question about the image"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="Optional Google Gemini API key")
) -> AnalysisResponse:
    """
    Core vision-language query endpoint.
    Processes the uploaded satellite image and question and returns the grounded answer.
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

        response = await vision_service.analyze(
            image_bytes=image_bytes,
            filename=image.filename or "image.png",
            question=question.strip(),
            api_key=x_api_key
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
