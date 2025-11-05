"""
헬스체크 엔드포인트
"""
from fastapi import APIRouter
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    서버 상태를 확인합니다.
    """
    return HealthResponse(
        status="ok",
        message="Tax RAG API is running"
    )