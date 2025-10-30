from fastapi import APIRouter
from ..schemas import EchoRequest, EchoResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """간단한 헬스 체크 엔드포인트."""
    return {"status": "ok"}


@router.post("/echo", response_model=EchoResponse)
async def echo(req: EchoRequest) -> EchoResponse:
    """요청 메시지를 그대로 반환."""
    return EchoResponse(echo=req.message)
