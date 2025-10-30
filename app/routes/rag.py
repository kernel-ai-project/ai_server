from fastapi import APIRouter

from ..schemas import AskRequest, AskResponse, EchoRequest, EchoResponse
from rag.workflow import run_rag

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """간단한 헬스 체크 엔드포인트."""
    return {"status": "ok"}


@router.post("/echo", response_model=EchoResponse)
async def echo(req: EchoRequest) -> EchoResponse:
    """요청 메시지를 그대로 반환."""
    return EchoResponse(echo=req.message)


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """질문을 LangGraph RAG 워크플로에 전달."""
    answer = run_rag(req.question)
    return AskResponse(answer=answer)
