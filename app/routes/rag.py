from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..schemas import AskRequest, AskResponse
from rag.workflow import run_rag, stream_rag

router = APIRouter()


# 일반 답변
@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """질문을 LangGraph RAG 워크플로에 전달."""
    answer = run_rag(req.question)
    return AskResponse(answer=answer)


# 스트리밍 방식 답변
@router.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    async def plain_stream():
        async for chunk in stream_rag(req.question):
            yield chunk  # SSE 포맷 없이 문자열만 흘려보냄

    return StreamingResponse(
        plain_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )
