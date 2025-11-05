"""
RAG 엔드포인트
"""
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from app.schemas import AskRequest, AskResponse
from app.services.workflow import run_workflow, stream_workflow

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """
    질문을 받아 답변을 생성합니다.
    
    - **question**: 세법 관련 질문
    """
    start_time = time.time()
    
    # 워크플로우 실행
    result = run_workflow(req.question)
    
    elapsed_time = time.time() - start_time
    
    return AskResponse(
        answer=result['answer'],
        elapsed_time=round(elapsed_time, 2),
        is_web_search=result['is_web_search']
    )


@router.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """
    질문을 받아 스트리밍 방식으로 답변을 생성합니다.
    
    - **question**: 세법 관련 질문
    - 답변이 생성되는 즉시 토큰 단위로 반환됩니다.
    """
    async def plain_stream() -> AsyncGenerator[str, None]:
        """스트리밍 응답을 생성하는 제너레이터"""
        async for chunk in stream_workflow(req.question):
            yield chunk  # SSE 포맷 없이 문자열만 흘려보냄
    
    return StreamingResponse(
        plain_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )
