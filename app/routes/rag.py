"""
RAG 엔드포인트
"""
import time
from fastapi import APIRouter,HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from app.schemas import AskRequest, AskResponse,SummarizeResponse, SummarizeRequest
from app.services.workflow import run_workflow, stream_workflow
from app.services.summarization import generate_summary
router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_conversation(req: SummarizeRequest) -> SummarizeResponse:
    """
    대화 내용을 요약합니다.
    
    스프링에서 호출 예시:
    POST /history/summarize
    {
        "messages": [
            {"role": "user", "content": "소득세율은?"},
            {"role": "assistant", "content": "소득세율은..."}
        ]
    }
    """
    print("요약을 실시합니다!!!!!")
    if not req.messages:
        raise HTTPException(
            status_code=400,
            detail="messages 필드가 비어있습니다."
        )
    
    summary = generate_summary(req.messages, req.previousSummary)
    
    return SummarizeResponse(
        summary=summary,
        message_count=len(req.messages)
    )

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    start_time = time.time()
    
    # history, summary 전달
    result = run_workflow(req.question, req.history, req.summary)
    
    elapsed_time = time.time() - start_time
    
    return AskResponse(
        answer=result['answer'],
        elapsed_time=round(elapsed_time, 2),
        is_web_search=result['is_web_search']
    )


@router.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    async def plain_stream() -> AsyncGenerator[str, None]:
        async for chunk in stream_workflow(req.question, req.history, req.summary):
            yield chunk
    
    return StreamingResponse(
        plain_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )