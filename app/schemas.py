"""
API 요청/응답 스키마
"""
from pydantic import BaseModel
from typing import List, Dict,Optional
from pydantic import BaseModel, Field
class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "Tax RAG API is running"


class AskRequest(BaseModel):
    question: str
    history: Optional[List[Dict]] = None
    summary: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    elapsed_time: float
    is_web_search: bool = False


class SummarizeRequest(BaseModel):
    messages: List[Dict] = Field(..., description="요약할 대화 메시지 리스트")
    previousSummary: Optional[str] = Field(None, description="이전 요약 내용")


class SummarizeResponse(BaseModel):
    summary: str
    message_count: int