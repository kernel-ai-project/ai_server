"""
API 요청/응답 스키마
"""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    message: str = "Tax RAG API is running"


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    elapsed_time: float
    is_web_search: bool = False


