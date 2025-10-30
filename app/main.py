from fastapi import FastAPI

from app.routes import rag_router, echo_router

app = FastAPI()

# 라우터를 별도 모듈로 분리해 API 구성을 단순화.
app.include_router(rag_router)
app.include_router(echo_router)
