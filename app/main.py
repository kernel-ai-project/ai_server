"""
FastAPI 메인 애플리케이션
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routes import health_router, rag_router
from app.services.retriever import initialize_retriever
from app.services.generator import initialize_llm
from app.services.workflow import initialize_workflow
from app.services.summarization import initialize_summary_llm

# 환경 변수 로드
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 시작/종료 시 실행되는 함수
    """
    # 시작 시 초기화
    print("\n" + "="*60)
    print("Tax RAG API 초기화 시작")
    print("="*60 + "\n")

    initialize_summary_llm()

    try:
        # 1. 검색 시스템 초기화 (벡터스토어, BM25)
        initialize_retriever()
        
        # 2. LLM 초기화 (반드시 retriever_chain 전에)
        initialize_llm()
        
        # 3. 워크플로우 초기화 (LLM 의존)
        initialize_workflow()
        
        print("="*60)
        print("✅ Tax RAG API 준비 완료!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 초기화 실패: {e}\n")
        import traceback
        traceback.print_exc()
        raise
    
    yield  # 앱 실행
    
    # 종료 시 정리 (필요한 경우)
    print("\nTax RAG API 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="Tax RAG API",
    description="한국 세법 질의응답 시스템 (LangGraph + RAG)",
    version="1.0.0",
    lifespan=lifespan
)


# CORS 설정 (필요한 경우)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 라우터 등록
app.include_router(health_router, tags=["Health"])
app.include_router(rag_router, tags=["RAG"])


# 루트 엔드포인트
@app.get("/")
async def root():
    """
    API 루트 엔드포인트
    """
    return {
        "message": "Tax RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)