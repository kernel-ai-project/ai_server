"""
애플리케이션 설정 관리
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """환경 설정"""
    
    # API Keys (필수 - .env에서 로드)
    OPENAI_API_KEY: str
    UPSTAGE_API_KEY: str
    TAVILY_API_KEY: str
    PINECONE_API_KEY: str | None = None
    BOK_API_KEY: str | None = None
    
    # 디렉토리 경로
    CHROMA_BASE_DIR: str = "./chroma"
    BM25_CACHE_DIR: str = "./bm25_cache"
    
    # LLM 설정
    MAIN_MODEL: str = "gpt-4o"
    SEARCH_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 400
    
    # Embedding 설정
    EMBEDDING_MODEL: str = "solar-embedding-1-large"
    
    # 검색 설정
    TOP_K_VECTOR: int = 2
    TOP_K_BM25: int = 2
    MAX_DOCS_LIMIT: int = 8
    MAX_CONTEXT_DOCS: int = 4
    CONTEXT_CHAR_LIMIT: int = 600
    
    # 병렬 처리 설정
    MAX_WORKERS: int = 3
    
    # 웹 검색 설정
    TAVILY_MAX_RESULTS: int = 3
    TAVILY_SEARCH_DEPTH: str = "basic"
    
    # 앙상블 가중치
    VECTOR_WEIGHT: float = 0.6
    BM25_WEIGHT: float = 0.4
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # 👈 이 줄 추가!


# 싱글톤 인스턴스
settings = Settings()


# 법률 목록 (검색에 사용)
AVAILABLE_LAWS = [
    "national-tax-framework-act",
    "income-tax-act",
    "corporate-tax-act",
    "inheritance-gift-tax-act",
    "comprehensive-real-estate-tax-act",
    "value-added-tax-act",
    "individual-consumption-tax-act",
    "transportation-energy-environment-tax-act",
    "liquor-tax-act",
    "securities-transaction-tax-act",
    "local-tax-act",
    "local-tax-framework-act",
    "local-tax-collection-act",
    "corporation_public_cooperation",
    "corporation_value-added-tax-act",
    "corporation_withholding-tax",
    "corporation_national-tax-framework-act",
    "corporation_comprehensive-real-estate-tax-act",
]