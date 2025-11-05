


"""
답변 생성 관련 로직
"""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.config import settings


# ============================================================
# LLM 전역 변수
# ============================================================
llm = None
search_llm = None


# ============================================================
# 프롬프트 템플릿
# ============================================================
TAX_LAW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 세법 전문가입니다. 주어진 법률 조항을 바탕으로 명확한 답변을 제공하세요.

답변에 포함돼야 하는 내용:
결론을 먼저 한 문장으로 제시
구체적인 관련 법률 조항 명시 (ex. 소득세법 제55조 2항)
핵심 내용 간결하게 설명
구체적 수치/기준 제시

참고 문서:
{context}"""),
    ("user", "{question}")
])


WEB_SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 세금 질문에 답변하는 AI입니다. 웹 검색 결과를 바탕으로 답변하세요.

답변에 포함돼야 하는 내용:
1. 결론을 먼저 한 문장으로 제시
2. 핵심 내용 간결하게 설명
3. 구체적 수치/기준 제시

검색 결과:
{context}"""),
    ("user", "{question}")
])


# ============================================================
# LLM 초기화
# ============================================================
def initialize_llm():
    """LLM을 초기화합니다."""
    global llm, search_llm
    
    print("LLM 초기화 중...")
    
    llm = ChatOpenAI(
        model=settings.MAIN_MODEL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS
    )
    
    search_llm = ChatOpenAI(
        model=settings.SEARCH_MODEL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS
    )
    
    print("✅ LLM 초기화 완료")
    return llm, search_llm


# ============================================================
# 답변 생성 함수
# ============================================================
def generate_answer(query: str, context: List[Document], is_web_search: bool) -> str:
    """검색 결과를 바탕으로 답변을 생성합니다."""
    if not context:
        return "관련 정보를 찾을 수 없습니다."
    
    # 컨텍스트 길이 제한 (성능 최적화)
    limited_context = [
        Document(
            page_content=doc.page_content[:settings.CONTEXT_CHAR_LIMIT] if isinstance(doc, Document) else str(doc)[:settings.CONTEXT_CHAR_LIMIT],
            metadata=doc.metadata if isinstance(doc, Document) and hasattr(doc, 'metadata') else {}
        )
        for doc in context[:settings.MAX_CONTEXT_DOCS]
    ]
    
    # 웹 검색 여부에 따라 프롬프트와 LLM 선택
    if is_web_search:
        chain = WEB_SEARCH_PROMPT | search_llm
    else:
        chain = TAX_LAW_PROMPT | llm
    
    response = chain.invoke({'question': query, 'context': limited_context})
    
    return response.content


async def stream_generate_answer(query: str, context: List[Document], is_web_search: bool):
    """검색 결과를 바탕으로 스트리밍 방식으로 답변을 생성합니다."""
    if not context:
        yield "관련 정보를 찾을 수 없습니다."
        return
    
    # 컨텍스트 길이 제한 (성능 최적화)
    limited_context = [
        Document(
            page_content=doc.page_content[:settings.CONTEXT_CHAR_LIMIT] if isinstance(doc, Document) else str(doc)[:settings.CONTEXT_CHAR_LIMIT],
            metadata=doc.metadata if isinstance(doc, Document) and hasattr(doc, 'metadata') else {}
        )
        for doc in context[:settings.MAX_CONTEXT_DOCS]
    ]
    
    # 웹 검색 여부에 따라 프롬프트와 LLM 선택
    if is_web_search:
        chain = WEB_SEARCH_PROMPT | search_llm
    else:
        chain = TAX_LAW_PROMPT | llm
    
    # 스트리밍 방식으로 LLM 호출 - astream() 사용
    async for chunk in chain.astream({'question': query, 'context': limited_context}):
        if chunk.content:
            yield chunk.content