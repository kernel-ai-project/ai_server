
"""
답변 생성 관련 로직
"""
from typing import List,Dict
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
    ("system", """당신은 세법 전문가입니다. 주어진 법률 조항과 대화 맥락을 바탕으로 명확한 답변을 제공하세요.

{summary}

{history}

답변 규칙:
- 사용자가 "아까", "전에", "그거" 등 이전 대화를 언급하면 위의 대화 요약과 최근 대화 내역, 그리고 찾아온 문서를 바탕으로을 참고하여 답변하세요.
- 이전에 나눴던 대화 내용과 연결지어 답변하세요
- 결론 제시
- 구체적인 관련 법률 조항 명시 (ex. 소득세법 제55조 2항)
- 핵심 내용 간결하게 설명
- 구체적 수치/기준 제시

참고 문서:
{context}"""),
    ("user", "{question}")
])


WEB_SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 세금 질문에 답변하는 AI입니다. 웹 검색 결과와 대화 맥락을 바탕으로 답변하세요.

{summary}

{history}

답변 규칙:
- 사용자가 "아까", "전에", "그거" 등 이전 대화를 언급하면 위의 대화 요약과 최근 대화 내역, 그리고 찾아온 문서를 바탕으로을 참고하여 답변하세요.
- 이전에 나눴던 대화 내용과 연결지어 답변하세요
- 결론 제시
- 구체적인 관련 법률 조항 명시 (ex. 소득세법 제55조 2항)
- 핵심 내용 간결하게 설명
- 구체적 수치/기준 제시

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

def generate_answer(
    query: str,
    context: List[Document],
    is_web_search: bool,
    history: List[Dict] = None,
    summary: str = None
) -> str:
    if not context:
        return "관련 정보를 찾을 수 없습니다."
    
    limited_context = [
        Document(
            page_content=doc.page_content[:settings.CONTEXT_CHAR_LIMIT] if isinstance(doc, Document) else str(doc)[:settings.CONTEXT_CHAR_LIMIT],
            metadata=doc.metadata if isinstance(doc, Document) and hasattr(doc, 'metadata') else {}
        )
        for doc in context[:settings.MAX_CONTEXT_DOCS]
    ]
    
    if is_web_search:
        chain = WEB_SEARCH_PROMPT | search_llm
    else:
        chain = TAX_LAW_PROMPT | llm
    
    # history, summary 포맷팅
    history_text = ""
    if history:
        history_text = "이전 대화:\n" + "\n".join([
            f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}" 
            for msg in history[-3:]
        ])    
    summary_text = ""
    if summary:
        summary_text = f"대화 요약:\n{summary}"
    
    response = chain.invoke({
        'question': query,
        'context': limited_context,
        'history': history_text,
        'summary': summary_text
    })
    
    return response.content


async def stream_generate_answer(
    query: str,
    context: List[Document],
    is_web_search: bool,
    history: List[Dict] = None,
    summary: str = None
):
    if not context:
        yield "관련 정보를 찾을 수 없습니다."
        return
    
    limited_context = [
        Document(
            page_content=doc.page_content[:settings.CONTEXT_CHAR_LIMIT] if isinstance(doc, Document) else str(doc)[:settings.CONTEXT_CHAR_LIMIT],
            metadata=doc.metadata if isinstance(doc, Document) and hasattr(doc, 'metadata') else {}
        )
        for doc in context[:settings.MAX_CONTEXT_DOCS]
    ]
    
    if is_web_search:
        chain = WEB_SEARCH_PROMPT | search_llm
    else:
        chain = TAX_LAW_PROMPT | llm
    
    # history, summary 포맷팅
    history_text = ""
    if history:
        history_text = "이전 대화:\n" + "\n".join([
            f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}" 
            for msg in history[-3:]
        ])
        
    summary_text = ""
    if summary:
        summary_text = f"대화 요약:\n{summary}"
    
    async for chunk in chain.astream({
        'question': query,
        'context': limited_context,
        'history': history_text,
        'summary': summary_text
    }):
        if chunk.content:
            yield chunk.content