"""
대화 요약 생성
"""
from typing import List, Dict,Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings


summary_llm = None


INITIAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 대화 내용을 간결하게 요약하는 AI입니다.

다음 규칙에 따라 대화를 요약하세요:
1. 핵심 주제와 질문 내용을 파악
2. 중요한 정보와 답변 내용 정리
3. 3-5개 문장으로 간결하게 요약
4. 시간 순서대로 정리

요약 형식:
- 주요 질문: [질문 내용]
- 핵심 답변: [답변 요약]
- 추가 내용: [기타 중요 정보]"""),
    ("user", "다음 대화 내용을 요약해주세요:\n\n{conversation}")
])



# 누적 요약용 프롬프트
INCREMENTAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 대화 내용을 상세하게 요약하는 AI입니다.

이전 대화 요약과 새로운 대화를 종합하여 전체 대화를 요약하세요.

요약 원칙:
1. 사용자가 질문한 내용을 구체적으로 기록
2. AI가 답변한 핵심 내용을 상세히 기록
3. 법률 조항, 수치, 기준 등 구체적 정보는 반드시 포함
4. 시간 순서대로 정리
5. 이전 요약의 내용을 유지하면서 새로운 내용을 추가

요약 형식:
[이전 대화 내용]
- 질문1: [구체적 질문 내용]
  답변1: [핵심 답변 및 법률 조항, 수치]
- 질문2: [구체적 질문 내용]
  답변2: [핵심 답변 및 법률 조항, 수치]

[새로운 대화 내용]
- 질문3: [구체적 질문 내용]
  답변3: [핵심 답변 및 법률 조항, 수치]"""),
    ("user", """이전 대화 요약:
{previous_summary}

새로운 대화:
{new_conversation}

위 내용을 종합하여 전체 대화를 상세히 요약해주세요. 
사용자가 나중에 "아까 물어봤던 소득세 관련 질문"이라고 했을 때 
어떤 내용을 물어봤는지 알 수 있을 정도로 구체적으로 작성하세요.""")
])




def initialize_summary_llm():
    """요약용 LLM을 초기화합니다."""
    global summary_llm
    
    print("요약 LLM 초기화 중...")
    
    summary_llm = ChatOpenAI(
        model=settings.SEARCH_MODEL,
        temperature=0.5,
        max_tokens=1500
    )
    
    print("✅ 요약 LLM 초기화 완료")
    return summary_llm


def generate_summary(conversation_history: List[Dict], previous_summary: Optional[str] = None) -> str:
    """대화 히스토리를 요약합니다. (누적 요약)"""
    if not conversation_history:
        return "대화 내역이 없습니다."
    
    if summary_llm is None:
        raise RuntimeError("요약 LLM이 초기화되지 않았습니다.")
    
    # 새 대화 내용 포맷팅
    conversation_text = ""
    for msg in conversation_history:
        role = "사용자" if msg["role"] == "user" else "AI"
        content = msg.get("content", "")
        conversation_text += f"{role}: {content}\n\n"
    
    # 이전 요약이 있으면 누적 요약, 없으면 첫 요약
    if previous_summary:
        chain = INCREMENTAL_SUMMARY_PROMPT | summary_llm
        response = chain.invoke({
            "previous_summary": previous_summary,
            "new_conversation": conversation_text
        })
    else:
        chain = INITIAL_SUMMARY_PROMPT | summary_llm
        response = chain.invoke({
            "conversation": conversation_text
        })
    
    return response.content