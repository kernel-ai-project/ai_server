
"""
LangGraph 워크플로우
"""
from typing import List, Literal, AsyncGenerator
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from app.config import settings
from app.services.retriever import get_retriever_parallel
from app.services.generator import generate_answer, stream_generate_answer


# ============================================================
# State 정의
# ============================================================
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str
    is_web_search: bool


# ============================================================
# 문서 관련성 체크 스키마
# ============================================================
class RelevanceScore(BaseModel):
    score: Literal[0, 1] = Field(
        description="0=문서로 답변 가능, 1=문서로 답변 불가"
    )


# ============================================================
# 전역 변수
# ============================================================
tavily_search_tool = None
relevance_chain = None
graph = None


# ============================================================
# 웹 검색 도구 초기화
# ============================================================
def initialize_web_search():
    """웹 검색 도구를 초기화합니다."""
    global tavily_search_tool
    
    print("웹 검색 도구 초기화 중...")
    
    tavily_search_tool = TavilySearchResults(
        max_results=settings.TAVILY_MAX_RESULTS,
        search_depth=settings.TAVILY_SEARCH_DEPTH,
        include_answer=True,
    )
    
    print("✅ 웹 검색 도구 초기화 완료")


# ============================================================
# 관련성 체크 체인 초기화
# ============================================================
def initialize_relevance_chain():
    """관련성 체크 체인을 초기화합니다."""
    global relevance_chain
    
    print("관련성 체크 체인 초기화 중...")
    
    from app.services.generator import llm
    
    if llm is None:
        raise RuntimeError("LLM이 초기화되지 않았습니다.")
    
    relevance_system_prompt = """문서가 질문에 답변할 수 있는지 판단하세요.

답변 가능(0): 문서에 질문과 관련된 구체적 정보가 있음
답변 불가(1): 문서가 없거나 질문과 무관하거나 정보 부족

불확실하면 0을 반환하세요."""
    
    relevance_prompt = ChatPromptTemplate.from_messages([
        ('system', relevance_system_prompt),
        ('user', "질문: {question}\n\n문서:\n{documents}")
    ])
    
    structured_relevance_llm = llm.with_structured_output(RelevanceScore)
    relevance_chain = relevance_prompt | structured_relevance_llm
    
    print("✅ 관련성 체크 체인 초기화 완료")


# ============================================================
# 노드 함수들
# ============================================================
def retrieve_node(state: AgentState):
    """문서 검색 노드"""
    print(f"\n🔍 문서 검색 중: {state['query']}")
    docs = get_retriever_parallel(state['query'])
    return {'context': docs, 'is_web_search': False}


def generate_node(state: AgentState):
    """답변 생성 노드"""
    context = state['context']
    is_web_search = state.get('is_web_search', False)
    
    print(f"\n✏️ 답변 생성 중 (웹검색: {is_web_search})")
    
    if not context:
        return {'answer': "관련 정보를 찾을 수 없습니다."}
    
    answer = generate_answer(state['query'], context, is_web_search)
    return {'answer': answer}


def web_search(state: AgentState) -> AgentState:
    """웹 검색을 수행합니다."""
    query = state['query']
    print(f"\n🌐 웹 검색 중: {query}")
    results = tavily_search_tool.invoke(query)
    return {'context': results, 'is_web_search': True}


# ============================================================
# 조건부 엣지 함수
# ============================================================
def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    """문서 관련성을 체크합니다."""
    context = state['context']
    
    # 1. 문서가 없으면 irrelevant
    if not context:
        print("⚠️ 검색된 문서 없음 -> 웹서치")
        return 'irrelevant'
    
    # 2. 문서가 2개 이상이면 relevant (개선)
    if len(context) >= 2:
        print(f"✅ 문서 {len(context)}개 발견 -> 문서 기반 답변")
        return 'relevant'
    
    # 3. 문서가 1개일 때만 LLM으로 관련성 체크
    try:
        response = relevance_chain.invoke({
            'question': state['query'], 
            'documents': context[:3]
        })
        
        result = 'relevant' if response.score == 0 else 'irrelevant'
        print(f"📊 관련성 점수: {response.score} -> {result}")
        return result
        
    except Exception as e:
        # 4. 예외 발생 시 문서가 있으면 relevant로 처리 (개선)
        print(f"⚠️ 관련성 체크 실패: {e} -> 문서 기반 답변 시도")
        return 'relevant'


# ============================================================
# 그래프 구축
# ============================================================
def build_graph():
    """LangGraph 워크플로우를 구축합니다."""
    global graph
    
    print("LangGraph 워크플로우 구축 중...")
    
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node('retrieve_node', retrieve_node)
    graph_builder.add_node('generate_node', generate_node)
    graph_builder.add_node('web_search', web_search)
    
    graph_builder.add_edge(START, 'retrieve_node')
    graph_builder.add_conditional_edges(
        'retrieve_node',
        check_doc_relevance,
        {
            'irrelevant': 'web_search',
            'relevant': 'generate_node',
        }
    )
    graph_builder.add_edge('web_search', 'generate_node')
    graph_builder.add_edge('generate_node', END)
    
    graph = graph_builder.compile()
    
    print("✅ LangGraph 워크플로우 구축 완료")


# ============================================================
# 실행 함수
# ============================================================
def run_workflow(query: str) -> dict:
    """
    질문에 대한 답변을 생성합니다.
    
    Args:
        query: 사용자 질문
    
    Returns:
        워크플로우 실행 결과 (answer, is_web_search 포함)
    """
    initial_state = {"query": query}
    result = graph.invoke(initial_state)
    
    return {
        'answer': result.get('answer', '답변을 생성할 수 없습니다.'),
        'is_web_search': result.get('is_web_search', False)
    }


async def stream_workflow(query: str) -> AsyncGenerator[str, None]:
    """
    질문을 스트리밍 방식으로 처리하여 토큰 단위로 답변을 생성합니다.
    
    Args:
        query: 사용자 질문
    
    Yields:
        생성되는 답변 텍스트 조각
    """
    # 1. 문서 검색
    print(f"\n🔍 문서 검색 중: {query}")
    docs = get_retriever_parallel(query)
    
    # 2. 문서 관련성 체크
    is_web_search = False
    context = docs
    
    if not context:
        print("⚠️ 검색된 문서 없음 -> 웹서치")
        is_web_search = True
        context = tavily_search_tool.invoke(query)
    elif len(context) >= 2:
        print(f"✅ 문서 {len(context)}개 발견 -> 문서 기반 답변")
    else:
        # 문서가 1개일 때만 관련성 체크
        try:
            response = relevance_chain.invoke({
                'question': query, 
                'documents': context[:3]
            })
            
            if response.score == 1:
                print("📊 관련성 낮음 -> 웹서치")
                is_web_search = True
                context = tavily_search_tool.invoke(query)
            else:
                print("📊 관련성 충분 -> 문서 기반 답변")
        except Exception as e:
            print(f"⚠️ 관련성 체크 실패: {e} -> 문서 기반 답변 시도")
    
    # 3. 스트리밍 답변 생성
    print(f"\n✏️ 답변 생성 중 (웹검색: {is_web_search})")
    
    if not context:
        yield "관련 정보를 찾을 수 없습니다."
        return
    
    async for chunk in stream_generate_answer(query, context, is_web_search):
        yield chunk


# ============================================================
# 초기화 함수
# ============================================================
def initialize_workflow():
    """워크플로우를 초기화합니다."""
    initialize_web_search()
    initialize_relevance_chain()
    
    # retriever_chain 초기화 (LLM 의존)
    from app.services.retriever import setup_retriever_chain
    setup_retriever_chain()
    
    build_graph()
    print("✅ 워크플로우 초기화 완료\n")