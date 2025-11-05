"""
문서 검색 관련 로직
"""
import os
import pickle
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.config import settings, AVAILABLE_LAWS


# ============================================================
# 법률 선택 스키마
# ============================================================
class MultiRawRetriever(BaseModel):
    targets: List[Literal[
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
        "corporation_comprehensive-real-estate-tax-act"
    ]] = Field(description="가장 관련성 높은 법률 1-2개 선택")


# ============================================================
# 벡터스토어 및 BM25 전역 변수
# ============================================================
vector_stores = {}
bm25_retrievers = {}
retriever_chain = None


# ============================================================
# 초기화 함수들
# ============================================================
def load_vector_stores():
    """벡터스토어를 로드합니다."""
    global vector_stores
    
    print("벡터스토어 로드 중...")
    
    embedding = UpstageEmbeddings(model=settings.EMBEDDING_MODEL)
    
    for folder_name in os.listdir(settings.CHROMA_BASE_DIR):
        folder_path = os.path.join(settings.CHROMA_BASE_DIR, folder_name)
        
        if os.path.isdir(folder_path):
            vector_stores[folder_name] = Chroma(
                collection_name=folder_name,
                persist_directory=folder_path,
                embedding_function=embedding
            )
    
    print(f"✅ {len(vector_stores)}개의 Vector Store 로드 완료")


def load_bm25_retrievers():
    """BM25 retriever를 캐시에서 로드합니다."""
    global bm25_retrievers
    
    print("BM25 인덱스 로드 중...")
    
    os.makedirs(settings.BM25_CACHE_DIR, exist_ok=True)
    
    for law_name, vectorstore in vector_stores.items():
        cache_path = os.path.join(settings.BM25_CACHE_DIR, f"{law_name}_bm25.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                bm25_retrievers[law_name] = pickle.load(f)
        else:
            # 캐시가 없으면 생성
            all_docs_data = vectorstore.get()
            docs_list = [
                Document(
                    page_content=all_docs_data['documents'][i],
                    metadata=all_docs_data['metadatas'][i] if all_docs_data['metadatas'] else {}
                )
                for i in range(len(all_docs_data['documents']))
            ]
            
            bm25_retriever = BM25Retriever.from_documents(docs_list)
            bm25_retriever.k = settings.TOP_K_BM25
            
            with open(cache_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            
            bm25_retrievers[law_name] = bm25_retriever
    
    print(f"✅ {len(bm25_retrievers)}개의 BM25 인덱스 로드 완료")


def setup_retriever_chain():
    """법률 선택 체인을 설정합니다."""
    global retriever_chain
    
    print("법률 선택 체인 설정 중...")
    
    # generator.py에서 초기화된 llm 사용
    from app.services.generator import llm
    
    if llm is None:
        raise RuntimeError("LLM이 초기화되지 않았습니다. initialize_llm()을 먼저 호출하세요.")
    
    retriever_system_prompt = """당신은 한국 세법 선택 전문가입니다.
사용자 질문에 가장 관련성 높은 법률 1-2개만 선택하세요.

법률 목록:
- national-tax-framework-act: 국세기본법 (세금 납부, 환급, 가산세 등 기본 절차)
- income-tax-act: 소득세법 (개인소득, 급여, 사업소득)
- corporate-tax-act: 법인세법 (법인 관련 세금)
- inheritance-gift-tax-act: 상속세 및 증여세법
- comprehensive-real-estate-tax-act: 종합부동산세법
- value-added-tax-act: 부가가치세법 (매출, 매입세액)
- individual-consumption-tax-act: 개별소비세법
- transportation-energy-environment-tax-act: 교통·에너지·환경세법
- liquor-tax-act: 주세법
- securities-transaction-tax-act: 증권거래세법
- local-tax-act: 지방세법 (취득세, 등록면허세, 재산세)
- local-tax-framework-act: 지방세기본법
- local-tax-collection-act: 지방세징수법
- corporation_public_cooperation: 법인 공익법인
- corporation_value-added-tax-act: 법인 부가가치세
- corporation_withholding-tax: 법인 원천징수
- corporation_national-tax-framework-act: 법인 세금 납부
- corporation_comprehensive-real-estate-tax-act: 법인 종합부동산세

세법과 무관한 질문은 빈 리스트를 반환하세요."""
    
    retriever_prompt = ChatPromptTemplate.from_messages([
        ('system', retriever_system_prompt),
        ('user', '{query}')
    ])
    
    structured_retriever_llm = llm.with_structured_output(MultiRawRetriever)
    retriever_chain = retriever_prompt | structured_retriever_llm
    
    print("✅ 법률 선택 체인 설정 완료")


# ============================================================
# 검색 함수들
# ============================================================
def retrieve_from_single_law(law_name: str, query: str) -> List[Document]:
    """단일 법률에서 하이브리드 검색을 수행합니다."""
    if law_name not in vector_stores or law_name not in bm25_retrievers:
        return []
    
    try:
        vector_retriever = vector_stores[law_name].as_retriever(
            search_type="similarity",
            search_kwargs={'k': settings.TOP_K_VECTOR}
        )
        
        bm25_retriever = bm25_retrievers[law_name]
        bm25_retriever.k = settings.TOP_K_BM25
        
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[settings.VECTOR_WEIGHT, settings.BM25_WEIGHT]
        )
        
        return ensemble.invoke(query)
    except Exception as e:
        print(f"⚠️ {law_name} 검색 실패: {e}")
        return []


def get_retriever_parallel(query: str) -> List[Document]:
    """병렬 처리로 여러 법률에서 동시 검색합니다."""
    try:
        result = retriever_chain.invoke({'query': query})
        selected_laws = result.targets
        
        if not selected_laws:
            print("⚠️ 선택된 법률 없음")
            return []
        
        print(f"📚 선택된 법률: {selected_laws}")
        
        all_docs = []
        with ThreadPoolExecutor(max_workers=min(len(selected_laws), settings.MAX_WORKERS)) as executor:
            futures = {executor.submit(retrieve_from_single_law, law, query): law for law in selected_laws}
            
            for future in as_completed(futures):
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"⚠️ 검색 실패: {e}")
                    continue
        
        # 중복 제거
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        print(f"✅ 검색된 문서: {len(unique_docs)}개")
        return unique_docs[:settings.MAX_DOCS_LIMIT]
        
    except Exception as e:
        print(f"⚠️ 검색 오류: {e}")
        return []


# ============================================================
# 초기화 함수 (startup시 호출)
# ============================================================
def initialize_retriever():
    """검색 시스템을 초기화합니다."""
    load_vector_stores()
    load_bm25_retrievers()
    # setup_retriever_chain()은 LLM 초기화 후에 호출되어야 함
    print("✅ 검색 시스템 초기화 완료\n")