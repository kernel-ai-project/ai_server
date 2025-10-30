import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import PDF_DIR, PERSIST_DIR

PRODUCT_MAP: Dict[str, Dict[str, str]] = {
    "KB다이렉트개인용자동차보험": {
        "product_name": "KB다이렉트개인용자동차보험",
        "product_type": "개인용자동차보험",
        "product_code": "KB-AUTO-PERSONAL-DIRECT",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "KB개인용자동차보험": {
        "product_name": "KB개인용자동차보험",
        "product_type": "개인용자동차보험",
        "product_code": "KB-AUTO-PERSONAL",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "KB다이렉트(인터넷)개인용자동차보험": {
        "product_name": "KB다이렉트(인터넷)개인용자동차보험",
        "product_type": "개인용자동차보험",
        "product_code": "KB-AUTO-PERSONAL-WEB",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "KB다이렉트(플랫폼)개인용자동차보험": {
        "product_name": "KB다이렉트(플랫폼)개인용자동차보험",
        "product_type": "개인용자동차보험",
        "product_code": "KB-AUTO-PERSONAL-PLATFORM",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "KB업무용자동차보험": {
        "product_name": "KB업무용자동차보험",
        "product_type": "업무용자동차보험",
        "product_code": "KB-AUTO-BIZ",
        "version": "1",
        "effective_from": "2023-09-01",
    },
    "KB다이렉트업무용자동차보험": {
        "product_name": "KB다이렉트업무용자동차보험",
        "product_type": "업무용자동차보험",
        "product_code": "KB-AUTO-BIZ-DIRECT",
        "version": "1",
        "effective_from": "2023-09-01",
    },
    "KB종합자동차보험": {
        "product_name": "KB종합자동차보험",
        "product_type": "종합자동차보험",
        "product_code": "KB-AUTO-TOTAL",
        "version": "1",
        "effective_from": "2023-09-01",
    },
    "KB다이렉트(인터넷)업무용자동차보험": {
        "product_name": "KB다이렉트(인터넷)업무용자동차보험",
        "product_type": "업무용자동차보험",
        "product_code": "KB-AUTO-BIZ-WEB",
        "version": "1",
        "effective_from": "2023-09-01",
    },
    "KB다이렉트종합자동차보험": {
        "product_name": "KB다이렉트종합자동차보험",
        "product_type": "종합자동차보험",
        "product_code": "KB-AUTO-TOTAL-DIRECT",
        "version": "1",
        "effective_from": "2023-09-01",
    },
    "KB영업용자동차보험": {
        "product_name": "KB영업용자동차보험",
        "product_type": "영업용자동차보험",
        "product_code": "KB-AUTO-COMMERCIAL",
        "version": "1",
        "effective_from": "2023-08-01",
    },
    "KB다이렉트영업용자동차보험": {
        "product_name": "KB다이렉트영업용자동차보험",
        "product_type": "영업용자동차보험",
        "product_code": "KB-AUTO-COMMERCIAL-DIRECT",
        "version": "1",
        "effective_from": "2023-08-01",
    },
    "KB다이렉트(인터넷)영업용자동차보험": {
        "product_name": "KB다이렉트(인터넷)영업용자동차보험",
        "product_type": "영업용자동차보험",
        "product_code": "KB-AUTO-COMMERCIAL-WEB",
        "version": "1",
        "effective_from": "2023-08-01",
    },
    "KB이륜자동차보험": {
        "product_name": "KB이륜자동차보험",
        "product_type": "이륜자동차보험",
        "product_code": "KB-MOTO",
        "version": "1",
        "effective_from": "2024-02-01",
    },
    "KB다이렉트이륜자동차보험": {
        "product_name": "KB다이렉트이륜자동차보험",
        "product_type": "이륜자동차보험",
        "product_code": "KB-MOTO-DIRECT",
        "version": "1",
        "effective_from": "2024-02-01",
    },
    "KB이륜자동차보험Ⅱ": {
        "product_name": "KB이륜자동차보험Ⅱ",
        "product_type": "이륜자동차보험Ⅱ",
        "product_code": "KB-MOTO-II",
        "version": "1",
        "effective_from": "2024-03-01",
    },
    "KB다이렉트(인터넷)이륜자동차": {
        "product_name": "KB다이렉트(인터넷)이륜자동차",
        "product_type": "이륜자동차보험",
        "product_code": "KB-MOTO-WEB",
        "version": "1",
        "effective_from": "2024-02-01",
    },
    "KB자동차취급업자종합보험": {
        "product_name": "KB자동차취급업자종합보험",
        "product_type": "자동차취급업자종합보험",
        "product_code": "KB-AUTO-HANDLER",
        "version": "1",
        "effective_from": "2023-07-01",
    },
    "KB운전자보험": {
        "product_name": "KB운전자보험",
        "product_type": "운전자보험",
        "product_code": "KB-DRIVER",
        "version": "1",
        "effective_from": "2023-07-01",
    },
    "KB대리운전종합보험": {
        "product_name": "KB대리운전종합보험",
        "product_type": "대리운전종합보험",
        "product_code": "KB-DRIVER-SUBSTITUTE",
        "version": "1",
        "effective_from": "2023-07-01",
    },
    "KB모바일하루자동차보험": {
        "product_name": "KB모바일하루자동차보험",
        "product_type": "단기자동차보험",
        "product_code": "KB-AUTO-DAILY",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "KB플랫폼배달업자자동차보험": {
        "product_name": "KB플랫폼배달업자자동차보험",
        "product_type": "플랫폼배달자동차보험",
        "product_code": "KB-AUTO-DELIVERY-PLATFORM",
        "version": "1",
        "effective_from": "2024-03-01",
    },
    "KB다이렉트(인터넷)대리운전종합보험": {
        "product_name": "KB다이렉트(인터넷)대리운전종합보험",
        "product_type": "대리운전종합보험",
        "product_code": "KB-DRIVER-SUBSTITUTE-WEB",
        "version": "1",
        "effective_from": "2023-07-01",
    },
    "KB단기이륜차운전자보험": {
        "product_name": "KB단기이륜차운전자보험",
        "product_type": "이륜차운전자보험",
        "product_code": "KB-MOTO-DRIVER-SHORT",
        "version": "1",
        "effective_from": "2024-03-01",
    },
    "KB배달·대여라이더이륜자동차보험": {
        "product_name": "KB배달·대여라이더이륜자동차보험",
        "product_type": "배달-대여 이륜차 자동차보험",
        "product_code": "KB-MOTO-LEASE",
        "version": "1",
        "effective_from": "2024-03-01",
    },
    "KB전기자동차종합보험": {
        "product_name": "KB전기자동차종합보험",
        "product_type": "전기자동차종합보험",
        "product_code": "KB-EV-TOTAL",
        "version": "1",
        "effective_from": "2024-04-01",
    },
    "KB음주단속경찰관자동차보험": {
        "product_name": "KB음주단속경찰관자동차보험",
        "product_type": "특수자동차보험",
        "product_code": "KB-AUTO-POLICE",
        "version": "1",
        "effective_from": "2024-04-01",
    },
    "KB모터바이크종합보험": {
        "product_name": "KB모터바이크종합보험",
        "product_type": "모터바이크종합보험",
        "product_code": "KB-MOTORBIKE-TOTAL",
        "version": "1",
        "effective_from": "2024-03-01",
    },
    "개인용자동차보험(공동)": {
        "product_name": "개인용자동차보험(공동)",
        "product_type": "개인용자동차보험(공동)",
        "product_code": "KB-AUTO-PERSONAL-JOINT",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "업무용자동차보험(공동)": {
        "product_name": "업무용자동차보험(공동)",
        "product_type": "업무용자동차보험(공동)",
        "product_code": "KB-AUTO-BIZ-JOINT",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "영업용자동차보험(공동)": {
        "product_name": "영업용자동차보험(공동)",
        "product_type": "영업용자동차보험(공동)",
        "product_code": "KB-AUTO-COMMERCIAL-JOINT",
        "version": "1",
        "effective_from": "2024-01-01",
    },
    "이륜자동차보험(공동)": {
        "product_name": "이륜자동차보험(공동)",
        "product_type": "이륜자동차보험(공동)",
        "product_code": "KB-MOTO-JOINT",
        "version": "1",
        "effective_from": "2024-02-01",
    },
    "KB운전면허교습생자동차보험": {
        "product_name": "KB운전면허교습생자동차보험",
        "product_type": "운전면허교습생자동차보험",
        "product_code": "KB-DRIVER-LEARNER",
        "version": "1",
        "effective_from": "2024-02-01",
    },
}

DEFAULT_PRODUCT_META = {
    "product_name": "미분류",
    "product_type": "미분류",
    "product_code": "UNKNOWN",
    "version": "UNKNOWN",
    "effective_from": "UNKNOWN",
}


def iter_pdf_paths(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.glob("*.pdf") if path.is_file())


def infer_product_metadata(path: Path) -> Dict[str, str]:
    stem = path.stem.replace(" ", "")
    for key, meta in PRODUCT_MAP.items():
        if key.replace(" ", "") in stem:
            return meta
    return {**DEFAULT_PRODUCT_META, "product_name": path.stem}


def load_pdf(path: Path) -> List[Document]:
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    base_meta = infer_product_metadata(path)
    for doc in docs:
        doc.metadata.update(base_meta)
        doc.metadata["source"] = str(path)
        doc.metadata["page"] = doc.metadata.get("page", doc.metadata.get("page_number"))
        doc.metadata["modified_at"] = datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat()
    return docs


def main() -> None:
    # 1) 문서 로드(빌드 때만)
    documents: List[Document] = []
    failed: List[Path] = []

    for pdf_path in iter_pdf_paths(PDF_DIR):
        try:
            documents.extend(load_pdf(pdf_path))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] {pdf_path.name} 로딩 실패: {exc}")
            failed.append(pdf_path)

    if failed:
        print(f"[INFO] 실패한 파일 수: {len(failed)}")

    if not documents:
        print("[INFO] 처리할 문서가 없습니다.")
        return

    # 2) 청크 적게(짧을 수록 프롬프트 모두 빠르다고 함)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    splits = splitter.split_documents(documents)

    for idx, doc in enumerate(splits):
        product_code = doc.metadata.get("product_code", "UNKNOWN")
        doc.metadata["chunk_index"] = idx
        doc.metadata["chunk_id"] = f"{product_code}-{idx:05d}"

    # 3) 임베딩
    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="embedding-query",
    )

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    print(f"[DONE] 인덱싱 완료. 총 문서 수: {len(documents)}, 청크 수: {len(splits)}")


if __name__ == "__main__":
    main()
