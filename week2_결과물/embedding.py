"""
embedding.py
────────────
CSV 데이터셋을 읽어 ChromaDB 벡터 DB를 구축합니다.

💡 CSVLoader란?
  LangChain이 제공하는 CSV 전용 로더.
  각 행을 자동으로 Document 객체로 변환해줍니다.
  page_content_column : LLM 검색에 쓸 메인 텍스트 컬럼
  metadata_columns    : 필터링/참조용으로 저장할 컬럼들

💡 벡터 DB란?
  텍스트를 숫자(벡터)로 변환해 저장하는 DB.
  "의미가 비슷한" 문서를 빠르게 검색할 수 있습니다.
  예) "연결 오류" 검색 → "커넥션 누수" 이슈가 나옴
"""

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import embeddings


def build_vectorstore(csv_path: str = "dataset.csv") -> tuple[list[Document], Chroma]:
    """
    CSVLoader로 CSV를 읽어 ChromaDB에 저장합니다.

    💡 page_content_column은 문자열 하나만 받습니다.
       title + content 두 컬럼을 합치려면 CSVLoader를 두 번 돌린 뒤
       page_content를 직접 이어붙이는 방식을 사용합니다.

    Returns:
        docs        : Document 리스트 (chains.py에서 ID 조회 시 사용)
        vectorstore : ChromaDB 인스턴스
    """
    # content 기준으로 로드 (메인 텍스트)
    loader = CSVLoader(
        file_path=csv_path,
        encoding="utf-8",
        content_columns="content",
        metadata_columns=["id", "type", "title", "author"],
    )
    docs = loader.load()

    # title을 page_content 앞에 붙여서 검색 품질 향상
    for doc in docs:
        title = doc.metadata.get("title")
        doc.page_content = f"제목: {title}\n내용: {doc.page_content}"
        # id를 int로 변환 (CSVLoader는 모든 metadata를 str로 읽음)
        doc.metadata["id"] = int(doc.metadata["id"])

    print(f"📂 데이터 로드: 총 {len(docs)}개 문서")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="issues",
        persist_directory="./chroma_db",
    )
    print(f"✅ 벡터 DB 완료: {len(docs)}개 문서 인덱싱\n")
    return docs, vectorstore



def get_retriever(vectorstore: Chroma, k: int = 6):
    """벡터 유사도 검색 retriever를 반환합니다."""
    return vectorstore.as_retriever(search_kwargs={"k": k})