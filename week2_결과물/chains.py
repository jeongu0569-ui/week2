"""
chains.py
─────────
기능 1, 2의 LCEL Chain을 정의합니다.

💡 Chain이란?
  여러 작업을 | 기호(파이프)로 연결한 파이프라인.
  입력 → 검색 → 프롬프트 조합 → LLM → 출력 순서로 자동 실행됩니다.

💡 RAG(Retrieval-Augmented Generation)란?
  검색(Retrieval) + AI 생성(Generation)의 결합.
  LLM이 모르는 내용을 DB에서 찾아 참고하며 답변합니다.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm
from embedding import get_retriever


# ──────────────────────────────────────────────
# 기능 1: 에러 분석 리포트 Chain
# ──────────────────────────────────────────────

def run_error_report(vectorstore: Chroma, error_message: str) -> str:
    """
    특정 에러 메시지로 유사 이슈를 검색하고
    원인 & 해결책을 리포트로 요약합니다.

    사용법:
        result = run_error_report(vectorstore, "Tool Choice 에러")
    """
    retriever = get_retriever(vectorstore, k=6)

    def format_docs(docs: list) -> str:
        parts = []
        for d in docs:
            label = "📌 이슈" if d.metadata["type"] == "issue" else "💬 댓글"
            parts.append(f"{label} [ID:{d.metadata['id']}]\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_template(
        """당신은 LangChain/LangGraph 기술 지원 전문가입니다.

아래는 관련 이슈와 댓글입니다:
{context}

에러 메시지: "{error_message}"

다음 형식으로 리포트를 작성하세요.

## 🔍 에러 분석 리포트: {error_message}

### 1. 에러 개요
### 2. 근본 원인
### 3. 최종 해결책 (코드 예시 포함)
### 4. 예방 방법
"""
    )

    chain = (
        {
            "context":       lambda x: format_docs(retriever.invoke(x["error_message"])),
            "error_message": lambda x: x["error_message"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"error_message": error_message})


# ──────────────────────────────────────────────
# 기능 2: 통합 체크리스트 RAG
# ──────────────────────────────────────────────

def run_checklist(
    vectorstore: Chroma,
    docs: list[Document],
    technology: str,
    core_ids: list[int],
) -> str:
    """
    핵심 이슈(ID 직접 지정) + 유사 이슈(벡터 검색)를 조합해
    기술 도입 시 '통합 체크리스트'를 생성합니다.

    사용법:
        result = run_checklist(vectorstore, docs, "PostgresSaver", [1, 10, 11])
    """
    # ① 핵심 이슈 직접 조회 (ID 기반 필터링)
    core_docs = [d for d in docs if d.metadata["id"] in core_ids]
    core_text = "\n\n".join(
        f"[ID:{d.metadata['id']}] {d.page_content}" for d in core_docs
    )

    # ② 연관 이슈 벡터 검색 (의미 유사 문서 추가)
    retriever = get_retriever(vectorstore, k=8)
    related_docs = retriever.invoke(f"{technology} 데이터베이스 연결 설정 문제")
    related_text = "\n\n".join(
        f"[ID:{d.metadata['id']}] {d.page_content}" for d in related_docs
    )

    prompt = ChatPromptTemplate.from_template(
        """당신은 LangGraph/LangChain DB 통합 아키텍처 전문가입니다.

[핵심 이슈]
{core_issues}

[연관 이슈]
{related_context}

"{technology}" 기술을 새 프로젝트에 도입할 때 필요한 체크리스트를 작성하세요.

## ✅ {technology} 통합 체크리스트

### 🔴 필수 확인 사항 (도입 전)
### 🟡 설정 체크리스트
### 🟢 권장 모범 사례
### ⚠️ 자주 발생하는 에러 & 해결책
"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "technology":      technology,
        "core_issues":     core_text,
        "related_context": related_text,
    })