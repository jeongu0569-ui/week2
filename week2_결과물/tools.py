"""
tools.py
────────
에이전트(기능 3)가 사용할 도구(Tool) 모음입니다.

💡 Tool이란?
  LLM이 스스로 호출할 수 있는 함수.
  @tool 데코레이터를 붙이면 LangChain이 함수 설명을 읽고
  언제 이 도구를 써야 할지 판단합니다.
"""

from langchain_core.tools import tool
from langchain_chroma import Chroma


def make_tools(vectorstore: Chroma) -> list:
    """
    vectorstore를 주입받아 도구 리스트를 생성합니다.

    Returns:
        tools    : [search_similar_issues, search_solution_comments]
        tool_map : { "도구이름": 도구함수 }  ← tool loop에서 실행할 때 사용
    """

    @tool
    def search_similar_issues(query: str) -> str:
        """코드나 에러와 관련된 유사 이슈를 이슈 DB에서 검색합니다."""
        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query)
        if not docs:
            return "관련 이슈를 찾지 못했습니다."
        return "\n\n---\n\n".join(
            f"[{'이슈' if d.metadata['type'] == 'issue' else '해결댓글'} ID:{d.metadata['id']}]\n{d.page_content}"
            for d in docs
        )

    @tool
    def search_solution_comments(keyword: str) -> str:
        """특정 키워드로 해결책이 담긴 댓글을 검색합니다."""
        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(f"해결 방법 {keyword}")
        comments = [d for d in docs if d.metadata["type"] == "comment"]
        if not comments:
            return "관련 해결 댓글을 찾지 못했습니다."
        return "\n\n".join(
            f"💬 [ID:{d.metadata['id']}]\n{d.page_content}" for d in comments
        )

    tools = [search_similar_issues, search_solution_comments]
    tool_map = {t.name: t for t in tools}

    return tools, tool_map
