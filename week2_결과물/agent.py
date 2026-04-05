"""
agent.py
────────
기능 3: 코드 버그 감지 Agent (수동 Tool Loop)

💡 AgentExecutor 없이 직접 구현하는 이유:
  LangChain 1.x 에서 AgentExecutor / create_tool_calling_agent 가 제거되었습니다.
  대신 아래처럼 메시지 루프를 직접 돌리는 방식이 표준입니다.

  ┌─────────────────────────────────────┐
  │  1. LLM 호출                        │
  │  2. tool_calls 있음?                │
  │     Yes → 도구 실행                 │
  │           ToolMessage 메시지에 추가 │
  │           → 1로 돌아감              │
  │     No  → 최종 답변 반환            │
  └─────────────────────────────────────┘
"""

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from config import llm
from tools import make_tools


SYSTEM_PROMPT = """당신은 LangChain/LangGraph 코드 리뷰 전문가입니다.
코드를 분석하고 도구로 이슈 DB를 검색하여 버그를 찾아 수정 코드를 제안하세요.

출력 형식:
## 🐛 버그 분석 리포트
### 발견된 잠재적 버그
### 관련 실제 이슈 (검색 결과 기반)
### 🔧 수정된 코드
```python
(수정 코드)
```
### 핵심 변경 사항"""


def run_bug_agent(vectorstore: Chroma, code: str, max_steps: int = 5) -> str:
    """
    사용자 코드를 분석하여 잠재적 버그를 찾고 수정 코드를 제안합니다.

    Args:
        vectorstore : 검색에 사용할 ChromaDB 인스턴스
        code        : 분석할 파이썬 코드 문자열
        max_steps   : tool loop 최대 반복 횟수 (기본값 5)

    사용법:
        result = run_bug_agent(vectorstore, my_code)
    """
    tools, tool_map = make_tools(vectorstore)
    llm_with_tools = llm.bind_tools(tools)

    # 초기 메시지 구성
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"아래 코드의 잠재적 버그를 찾아 수정해주세요:\n```python\n{code}\n```"),
    ]

    # Tool Loop
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # tool_calls 없음 → 최종 답변
        if not response.tool_calls:
            return response.content

        # tool_calls 있음 → 도구 실행 후 결과 추가
        print(f"  🔧 [{step + 1}단계] 도구 호출 중...")
        for tc in response.tool_calls:
            print(f"       └─ {tc['name']}({tc['args']})")
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                )
            )

    return "최대 반복 횟수에 도달했습니다."
