"""
main.py
───────
실행 진입점입니다.
각 기능의 질문/코드를 여기서 수정하고 python main.py 로 실행하세요.
"""

from embedding import build_vectorstore
from chains import run_error_report, run_checklist
from agent import run_bug_agent


# ══════════════════════════════════════════════
# 공통 준비
# ══════════════════════════════════════════════

print("=" * 55)
print(" LangChain 1.x 에이전트 시스템")
print("=" * 55)

docs, vectorstore = build_vectorstore("dataset.csv")


# ══════════════════════════════════════════════
# 기능 1: 에러 분석 리포트
# ══════════════════════════════════════════════

print("\n[기능 1] 에러 분석 리포트")
print("-" * 40)

result1 = run_error_report(
    vectorstore=vectorstore,
    error_message="Tool Choice 에러",   # ← 원하는 에러 메시지로 변경
)
print(result1)


# ══════════════════════════════════════════════
# 기능 2: 통합 체크리스트 (RAG)
# ══════════════════════════════════════════════

print("\n[기능 2] 통합 체크리스트 (RAG)")
print("-" * 40)

result2 = run_checklist(
    vectorstore=vectorstore,
    docs=docs,
    technology="PostgresSaver",     # ← 원하는 기술명으로 변경
    core_ids=[1, 10, 11],           # ← 핵심 이슈 ID 목록
)
print(result2)


# ══════════════════════════════════════════════
# 기능 3: 코드 버그 감지 에이전트
# ══════════════════════════════════════════════

print("\n[기능 3] 코드 버그 감지 에이전트")
print("-" * 40)

# ↓ 분석할 코드를 여기에 붙여넣으세요
my_code = """
from langchain_groq import ChatGroq
import psycopg2

conn = psycopg2.connect("postgresql://user:pass@localhost/db")

def get_user(user_id: int):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()
    # cursor.close() 빠짐!

llm = ChatGroq(model="llama-3.1-8b-instant")

while True:
    response = llm.invoke("사용자 정보 알려줘")
    print(response)
    # 종료 조건 없음!
"""

result3 = run_bug_agent(
    vectorstore=vectorstore,
    code=my_code,
)
print(result3)