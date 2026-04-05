"""
config.py
─────────
LLM & 임베딩 모델 초기화
모든 파일에서 이 모듈에서 llm, embeddings 를 가져다 씁니다.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# Ollama 임베딩 (로컬 실행)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe"
)
