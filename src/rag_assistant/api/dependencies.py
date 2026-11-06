"""Composición de la API: arma los casos de uso con los adaptadores reales una sola vez."""

from functools import lru_cache

from fastapi import HTTPException, status

from rag_assistant.adapters.database import create_database_engine
from rag_assistant.adapters.embeddings.ollama import OllamaEmbeddings
from rag_assistant.adapters.llm.factory import create_llm
from rag_assistant.adapters.vectorstore.pgvector import PgVectorStore
from rag_assistant.application.ask import AskService
from rag_assistant.config import get_settings


@lru_cache
def get_ask_service() -> AskService:
    settings = get_settings()
    if not settings.database_url:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Falta RAG_DATABASE_URL: defínela en .env (ver .env.example).",
        )
    return AskService(
        OllamaEmbeddings.from_url(
            settings.ollama_url,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        ),
        PgVectorStore(create_database_engine(settings.database_url)),
        create_llm(settings),
        k=settings.retrieval_k,
        threshold=settings.relevance_threshold,
        hybrid=settings.hybrid_search,
    )
