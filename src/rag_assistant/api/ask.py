"""Endpoint `POST /ask`: pregunta en lenguaje natural → respuesta con citas (RF-02, RF-03)."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from rag_assistant.api.dependencies import get_ask_service
from rag_assistant.application.ask import MAX_QUESTION_CHARS, AskService
from rag_assistant.domain.answer import Answer

router = APIRouter(tags=["ask"])
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class CitationResponse(BaseModel):
    id: int
    document: str
    section: str | None
    page: int | None
    source_path: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    found: bool

    @classmethod
    def from_answer(cls, answer: Answer) -> "AskResponse":
        return cls(
            answer=answer.text,
            citations=[
                CitationResponse(
                    id=c.id,
                    document=c.document,
                    section=c.section,
                    page=c.page,
                    source_path=c.source_path,
                    excerpt=c.excerpt,
                )
                for c in answer.citations
            ],
            found=answer.found,
        )


@router.post("/ask")
def ask(
    request: AskRequest, service: Annotated[AskService, Depends(get_ask_service)]
) -> AskResponse:
    """Responde solo con la documentación ingestada; `found` es falso si no hay información."""
    answer = service.ask(request.question)
    # Sin el texto de la pregunta: puede traer datos personales (design.md, RNF-04).
    logger.info(
        "Pregunta respondida",
        extra={
            "found": answer.found,
            "citations": len(answer.citations),
            "question_chars": len(request.question),
        },
    )
    return AskResponse.from_answer(answer)
