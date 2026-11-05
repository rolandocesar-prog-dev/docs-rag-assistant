"""Fragmento de un documento: la unidad que se indexa, se recupera y se cita (RF-01, RF-03)."""

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Chunk:
    document_id: UUID
    position: int
    content: str
    section: str | None = None
    page: int | None = None
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("La posición no puede ser negativa.")
        if not self.content.strip():
            raise ValueError("El fragmento no tiene texto.")
        if self.section is not None and not self.section.strip():
            raise ValueError("La sección, si se indica, no puede estar vacía.")
        if self.page is not None and self.page < 1:
            raise ValueError("La página debe ser mayor o igual a 1.")
