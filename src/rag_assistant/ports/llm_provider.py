"""Puerto del modelo de lenguaje: genera la respuesta a partir del prompt (RF-02, RF-04)."""

from typing import Protocol

from rag_assistant.domain.completion import Completion


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> Completion:
        """Responde al mensaje `user` siguiendo la instrucción `system`. Lanza `LLMError`."""
        ...
