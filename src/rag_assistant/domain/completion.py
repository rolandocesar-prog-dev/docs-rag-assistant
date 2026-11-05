"""Respuesta de un modelo de lenguaje, independiente del proveedor (RF-04)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    # La respuesta llegó al tope de tokens: suele ser un bucle o quedar cortada.
    truncated: bool = False
