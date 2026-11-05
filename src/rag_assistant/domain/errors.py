"""Errores del dominio. Los adaptadores de entrada los traducen a HTTP o a códigos de salida."""

from pathlib import Path


class EmbeddingError(Exception):
    """El proveedor de embeddings falló o dio vectores inválidos (RF-04): detiene la ingesta."""


class LLMError(Exception):
    """El modelo de lenguaje no respondió o respondió algo inválido (RF-02, RF-04)."""


class EmptyDocumentError(ValueError):
    """El documento no tiene texto tras la limpieza; la ingesta lo omite e informa."""


class UnsupportedFormatError(Exception):
    """Ningún cargador acepta el archivo (RF-01: HTTP 415 en la API, salida ≠ 0 en la CLI)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Formato no soportado: {path.name} ({path.suffix or 'sin extensión'}).")
