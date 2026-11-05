"""Errores del dominio. Los adaptadores de entrada los traducen a HTTP o a códigos de salida."""

from pathlib import Path


class UnsupportedFormatError(Exception):
    """Ningún cargador acepta el archivo (RF-01: HTTP 415 en la API, salida ≠ 0 en la CLI)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Formato no soportado: {path.name} ({path.suffix or 'sin extensión'}).")
