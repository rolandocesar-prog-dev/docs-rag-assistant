"""Reglas de Markdown compartidas por el cargador y el divisor de fragmentos."""

import re

_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def track_fence(line: str, fence: str | None) -> tuple[str | None, bool]:
    """Estado del bloque de código (```/~~~) después de `line`, y si `line` es una valla.

    `fence` es el marcador del bloque abierto, o `None` fuera de un bloque de código.
    """
    match = _FENCE.match(line)
    if match is None:
        return fence, False
    marker = match.group(1)
    if fence is None:
        return marker[0] * 3, True
    return (None if marker.startswith(fence) else fence), True
