"""Limpieza de etiquetas de Hugo (`{{< … >}}`, `{{% … %}}`), según la tabla de `design.md`."""

import re

_COMMENT_BLOCK = re.compile(
    r"\{\{[<%]\s*comment\s*[>%]\}\}.*?\{\{[<%]\s*/\s*comment\s*[>%]\}\}", re.DOTALL
)
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w-]+)(.*?)[>%]\}\}", re.DOTALL)
_NAMED_PARAM = re.compile(r'([\w-]+)="([^"]*)"')
_POSITIONAL_PARAM = re.compile(r'"([^"]*)"')

# Etiquetas que se reemplazan por el primer parámetro no vacío de la lista.
_LABEL_PARAMS: dict[str, tuple[str, ...]] = {
    "glossary_tooltip": ("text", "term_id"),
    "link": ("text",),
    "figure": ("caption", "title"),
    "table": ("caption",),
    "alert": ("title",),
    "tab": ("name",),
}
_HEADINGS = {"whatsnext": "¿Qué sigue?"}


def strip_hugo(text: str) -> str:
    return _SHORTCODE.sub(_render, _COMMENT_BLOCK.sub("", text))


def _render(match: re.Match[str]) -> str:
    closing, name, args = match.groups()
    if closing:
        return ""
    params = dict(_NAMED_PARAM.findall(args))
    if name in _LABEL_PARAMS:
        return next((params[key] for key in _LABEL_PARAMS[name] if params.get(key)), "")
    if name == "feature-state":
        version, state = params.get("for_k8s_version"), params.get("state")
        return f"Estado de la función: Kubernetes {version} [{state}]" if version and state else ""
    if name == "heading":
        key = next(iter(_POSITIONAL_PARAM.findall(args)), "")
        return _HEADINGS.get(key, key)
    return ""
