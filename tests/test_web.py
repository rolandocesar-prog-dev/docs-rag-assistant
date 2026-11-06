"""Página web de chat (RF-05, design.md): se sirve completa y sin dependencias externas."""

import re

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.web import WEB_DIR
from rag_assistant.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_root_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert 'id="ask-form"' in html and 'id="question"' in html
    assert 'maxlength="1000"' in html
    assert 'aria-live="polite"' in html
    assert '<label for="question"' in html


@pytest.mark.parametrize(
    ("path", "content_type"),
    [("/static/app.js", "javascript"), ("/static/styles.css", "text/css")],
)
def test_static_assets_are_served(client: TestClient, path: str, content_type: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_page_only_loads_resources_from_the_same_server() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    references = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert references
    assert all(ref.startswith("/static/") for ref in references)


def test_script_never_inserts_api_text_as_html() -> None:
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    code = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("//"))

    for unsafe in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert unsafe not in code


def test_page_is_not_part_of_the_api_schema(client: TestClient) -> None:
    assert "/" not in client.get("/openapi.json").json()["paths"]
