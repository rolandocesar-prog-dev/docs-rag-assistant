// Chat de la interfaz web (RF-05, design.md). Sin framework y sin innerHTML: todo el texto de la
// API se inserta con textContent, para que una respuesta no pueda inyectar código en la página.
"use strict";

const MAX_CHARS = 1000;
const CITATION = /\[(\d+)\]/g;

const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const counter = document.getElementById("counter");
const conversation = document.getElementById("conversation");
const emptyHint = document.getElementById("empty");

let askCount = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function append(node) {
  emptyHint?.remove();
  conversation.appendChild(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

function showLoading() {
  const box = element("div", "message loading");
  box.setAttribute("role", "status");
  const dots = element("span", "dots");
  dots.setAttribute("aria-hidden", "true");
  dots.append(element("span"), element("span"), element("span"));
  box.append(dots, element("span", "", "Buscando en la documentación…"));
  return append(box);
}

function sourceId(answerId, citationId) {
  return `fuente-${answerId}-${citationId}`;
}

function openSource(id) {
  const card = document.getElementById(id);
  if (!card) return;
  card.open = true;
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.classList.add("highlight");
  setTimeout(() => card.classList.remove("highlight"), 1200);
}

// Convierte el texto en nodos: las marcas [n] con fuente pasan a ser botones hacia su tarjeta.
function answerText(text, answerId, citedIds) {
  const fragment = document.createDocumentFragment();
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const id = Number(match[1]);
    if (!citedIds.has(id)) continue;
    fragment.append(text.slice(last, match.index));
    const button = element("button", "cite", `[${id}]`);
    button.type = "button";
    button.setAttribute("aria-label", `Ver la fuente ${id}`);
    button.addEventListener("click", () => openSource(sourceId(answerId, id)));
    fragment.append(button);
    last = match.index + match[0].length;
  }
  fragment.append(text.slice(last));
  return fragment;
}

function sourceCard(citation, answerId) {
  const card = element("details", "source");
  card.id = sourceId(answerId, citation.id);
  const location = [citation.document, citation.section].filter(Boolean).join(" › ");
  const page = citation.page ? ` (pág. ${citation.page})` : "";
  card.append(element("summary", "", `[${citation.id}] ${location}${page}`));
  card.append(element("p", "", citation.excerpt));
  card.append(element("p", "path", citation.source_path));
  return card;
}

function showAnswer(data) {
  const answerId = ++askCount;
  if (!data.found) {
    append(element("div", "message notice", data.answer));
    return;
  }
  const box = element("div", "message answer");
  const citedIds = new Set(data.citations.map((c) => c.id));
  box.append(answerText(data.answer, answerId, citedIds));
  const sources = element("div", "sources");
  sources.append(element("h2", "", "Fuentes"));
  for (const citation of data.citations) sources.append(sourceCard(citation, answerId));
  box.append(sources);
  append(box);
}

function errorMessage(status, body) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return "La pregunta no es válida: no puede estar vacía ni superar 1 000 caracteres.";
  return `El servidor respondió ${status}.`;
}

async function ask(question) {
  append(element("div", "message user", question));
  const loading = showLoading();
  sendButton.disabled = true;
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const body = await response.json().catch(() => null);
    loading.remove();
    if (response.ok) showAnswer(body);
    else append(element("div", "message error", errorMessage(response.status, body)));
  } catch {
    loading.remove();
    append(element("div", "message error", "No se pudo conectar con el servidor."));
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

function updateCounter() {
  counter.textContent = `${input.value.length} / ${MAX_CHARS}`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendButton.disabled) return;
  input.value = "";
  updateCounter();
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", updateCounter);
