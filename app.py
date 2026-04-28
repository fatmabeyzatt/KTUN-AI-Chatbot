__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
try:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import io
import contextlib
import re

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from chromadb.config import Settings
from embedding_config import create_embedding_model
from query_router import QueryRouter
from structured_store import StructuredStore, normalize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
STRUCTURED_DB_PATH = os.path.join(BASE_DIR, "structured.db")
COLLECTION_NAME = "ktun_rag"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def env_bool(name, default=False):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


ANSWER_ONLY = env_bool("ANSWER_ONLY", False)
SHOW_SOURCES = env_bool("SHOW_SOURCES", False)
STRUCTURED_ONLY = env_bool("STRUCTURED_ONLY", False)
AUTO_INGEST = env_bool("AUTO_INGEST", True)
AUTO_INGEST_VERBOSE = env_bool("AUTO_INGEST_VERBOSE", False)
CONVERSATION_MEMORY = env_bool("CONVERSATION_MEMORY", True)


def run_auto_ingest():
    if not AUTO_INGEST:
        return
    try:
        from ingest import create_pipeline

        if AUTO_INGEST_VERBOSE:
            create_pipeline()
            return

        # Keep terminal output clean in answer-only mode.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            create_pipeline()
    except Exception as exc:
        # In non-verbose answer-only mode we stay silent and continue with existing DB.
        if AUTO_INGEST_VERBOSE or not ANSWER_ONLY:
            print(f"Otomatik Chroma guncellemesi basarisiz: {exc}")


run_auto_ingest()


try:
    embedding_model = create_embedding_model()
except Exception as exc:
    raise SystemExit(f"Baslangic hatasi (embedding): {exc}")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    client_settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
    ),
)

llm = ChatOllama(
    model="qwen3:4b",
    base_url=OLLAMA_URL,
    temperature=0.1,
)

structured_store = StructuredStore(STRUCTURED_DB_PATH)
structured_store.ensure_ready(DATA_DIR)
query_router = QueryRouter(structured_store)

SYSTEM_PROMPT = (
    "Sen KTUN Bilgisayar Muhendisligi bolumu asistansin.\n"
    "Sadece verilen baglam metnindeki bilgiyle cevap ver.\n"
    "Baglamda bilgi yoksa: 'Bu bilgiye sahip degilim.' yaz.\n"
    "Kendi bilgisini, tahmini veya baska universite bilgisini kullanma.\n"
    "Cevabi kisa ver; aciklama, kaynak, etiket veya on ek yazma.\n"
    "'Belge', 'Kaynak', 'Cevap:' gibi ifadeleri kullanma.\n"
)

LINK_HINTS = {"link", "pdf", "dosya", "sunum", "indir"}
ANNOUNCEMENT_HINTS = {
    "duyuru",
    "duyurular",
    "basvuru",
    "basvurular",
    "etkinlik",
    "toplanti",
    "programi",
    "staj",
    "teknofest",
    "hangar",
}
FOLLOWUP_CUES = {
    "bu",
    "bunu",
    "bunun",
    "buna",
    "bunda",
    "bunu da",
    "onun",
    "onu",
    "ona",
    "oradaki",
    "buradaki",
    "az onceki",
    "demin",
    "onceki",
    "bahsettigin",
    "linki",
    "tarihi",
    "detayi",
}
FOLLOWUP_GENERIC = {
    "bu",
    "bunu",
    "bunun",
    "buna",
    "bunda",
    "onun",
    "onu",
    "ona",
    "oradaki",
    "buradaki",
    "link",
    "linki",
    "tarih",
    "tarihi",
    "detay",
    "detayi",
    "ver",
    "verir",
    "misin",
}
FOLLOWUP_GENERIC_PREFIXES = (
    "bu",
    "bun",
    "on",
    "link",
    "tarih",
    "detay",
    "ver",
    "eris",
    "ula",
    "var",
    "mi",
    "mı",
    "mu",
    "mü",
)
QUERY_STOPWORDS = {
    "ve",
    "ile",
    "icin",
    "mi",
    "mı",
    "mu",
    "mü",
    "bir",
    "bu",
    "su",
    "o",
    "da",
    "de",
}
URL_PATTERN = re.compile(r"https?://[^\s>\"]+", re.IGNORECASE)


def has_link_intent(query):
    query_l = normalize_text(query)
    return any(token in query_l for token in LINK_HINTS)


def has_announcement_intent(query):
    query_l = normalize_text(query)
    return any(token in query_l for token in ANNOUNCEMENT_HINTS)


def score_link_doc(query, doc):
    query_l = normalize_text(query)
    content_l = normalize_text(doc.page_content)
    source_l = normalize_text(str(doc.metadata.get("source", "")))
    score = 0.0

    if "http" in content_l or "http" in source_l:
        score += 2.0
    if ".pdf" in content_l or ".pdf" in source_l:
        score += 3.0
    if "url:" in content_l or "link" in content_l:
        score += 1.0

    terms = [term for term in query_l.split() if len(term) >= 4]
    overlap = sum(1 for term in terms if term in content_l or term in source_l)
    score += overlap * 0.3
    return score


def score_announcement_doc(query, doc):
    query_l = normalize_text(query)
    query_terms = [term for term in query_l.split() if len(term) >= 3]
    query_roots = {term[:5] for term in query_terms if len(term) >= 4}

    content_l = normalize_text(doc.page_content)
    content_tokens = content_l.split()
    content_roots = {term[:5] for term in content_tokens if len(term) >= 4}
    source_l = normalize_text(
        " ".join(
            [
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("source_path", "")),
                str(doc.metadata.get("csv_file", "")),
            ]
        )
    )
    title_l = ""
    for line in (doc.page_content or "").splitlines():
        if line.lower().startswith("baslik:"):
            title_l = normalize_text(line.split(":", 1)[1])
            break
    title_roots = {term[:5] for term in title_l.split() if len(term) >= 4}

    score = 0.0
    if "guncel duyurular" in content_l:
        score += 3.0
    if "guncel_duyurular.csv" in source_l or "guncel duyurular" in source_l:
        score += 3.0
    if "tarih" in content_l:
        score += 0.5

    title_overlap = sum(1 for root in query_roots if root in title_roots)
    content_overlap = sum(1 for root in query_roots if root in content_roots)
    raw_overlap = sum(1 for term in query_terms if term in content_l or term in source_l)

    score += title_overlap * 2.2
    score += max(0, content_overlap - title_overlap) * 1.0
    score += raw_overlap * 0.6
    return score


def tokenize_query(text):
    return [t for t in normalize_text(text).split() if t and t not in QUERY_STOPWORDS]


def should_use_context(query, memory_state):
    if not CONVERSATION_MEMORY or not memory_state.get("last_question"):
        return False

    q_norm = normalize_text(query)
    has_followup_cue = any(cue in q_norm for cue in FOLLOWUP_CUES)
    if not has_followup_cue:
        return False
    if re.search(r"\b20\d{2}\b", query):
        return False

    query_tokens = tokenize_query(query)
    prev_tokens = set(memory_state.get("topic_tokens", []))
    if not prev_tokens:
        return True

    def is_generic_token(token):
        if token in FOLLOWUP_GENERIC:
            return True
        return any(token.startswith(prefix) for prefix in FOLLOWUP_GENERIC_PREFIXES)

    core_tokens = [t for t in query_tokens if not is_generic_token(t)]
    if not core_tokens:
        return True

    overlap = sum(1 for token in core_tokens if token in prev_tokens)
    if overlap > 0:
        return True

    # If user gave a clearly new topic with multiple concrete tokens, do not force context.
    if len(core_tokens) >= 2:
        return False
    return True


def build_effective_query(query, memory_state):
    if not should_use_context(query, memory_state):
        return query, False

    anchor = memory_state.get("last_question", "").strip()
    if not anchor:
        return query, False
    return f"Baglam konu: {anchor}\nTakip sorusu: {query}", True


def extract_urls_from_text(text):
    urls = []
    for match in URL_PATTERN.findall(text or ""):
        cleaned = match.rstrip(".,;]}>\"'")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def try_followup_link_from_memory(query, memory_state, context_used):
    if not context_used or not has_link_intent(query):
        return None

    doc_candidates = []
    target_tokens = set(
        tokenize_query(
            f"{memory_state.get('last_question', '')} {memory_state.get('last_answer', '')}"
        )
    )

    for idx, doc in enumerate(memory_state.get("last_docs", [])):
        doc_text = normalize_text(doc.page_content)
        overlap = sum(1 for token in target_tokens if token in doc_text)
        order_bonus = max(0, 6 - idx) * 0.2
        score = overlap + order_bonus
        urls = extract_urls_from_text(doc.page_content)
        source = str(doc.metadata.get("source", "")).strip()
        urls.extend(extract_urls_from_text(source))
        for url in urls:
            doc_candidates.append((score, idx, url))

    dedup = []
    for src in memory_state.get("last_sources", []):
        for url in extract_urls_from_text(str(src)):
            doc_candidates.append((0.0, 999, url))

    for _, _, url in doc_candidates:
        if url not in dedup:
            dedup.append(url)

    if not dedup:
        return None

    ranking = {url: (-1.0, 999) for url in dedup}
    for score, idx, url in doc_candidates:
        best_score, best_idx = ranking[url]
        if score > best_score or (score == best_score and idx < best_idx):
            ranking[url] = (score, idx)

    priority = sorted(
        dedup,
        key=lambda u: (
            -ranking[u][0],
            ranking[u][1],
            "duyurudetay" not in normalize_text(u),
            "ktun.edu.tr" not in normalize_text(u),
            len(u),
        ),
    )
    return priority[0]


def get_relevant_docs(query, effective_query=None):
    search_query = effective_query or query
    if has_link_intent(query):
        candidate_docs = vectorstore.similarity_search(search_query, k=12)
        ranked = sorted(candidate_docs, key=lambda doc: score_link_doc(query, doc), reverse=True)
        return ranked[:6]
    if has_announcement_intent(query):
        candidate_docs = vectorstore.similarity_search(search_query, k=24)
        duyuru_docs = []
        for doc in candidate_docs:
            csv_file = str(doc.metadata.get("csv_file", ""))
            src = str(doc.metadata.get("source_path", "")) + " " + str(doc.metadata.get("source", ""))
            if "guncel_duyurular.csv" in csv_file or "guncel_duyurular.csv" in src:
                duyuru_docs.append(doc)
        if duyuru_docs:
            candidate_docs = duyuru_docs
        ranked = sorted(candidate_docs, key=lambda doc: score_announcement_doc(query, doc), reverse=True)
        return ranked[:6]
    return vectorstore.max_marginal_relevance_search(search_query, k=6, fetch_k=24)


def build_prompt(query, docs):
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Bilinmiyor")
        context_blocks.append(f"[Belge {i}] Kaynak: {source}\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"BAGLAM:\n{context_text}\n\n"
        f"SORU: {query}\n"
        "CEVAP:"
    )


def is_suspicious_answer(answer):
    suspicious_words = [
        "bilkent",
        "odtu",
        "bogazici",
        "hacettepe",
        "ankara universitesi",
        "istanbul universitesi",
        "turkiyede",
        "universitelerde",
        "genellikle",
        "ornegin",
        "bazi universite",
    ]
    answer_l = answer.lower()
    return any(word in answer_l for word in suspicious_words)


def emit_answer(answer, sources=None):
    sources = sources or []
    if not SHOW_SOURCES:
        lines = []
        for line in str(answer).splitlines():
            if line.strip().lower().startswith("kaynak url:"):
                continue
            lines.append(line)
        answer = "\n".join(lines).strip()

    print(f"\nCevap: {answer}")

    if SHOW_SOURCES and sources:
        print(f"\nKaynaklar: {sources}")
    sys.stdout.flush()


if __name__ == "__main__":
    print("Asistan calisiyor... (Cikmak icin 'exit' yazin)")
    sys.stdout.flush()

    memory_state = {
        "last_question": "",
        "last_answer": "",
        "last_docs": [],
        "last_sources": [],
        "topic_tokens": [],
    }

    while True:
        try:
            sys.stdout.write("\nSoru: ")
            sys.stdout.flush()
            query = sys.stdin.readline().strip()

            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            print("Cevap üretiliyor...")
            sys.stdout.flush()

            effective_query, context_used = build_effective_query(query, memory_state)
            link_from_memory = try_followup_link_from_memory(query, memory_state, context_used)
            if link_from_memory:
                emit_answer(link_from_memory, [link_from_memory])
                memory_state = {
                    "last_question": query,
                    "last_answer": link_from_memory,
                    "last_docs": memory_state.get("last_docs", []),
                    "last_sources": [link_from_memory],
                    "topic_tokens": tokenize_query(memory_state.get("last_question", "") + " " + query),
                }
                continue

            structured_result = query_router.try_answer(query)
            if structured_result:
                answer_text = structured_result["answer"]
                source_list = structured_result.get("sources", [])
                emit_answer(answer_text, source_list)
                memory_state = {
                    "last_question": query,
                    "last_answer": answer_text,
                    "last_docs": [],
                    "last_sources": source_list,
                    "topic_tokens": tokenize_query(query + " " + answer_text),
                }
                continue

            if STRUCTURED_ONLY:
                emit_answer("Bu bilgiye sahip degilim.")
                continue

            relevant_docs = get_relevant_docs(query, effective_query=effective_query)

            if not relevant_docs:
                emit_answer("Bu bilgiye sahip degilim.")
                continue

            prompt = build_prompt(query, relevant_docs)
            llm_response = llm.invoke(prompt)
            answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

            if is_suspicious_answer(answer):
                emit_answer("Bu bilgiye sahip degilim.")
            else:
                sources = list({doc.metadata.get("source", "Bilinmiyor") for doc in relevant_docs})
                emit_answer(answer, sources)
                memory_state = {
                    "last_question": query,
                    "last_answer": answer,
                    "last_docs": relevant_docs,
                    "last_sources": sources,
                    "topic_tokens": tokenize_query(query + " " + answer),
                }
        except (EOFError, KeyboardInterrupt):
            print("\nCikis yapiliyor...")
            break
        except Exception as exc:
            print(f"\nHata: {exc}")
