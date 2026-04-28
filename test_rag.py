import os
import sys

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from embedding_config import create_embedding_model
from query_router import QueryRouter
from structured_store import StructuredStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_KLASORU = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
STRUCTURED_DB_PATH = os.path.join(BASE_DIR, "structured.db")
COLLECTION_NAME = "ktun_rag"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

SYSTEM_PROMPT = (
    "Sen KTUN Bilgisayar Muhendisligi bolumu asistansin.\n"
    "Sadece verilen baglamdaki bilgiyle cevap ver.\n"
    "Baglamda yoksa: 'Bu bilgiye sahip degilim.' yaz.\n"
    "Tahmin veya baglam disi bilgi kullanma.\n"
)

def build_prompt(question, docs):
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Bilinmiyor")
        context_blocks.append(f"[Belge {i}] Kaynak: {source}\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n"
        f"BAGLAM:\n{context_text}\n\n"
        f"SORU: {question}\n"
        "CEVAP:"
    )


def test_database():
    print("--- RAG TEST (TEK CEVAP MODU) ---")

    if not os.path.exists(VECTOR_DB_KLASORU):
        print(f"HATA: '{VECTOR_DB_KLASORU}' klasoru bulunamadi.")
        return

    print("Model + veritabani yukleniyor...")
    embedding_model = create_embedding_model()
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.1)

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_KLASORU,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    structured_store = StructuredStore(STRUCTURED_DB_PATH)
    structured_store.ensure_ready(DATA_DIR)
    query_router = QueryRouter(structured_store)

    print("Hazir. Cikmak icin 'q' yaz.")
    print("Not: debug ac/kapat icin '/debug on' veya '/debug off' yazabilirsin.")

    debug_mode = False

    while True:
        soru = input("\nSoru: ").strip()
        if not soru:
            continue
        if soru.lower() == "q":
            break

        if soru.lower() == "/debug on":
            debug_mode = True
            print("Debug modu acildi.")
            continue
        if soru.lower() == "/debug off":
            debug_mode = False
            print("Debug modu kapatildi.")
            continue

        structured_result = query_router.try_answer(soru)
        if structured_result:
            print(f"\nCevap: {structured_result['answer']}")
            if debug_mode:
                print(f"[DEBUG] Route: {structured_result.get('route')}")
                print(f"[DEBUG] Sources: {structured_result.get('sources', [])}")
            continue

        docs = vectorstore.max_marginal_relevance_search(
            soru,
            k=6,
            fetch_k=24,
        )

        if not docs:
            print("Cevap: Bu bilgiye sahip degilim.")
            continue

        if debug_mode:
            print(f"\n[DEBUG] {len(docs)} dokuman bulundu:")
            for i, doc in enumerate(docs, start=1):
                kaynak = doc.metadata.get("source", "Bilinmiyor")
                icerik_ozeti = doc.page_content.replace("\n", " ")[:220]
                print(f"- {i}. {kaynak}")
                print(f"  {icerik_ozeti}...")

        prompt = build_prompt(soru, docs)
        llm_response = llm.invoke(prompt)
        answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        print(f"\nCevap: {answer}")


if __name__ == "__main__":
    try:
        test_database()
    except KeyboardInterrupt:
        sys.exit(0)
