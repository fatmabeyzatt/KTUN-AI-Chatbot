import csv
import hashlib
import json
import os
import shutil
import sys

# SQLite fix for Docker/Linux if needed
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from chromadb.config import Settings
from data_layout import DATA_ROOT, discover_pdf_files, discover_structured_files, ensure_data_directories
from embedding_config import create_embedding_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from structured_store import StructuredStore

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_KLASORU = DATA_ROOT
VECTOR_DB_KLASORU = os.path.join(BASE_DIR, "chroma_db")
STRUCTURED_DB_PATH = os.path.join(BASE_DIR, "structured.db")
COLLECTION_NAME = "ktun_rag"
INDEX_STATE_PATH = os.path.join(VECTOR_DB_KLASORU, ".ingest_state.json")
PDF_PARSE_VERSION = "pdf_parse_v1"

ENCODINGS = ("utf-8", "utf-8-sig", "cp1254", "latin-1")
TURKISH_TRANSLATE = str.maketrans(
    {
        "I": "i",
        "İ": "i",
        "ı": "i",
        "Ş": "s",
        "ş": "s",
        "Ğ": "g",
        "ğ": "g",
        "Ü": "u",
        "ü": "u",
        "Ö": "o",
        "ö": "o",
        "Ç": "c",
        "ç": "c",
    }
)

SOURCE_KEY_SET = {
    "source",
    "url",
    "link",
    "kaynak",
    "kaynaklink",
    "kaynakurl",
}
TITLE_KEYS = ("title", "baslik", "konu", "name", "subject")
CONTENT_KEYS = ("content", "text", "body", "icerik", "description", "metin", "answer")


def normalize_key(text):
    normalized = str(text).translate(TURKISH_TRANSLATE).lower()
    return "".join(ch for ch in normalized if ch.isalnum())


def detect_csv_dialect(file_path, encoding):
    with open(file_path, "r", encoding=encoding, newline="") as csv_file:
        sample = csv_file.read(4096)
    if not sample.strip():
        return ",", []

    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    header_line = sample.splitlines()[0]
    header = [part.strip() for part in header_line.split(delimiter)]
    return delimiter, header


def clear_vector_db():
    if not os.path.exists(VECTOR_DB_KLASORU):
        os.makedirs(VECTOR_DB_KLASORU, exist_ok=True)
        return True

    print("Old vector DB found, cleaning folder content...")
    locked_sqlite = False
    for filename in os.listdir(VECTOR_DB_KLASORU):
        path = os.path.join(VECTOR_DB_KLASORU, filename)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as exc:
            print(f"  WARNING: Could not remove '{path}': {repr(exc)}")
            if os.path.basename(path).lower() == "chroma.sqlite3":
                locked_sqlite = True
    if locked_sqlite:
        print("  ERROR: chroma.sqlite3 is locked by another process.")
        print("  Close any running app/test that uses Chroma, then run ingest.py again.")
        return False
    return True


def detect_csv_source_column(header):
    for column in header or []:
        normalized = normalize_key(column).lstrip("ufeff")
        if normalized in SOURCE_KEY_SET:
            return column
    for column in header or []:
        if normalize_key(column) in SOURCE_KEY_SET:
            return column
    return None


def load_csv_documents(file_path):
    last_error = None
    for encoding in ENCODINGS:
        try:
            delimiter, header = detect_csv_dialect(file_path, encoding)
            source_column = detect_csv_source_column(header)
            documents = []
            with open(file_path, "r", encoding=encoding, newline="") as csv_file:
                reader = csv.DictReader(csv_file, delimiter=delimiter)
                if reader.fieldnames:
                    reader.fieldnames = [
                        (field or "").strip().lstrip("\ufeff") for field in reader.fieldnames
                    ]
                    source_column = detect_csv_source_column(reader.fieldnames) or source_column

                for index, row in enumerate(reader):
                    cleaned = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        cleaned_key = str(key).strip().lstrip("\ufeff")
                        cleaned_value = "" if value is None else str(value).strip()
                        if cleaned_value:
                            cleaned[cleaned_key] = cleaned_value

                    if not cleaned:
                        continue

                    content_lines = []
                    for key, value in cleaned.items():
                        if normalize_key(key) in SOURCE_KEY_SET:
                            continue
                        content_lines.append(f"{key}: {value}")

                    file_tag = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")
                    prefix_lines = [f"Belge_Turu: {file_tag}"]
                    normalized_keys = {normalize_key(k) for k in cleaned.keys()}
                    if {"isimunvan", "adsoyad", "isim"} & normalized_keys:
                        prefix_lines.append("Kayit_Tipi: akademik personel")
                    if {"sayfaadi", "baslik"} & normalized_keys:
                        prefix_lines.append("Kayit_Tipi: bolum bilgisi")

                    merged_lines = prefix_lines + content_lines
                    page_content = "\n".join(merged_lines) or json.dumps(cleaned, ensure_ascii=False)

                    source = file_path
                    if source_column and cleaned.get(source_column):
                        source = cleaned[source_column]

                    documents.append(
                        Document(
                            page_content=page_content,
                            metadata={
                                "source": source,
                                "row": index,
                                "csv_file": os.path.basename(file_path),
                                "source_path": file_path,
                                "source_type": "csv",
                            },
                        )
                    )
            return documents, encoding, source_column, delimiter
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV load failed for '{file_path}': {last_error}")


def read_json_file(file_path):
    last_error = None
    for encoding in ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as json_file:
                payload = json.load(json_file)
            return payload, encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"JSON read failed for '{file_path}': {last_error}")


def extract_json_records(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("items", "data", "records", "documents", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if payload and all(not isinstance(v, (dict, list)) for v in payload.values()):
            return [{"title": str(k), "content": str(v), "source": str(v)} for k, v in payload.items()]
        return [payload]

    return [payload]


def pick_value(record, candidate_keys):
    key_map = {normalize_key(k): k for k in record.keys()}
    for candidate in candidate_keys:
        match = key_map.get(normalize_key(candidate))
        if not match:
            continue
        value = record.get(match)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def json_record_to_document(record, file_path, index):
    title = pick_value(record, TITLE_KEYS)
    content = pick_value(record, CONTENT_KEYS)
    source = pick_value(
        record,
        (
            "source",
            "url",
            "link",
            "kaynak",
            "kaynak link",
            "kaynak_link",
        ),
    )

    if title:
        title = title.replace("_", " ").strip()
        if title.lower().endswith(".pdf"):
            title = title[:-4]

    if content is None:
        lines = []
        for key, value in record.items():
            if value is None:
                continue
            if normalize_key(key) in SOURCE_KEY_SET:
                continue
            lines.append(f"{key}: {value}")
        content = "\n".join(lines).strip() or json.dumps(record, ensure_ascii=False)

    page_content = f"Baslik: {title}\nIcerik: {content}" if title else content
    metadata = {
        "source": source or file_path,
        "json_file": os.path.basename(file_path),
        "record_index": index,
        "source_path": file_path,
        "source_type": "json",
    }
    return Document(page_content=page_content, metadata=metadata)


def load_json_documents(file_path):
    payload, encoding = read_json_file(file_path)
    records = extract_json_records(payload)
    documents = []

    for index, record in enumerate(records):
        if isinstance(record, dict):
            documents.append(json_record_to_document(record, file_path, index))
            continue

        text = str(record).strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "json_file": os.path.basename(file_path),
                    "record_index": index,
                    "source_path": file_path,
                    "source_type": "json",
                },
            )
        )

    return documents, encoding


def load_pdf_documents(file_path):
    if PdfReader is None:
        raise RuntimeError("pypdf kutuphanesi kurulu degil. `pip install pypdf` gerekli.")

    reader = PdfReader(file_path)
    documents = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""

        if not text:
            continue
        page_content = (
            f"Belge_Turu: pdf\n"
            f"Kayit_Tipi: pdf_dokumani\n"
            f"PDF_Adi: {os.path.basename(file_path)}\n"
            f"Sayfa: {page_no}\n"
            f"Icerik:\n{text}"
        )
        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": file_path,
                    "pdf_file": os.path.basename(file_path),
                    "page": page_no,
                    "source_path": file_path,
                    "source_type": "pdf",
                },
            )
        )
    return documents


def source_key(file_path):
    rel = os.path.relpath(file_path, BASE_DIR)
    return rel.replace("\\", "/")


def file_signature(file_path, extra=""):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    if extra:
        hasher.update(extra.encode("utf-8"))
    return hasher.hexdigest()


def make_chunk_id(src_key, index, text):
    material = f"{src_key}:{index}:{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(material).hexdigest()


def load_index_state():
    if not os.path.exists(INDEX_STATE_PATH):
        return {"version": 1, "sources": {}}
    try:
        with open(INDEX_STATE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            return {"version": 1, "sources": {}}
        payload.setdefault("version", 1)
        payload.setdefault("sources", {})
        return payload
    except Exception:
        return {"version": 1, "sources": {}}


def save_index_state(state):
    os.makedirs(VECTOR_DB_KLASORU, exist_ok=True)
    with open(INDEX_STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def delete_ids(vectorstore, ids, batch_size=512):
    for start in range(0, len(ids), batch_size):
        vectorstore.delete(ids=ids[start : start + batch_size])


def collect_source_payloads():
    structured_files = discover_structured_files(DATA_KLASORU)
    pdf_files = discover_pdf_files(DATA_KLASORU)
    csv_files = [path for path in structured_files if path.lower().endswith(".csv")]
    json_files = [path for path in structured_files if path.lower().endswith(".json")]

    print(
        f"Discovered files | csv={len(csv_files)} | json={len(json_files)} | pdf={len(pdf_files)}"
    )
    payloads = {}

    for file_path in csv_files:
        name = os.path.basename(file_path)
        try:
            docs, encoding, source_column, delimiter = load_csv_documents(file_path)
            key = source_key(file_path)
            payloads[key] = {
                "kind": "csv",
                "path": file_path,
                "signature": file_signature(file_path),
                "documents": docs,
            }
            source_info = source_column or "auto(row index)"
            print(
                f"CSV OK -> {name} | docs={len(docs)} | enc={encoding} | delim={repr(delimiter)} | source={source_info}"
            )
        except Exception as exc:
            print(f"CSV ERROR -> {name}: {exc}")

    for file_path in json_files:
        name = os.path.basename(file_path)
        try:
            docs, encoding = load_json_documents(file_path)
            key = source_key(file_path)
            payloads[key] = {
                "kind": "json",
                "path": file_path,
                "signature": file_signature(file_path),
                "documents": docs,
            }
            print(f"JSON OK -> {name} | docs={len(docs)} | enc={encoding}")
        except Exception as exc:
            print(f"JSON ERROR -> {name}: {exc}")

    if pdf_files and PdfReader is None:
        print("PDF WARNING: pypdf kurulu olmadigi icin PDF dosyalari atlandi.")

    for file_path in pdf_files:
        if PdfReader is None:
            break
        name = os.path.basename(file_path)
        try:
            docs = load_pdf_documents(file_path)
            key = source_key(file_path)
            payloads[key] = {
                "kind": "pdf",
                "path": file_path,
                "signature": file_signature(file_path, extra=PDF_PARSE_VERSION),
                "documents": docs,
            }
            print(f"PDF OK -> {name} | docs={len(docs)}")
        except Exception as exc:
            print(f"PDF ERROR -> {name}: {exc}")

    total_docs = sum(len(payload["documents"]) for payload in payloads.values())
    print(f"Total raw document count: {total_docs}")
    return payloads


def create_pipeline():
    print("--- KTUN RAG ingest started (incremental) ---")
    print(f"Data folder: {DATA_KLASORU}")
    print(f"DB folder: {VECTOR_DB_KLASORU}")

    ensure_data_directories()

    payloads = collect_source_payloads()
    if not payloads:
        print("No document loaded. Process stopped.")
        return

    print("Rebuilding structured database...")
    structured_store = StructuredStore(STRUCTURED_DB_PATH)
    stats = structured_store.rebuild_from_data(DATA_KLASORU)
    print(
        "Structured DB ready | "
        f"personnel={stats.get('personnel', 0)} | "
        f"sections={stats.get('sections', 0)} | "
        f"links={stats.get('links', 0)}"
    )

    print("Preparing embedding model...")
    try:
        embedding_model = create_embedding_model()
    except Exception as exc:
        print(f"ERROR: Embedding modeli hazirlanamadi: {exc}")
        return

    client_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    # If state is missing, enforce one full clean so incremental IDs don't stack on unknown old vectors.
    fresh_index = not os.path.exists(INDEX_STATE_PATH)
    if fresh_index and not clear_vector_db():
        print("Process stopped due to locked DB file.")
        return

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_KLASORU,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        client_settings=client_settings,
    )

    old_state = {"version": 1, "sources": {}} if fresh_index else load_index_state()
    old_sources = old_state.get("sources", {})
    new_sources = {}

    current_keys = set(payloads.keys())
    previous_keys = set(old_sources.keys())
    removed_keys = sorted(previous_keys - current_keys)
    changed_or_new = []
    unchanged = []

    for src_key in sorted(current_keys):
        signature = payloads[src_key]["signature"]
        old_signature = old_sources.get(src_key, {}).get("signature")
        if old_signature == signature:
            unchanged.append(src_key)
        else:
            changed_or_new.append(src_key)

    print(
        f"Index diff | new_or_changed={len(changed_or_new)} | removed={len(removed_keys)} | unchanged={len(unchanged)}"
    )

    for src_key in removed_keys:
        previous_ids = old_sources.get(src_key, {}).get("ids", [])
        if previous_ids:
            delete_ids(vectorstore, previous_ids)

    for src_key in changed_or_new:
        payload = payloads[src_key]
        previous_ids = old_sources.get(src_key, {}).get("ids", [])
        if previous_ids:
            delete_ids(vectorstore, previous_ids)

        raw_docs = payload["documents"]
        if raw_docs:
            chunks = splitter.split_documents(raw_docs)
        else:
            chunks = []

        ids = [make_chunk_id(src_key, i, doc.page_content) for i, doc in enumerate(chunks)]
        if chunks:
            vectorstore.add_documents(chunks, ids=ids)

        new_sources[src_key] = {
            "kind": payload["kind"],
            "path": payload["path"],
            "signature": payload["signature"],
            "raw_count": len(raw_docs),
            "chunk_count": len(ids),
            "ids": ids,
        }

    for src_key in unchanged:
        new_sources[src_key] = old_sources[src_key]

    save_index_state({"version": 1, "sources": new_sources})
    print("Done. Incremental index update completed.")
    print(f"Tracked sources: {len(new_sources)}")


if __name__ == "__main__":
    create_pipeline()
