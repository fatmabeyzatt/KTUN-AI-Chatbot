import csv
import json
import os
import re
import sqlite3
from urllib.parse import unquote, urlparse

from data_layout import discover_pdf_files, discover_structured_files


TR_MAP = str.maketrans(
    {
        "\u0131": "i",
        "\u011f": "g",
        "\u00fc": "u",
        "\u015f": "s",
        "\u00f6": "o",
        "\u00e7": "c",
        "\u0130": "i",
        "I": "i",
        "\u011e": "g",
        "\u00dc": "u",
        "\u015e": "s",
        "\u00d6": "o",
        "\u00c7": "c",
    }
)


def normalize_text(text):
    value = (text or "").translate(TR_MAP).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_csv_dialect(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as csv_file:
        sample = csv_file.read(4096)
    if not sample.strip():
        return ","
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        detected = dialect.delimiter
    except Exception:
        detected = ","

    # Prefer the delimiter that appears most in the header row when sniffing is ambiguous.
    counts = {",": first_line.count(","), ";": first_line.count(";"), "\t": first_line.count("\t"), "|": first_line.count("|")}
    best_delim = max(counts, key=counts.get)
    if counts.get(best_delim, 0) > counts.get(detected, 0):
        return best_delim
    return detected


def normalize_key(text):
    return "".join(ch for ch in normalize_text(text) if ch.isalnum())


def classify_title(name_title):
    value = normalize_text(name_title)
    if "prof" in value:
        return "prof"
    if "doc" in value:
        return "doc"
    if "ars gor" in value:
        return "ars_gor"
    if "dr ogr uyesi" in value:
        return "dr_ogr_uyesi"
    if "dr" in value:
        return "dr"
    return "other"


def _table_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _safe_basename_from_url(url):
    if not url:
        return ""
    try:
        path = urlparse(url).path
        return os.path.basename(unquote(path))
    except Exception:
        return ""


def _build_pdf_lookup(data_dir):
    lookup = {}
    for path in discover_pdf_files(data_dir):
        lookup[os.path.basename(path).lower()] = path
    return lookup


def _resolve_local_pdf_path(title, url, pdf_lookup):
    candidates = []
    title_text = (title or "").strip()
    if title_text and title_text.lower().endswith(".pdf"):
        candidates.append(os.path.basename(title_text))

    url_name = _safe_basename_from_url(url)
    if url_name:
        candidates.append(url_name)

    for candidate in candidates:
        matched = pdf_lookup.get(candidate.lower())
        if matched:
            return os.path.abspath(matched)
    return ""

class StructuredStore:
    def __init__(self, db_path):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_schema(self, conn):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS personnel (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name_title TEXT NOT NULL,
                title_kind TEXT NOT NULL,
                faculty TEXT,
                department TEXT,
                email TEXT,
                profile_url TEXT,
                source_file TEXT,
                row_idx INTEGER
            );

            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_name TEXT,
                title TEXT,
                content TEXT,
                url TEXT,
                source_file TEXT,
                row_idx INTEGER
            );

            CREATE TABLE IF NOT EXISTS links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                url TEXT,
                local_path TEXT,
                is_local INTEGER DEFAULT 0,
                source_file TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_personnel_title_kind ON personnel(title_kind);
            CREATE INDEX IF NOT EXISTS idx_personnel_name ON personnel(name_title);
            CREATE INDEX IF NOT EXISTS idx_sections_title ON sections(title);
            CREATE INDEX IF NOT EXISTS idx_links_title ON links(title);
            """
        )

        # Lightweight migration for existing DB files.
        link_cols = _table_columns(conn, "links")
        if "local_path" not in link_cols:
            conn.execute("ALTER TABLE links ADD COLUMN local_path TEXT")
        if "is_local" not in link_cols:
            conn.execute("ALTER TABLE links ADD COLUMN is_local INTEGER DEFAULT 0")
        conn.commit()

    def _clear_tables(self, conn):
        conn.execute("DELETE FROM personnel")
        conn.execute("DELETE FROM sections")
        conn.execute("DELETE FROM links")
        conn.commit()

    def rebuild_from_data(self, data_dir):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)
            self._clear_tables(conn)

            stats = {"personnel": 0, "sections": 0, "links": 0}
            pdf_lookup = _build_pdf_lookup(data_dir)
            structured_files = discover_structured_files(data_dir)
            csv_files = [path for path in structured_files if path.lower().endswith(".csv")]
            json_files = [path for path in structured_files if path.lower().endswith(".json")]

            for csv_path in csv_files:
                name = os.path.basename(csv_path)
                delimiter = detect_csv_dialect(csv_path)
                with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=delimiter)
                    if reader.fieldnames:
                        reader.fieldnames = [(f or "").strip().lstrip("\ufeff") for f in reader.fieldnames]
                    for idx, row in enumerate(reader):
                        cleaned = {}
                        for key, value in row.items():
                            if key is None:
                                continue
                            cleaned_key = str(key).strip().lstrip("\ufeff")
                            cleaned_value = "" if value is None else str(value).strip()
                            cleaned[cleaned_key] = cleaned_value

                        key_map = {normalize_key(k): k for k in cleaned.keys()}
                        is_personnel = {"isimunvan", "email", "profilurl"} <= set(key_map.keys())
                        is_section = {"baslik", "icerik"} <= set(key_map.keys())
                        is_announcement = {"baslik", "detayliicerik"} <= set(key_map.keys())

                        if is_personnel:
                            name_title = cleaned.get(key_map["isimunvan"], "")
                            faculty = cleaned.get(key_map.get("fakulte", ""), "")
                            department = cleaned.get(key_map.get("bolum", ""), "")
                            email = cleaned.get(key_map.get("email", ""), "")
                            profile_url = cleaned.get(key_map.get("profilurl", ""), "")
                            conn.execute(
                                """
                                INSERT INTO personnel
                                (name_title, title_kind, faculty, department, email, profile_url, source_file, row_idx)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    name_title,
                                    classify_title(name_title),
                                    faculty,
                                    department,
                                    email,
                                    profile_url,
                                    name,
                                    idx,
                                ),
                            )
                            stats["personnel"] += 1
                            continue

                        if is_section:
                            page_name = cleaned.get(key_map.get("sayfaadi", ""), "")
                            title = cleaned.get(key_map.get("baslik", ""), "")
                            content = cleaned.get(key_map.get("icerik", ""), "")
                            url = cleaned.get(key_map.get("url", ""), "")
                            conn.execute(
                                """
                                INSERT INTO sections
                                (page_name, title, content, url, source_file, row_idx)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (page_name, title, content, url, name, idx),
                            )
                            stats["sections"] += 1
                            continue

                        if is_announcement:
                            title = cleaned.get(key_map.get("baslik", ""), "")
                            detail = cleaned.get(key_map.get("detayliicerik", ""), "")
                            date_value = cleaned.get(key_map.get("tarih", ""), "")
                            url = cleaned.get(key_map.get("sayfaurl", ""), "")
                            composed = detail
                            if date_value:
                                composed = f"Tarih: {date_value}\n{detail}".strip()
                            conn.execute(
                                """
                                INSERT INTO sections
                                (page_name, title, content, url, source_file, row_idx)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                ("Duyurular", title, composed, url, name, idx),
                            )
                            stats["sections"] += 1
                            continue

            for json_path in json_files:
                name = os.path.basename(json_path)
                with open(json_path, "r", encoding="utf-8", errors="replace") as json_file:
                    payload = json.load(json_file)

                if isinstance(payload, dict):
                    for title, url in payload.items():
                        if isinstance(url, str) and url.strip().startswith("http"):
                            local_path = _resolve_local_pdf_path(str(title), url.strip(), pdf_lookup)
                            conn.execute(
                                """
                                INSERT INTO links (title, url, local_path, is_local, source_file)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (str(title), url.strip(), local_path, int(bool(local_path)), name),
                            )
                            stats["links"] += 1
                elif isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        title = item.get("title") or item.get("baslik") or item.get("name")
                        url = item.get("url") or item.get("link") or item.get("kaynak")
                        if title and isinstance(url, str) and url.strip().startswith("http"):
                            local_path = _resolve_local_pdf_path(str(title), url.strip(), pdf_lookup)
                            conn.execute(
                                """
                                INSERT INTO links (title, url, local_path, is_local, source_file)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (str(title), url.strip(), local_path, int(bool(local_path)), name),
                            )
                            stats["links"] += 1

            conn.commit()
            return stats

    def ensure_ready(self, data_dir):
        input_files = discover_structured_files(data_dir) + discover_pdf_files(data_dir)
        if not os.path.exists(self.db_path):
            return self.rebuild_from_data(data_dir)

        db_mtime = os.path.getmtime(self.db_path)
        for path in input_files:
            if os.path.getmtime(path) > db_mtime:
                return self.rebuild_from_data(data_dir)

        with self._connect() as conn:
            self._create_schema(conn)
            row = conn.execute("SELECT COUNT(*) AS c FROM personnel").fetchone()
            count = int(row["c"]) if row else 0
            if count == 0:
                return self.rebuild_from_data(data_dir)
        return None

    def list_personnel(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT name_title, title_kind, faculty, department, email, profile_url, source_file
                FROM personnel
                ORDER BY name_title ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_sections(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT page_name, title, content, url, source_file
                FROM sections
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_links(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT title, url, local_path, is_local, source_file
                FROM links
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]
