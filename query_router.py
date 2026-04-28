import re
import os

from structured_store import normalize_text


STOPWORDS = {
    "ve",
    "ile",
    "olan",
    "olanlar",
    "listeler",
    "misin",
    "nedir",
    "kim",
    "kimdir",
    "bolum",
    "bilgisayar",
    "muhendisligi",
    "universitesi",
    "konya",
    "teknik",
    "un",
    "nin",
    "nun",
    "yerine",
}

EMAIL_HINTS = {"mail", "email", "eposta", "e posta", "adresi", "adresi nedir"}
PERSONNEL_HINTS = {"personel", "hoca", "ogretim uyesi", "ogretim elemani", "unvan", "listele"}
LINK_HINTS = {"link", "pdf", "dosya", "sunum", "baglanti", "url"}
SECTION_HINTS = {
    "misyon",
    "vizyon",
    "farabi",
    "erasmus",
    "mezuniyet",
    "kabul",
    "amac",
    "program tanimi",
    "program dili",
    "bolum baskani",
    "baskan yardimcisi",
    "koordinator",
    "istihdam",
    "olcme",
    "degerlendirme",
    "onceki ogrenme",
    "ust derece",
}
ANNOUNCEMENT_HINTS = {
    "duyuru",
    "duyurular",
    "sinav",
    "program",
    "basvuru",
    "basvurular",
    "etkinlik",
    "toplanti",
    "staj",
    "teknofest",
    "hangar",
}

TITLE_FILTERS = {
    "prof": {"prof"},
    "profesor": {"prof"},
    "doct": {"doc"},
    "doc": {"doc"},
    "ars gor": {"ars_gor"},
    "arastirma gorevlisi": {"ars_gor"},
    "dr ogr uyesi": {"dr_ogr_uyesi"},
}


def extract_count(question, default=5):
    match = re.search(r"\b(\d+)\b", question)
    if not match:
        return default
    value = int(match.group(1))
    return max(1, min(value, 20))


def extract_tokens(text):
    tokens = normalize_text(text).split()
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def overlap_score(query_tokens, text):
    hay = set(extract_tokens(text))
    if not hay:
        return 0
    return sum(1 for tok in query_tokens if tok in hay)


def find_title_filter(normalized_question):
    for key, value in TITLE_FILTERS.items():
        if key in normalized_question:
            return value
    return None


class QueryRouter:
    def __init__(self, structured_store):
        self.store = structured_store

    def try_answer(self, question):
        q_norm = normalize_text(question)

        email_answer = self._answer_email(question, q_norm)
        if email_answer:
            return email_answer

        personnel_answer = self._answer_personnel(question, q_norm)
        if personnel_answer:
            return personnel_answer

        announcement_answer = self._answer_announcement(question, q_norm)
        if announcement_answer:
            return announcement_answer

        link_answer = self._answer_link(question, q_norm)
        if link_answer:
            return link_answer

        section_answer = self._answer_section(question, q_norm)
        if section_answer:
            return section_answer

        return None

    def _answer_email(self, question, q_norm):
        if not any(h in q_norm for h in EMAIL_HINTS):
            return None

        rows = self.store.list_personnel()
        if not rows:
            return None

        q_tokens = extract_tokens(question)
        best_row = None
        best_score = -1

        for row in rows:
            name = row.get("name_title", "")
            score = overlap_score(q_tokens, name)
            if score > best_score:
                best_score = score
                best_row = row

        if not best_row or best_score <= 0:
            return None

        email = (best_row.get("email") or "").strip()
        if not email:
            return {
                "answer": f"{best_row.get('name_title')} icin e-posta kaydi bulunamadi.",
                "sources": [best_row.get("source_file", "structured.db")],
                "route": "structured_email",
            }

        if "@" not in email:
            answer = (
                f"{best_row.get('name_title')} icin veri kaynaginda e-posta su sekilde geciyor: {email}\n"
                "Not: Bu alan ham veriden geldigi icin format bozuk olabilir."
            )
        else:
            answer = f"{best_row.get('name_title')} e-posta adresi: {email}"

        return {
            "answer": answer,
            "sources": [best_row.get("source_file", "structured.db"), best_row.get("profile_url", "")],
            "route": "structured_email",
        }

    def _answer_personnel(self, question, q_norm):
        if not any(h in q_norm for h in PERSONNEL_HINTS):
            return None

        rows = self.store.list_personnel()
        if not rows:
            return None

        title_filter = find_title_filter(q_norm)
        if title_filter:
            rows = [row for row in rows if row.get("title_kind") in title_filter]
            if not rows:
                return {
                    "answer": "Istenen unvanda personel bulunamadi.",
                    "sources": ["akademik_personel.csv"],
                    "route": "structured_personnel",
                }

        q_tokens = extract_tokens(question)
        rows = sorted(
            rows,
            key=lambda row: overlap_score(
                q_tokens,
                " ".join([row.get("name_title", ""), row.get("department", ""), row.get("faculty", "")]),
            ),
            reverse=True,
        )

        count = extract_count(question, default=5)
        selected = []
        seen = set()
        for row in rows:
            name = row.get("name_title", "").strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            selected.append(name)
            if len(selected) >= count:
                break

        if not selected:
            return None

        lines = [f"{i}. {name}" for i, name in enumerate(selected, start=1)]
        title_text = "Akademik personel listesi" if not title_filter else "Filtrelenmis akademik personel listesi"
        return {
            "answer": f"{title_text}:\n" + "\n".join(lines),
            "sources": ["akademik_personel.csv"],
            "route": "structured_personnel",
        }

    def _answer_link(self, question, q_norm):
        if not any(h in q_norm for h in LINK_HINTS):
            return None

        links = self.store.list_links()
        if not links:
            return None

        q_tokens = extract_tokens(question)
        scored = []
        for row in links:
            title = row.get("title", "")
            url = row.get("url", "")
            score = overlap_score(q_tokens, f"{title} {url}")
            if "pdf" in q_norm and ".pdf" in url.lower():
                score += 2
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score, top_row = scored[0]
        if top_score <= 0:
            return None
        local_path = (top_row.get("local_path") or "").strip()
        url = (top_row.get("url") or "").strip()

        if local_path and os.path.exists(local_path):
            answer = f"{top_row.get('title')}: {local_path}"
            if url:
                answer += f"\nKaynak link: {url}"
        else:
            answer = f"{top_row.get('title')}: {url}"

        return {
            "answer": answer,
            "sources": [top_row.get("source_file", "structured.db"), local_path, url],
            "route": "structured_link",
        }

    def _extract_first_url(self, text):
        match = re.search(r"https?://\S+", text or "")
        if not match:
            return ""
        return match.group(0).rstrip(".,;)]}>")

    def _answer_announcement(self, question, q_norm):
        if not any(h in q_norm for h in ANNOUNCEMENT_HINTS):
            return None

        sections = self.store.list_sections()
        if not sections:
            return None

        announcements = [
            row
            for row in sections
            if "guncel_duyurular.csv" in (row.get("source_file", "").lower())
            or "duyuru" in normalize_text(row.get("page_name", ""))
        ]
        if not announcements:
            return None

        q_tokens = extract_tokens(question)
        scored = []
        for row in announcements:
            title = row.get("title", "")
            hay = " ".join([title, row.get("content", ""), row.get("url", "")])
            hay_norm = normalize_text(hay)
            title_norm = normalize_text(title)
            score = 0.0
            for tok in q_tokens:
                if tok in hay_norm:
                    score += 1.0
                if tok in title_norm:
                    score += 1.5
            if "sinav" in q_norm and "sinav" in hay_norm:
                score += 2.0
            if "basvuru" in q_norm and "basvuru" in hay_norm:
                score += 1.0
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_score, top_row = scored[0]
        if top_score <= 0:
            return None

        url = (top_row.get("url") or "").strip()
        title = (top_row.get("title") or "").strip() or "Duyuru"
        content = (top_row.get("content") or "").strip()
        date_match = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", content)
        date_text = date_match.group(0) if date_match else ""

        if any(h in q_norm for h in LINK_HINTS):
            if url:
                answer = url
            else:
                answer = "Bu duyuru icin link bilgisi bulunamadi."
            return {
                "answer": answer,
                "sources": [top_row.get("source_file", "structured.db"), url],
                "route": "structured_announcement",
            }

        if "var mi" in q_norm or "yayinlanmis" in q_norm:
            if date_text:
                answer = f"Evet, {date_text} tarihli {title} duyurusu yayinlanmis."
            else:
                answer = f"Evet, {title} duyurusu yayinlanmis."
        else:
            snippet = content[:500] + ("..." if len(content) > 500 else "")
            answer = f"{title}:\n{snippet}"

        return {
            "answer": answer,
            "sources": [top_row.get("source_file", "structured.db"), url],
            "route": "structured_announcement",
        }

    def _answer_section(self, question, q_norm):
        if not any(h in q_norm for h in SECTION_HINTS):
            return None

        sections = self.store.list_sections()
        if not sections:
            return None

        q_tokens = extract_tokens(question)
        scored = []
        for row in sections:
            title_norm = normalize_text(row.get("title", ""))
            hay = " ".join([row.get("title", ""), row.get("content", ""), row.get("page_name", "")])
            score = overlap_score(q_tokens, hay)
            for hint in SECTION_HINTS:
                if hint in q_norm and hint in title_norm:
                    score += 3
            if score > 0:
                scored.append((score, row))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        top_row = scored[0][1]
        content = (top_row.get("content") or "").strip()
        snippet = content[:700] + ("..." if len(content) > 700 else "")
        title = top_row.get("title") or "Bolum bilgisi"
        url = top_row.get("url") or ""
        answer = f"{title}:\n{snippet}"
        if url:
            answer += f"\nKaynak URL: {url}"

        return {
            "answer": answer,
            "sources": [top_row.get("source_file", "structured.db"), url],
            "route": "structured_section",
        }
