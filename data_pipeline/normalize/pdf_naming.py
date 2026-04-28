import re


TURKISH_MAP = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ü": "u",
        "Ü": "u",
        "ğ": "g",
        "Ğ": "g",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ç": "c",
        "Ç": "c",
    }
)


def create_pdf_filename(link_parent_text):
    text = (link_parent_text or "").strip().lower()
    if len(text) < 5:
        text = "bolum_tanitim_sunumu"
    else:
        text = text.replace("tiklayiniz", "")
        text = text.replace("için", "")
        text = text.replace("icin", "")
        text = text.replace(".", " ")
        text = text.translate(TURKISH_MAP)

    text = re.sub(r"[^a-z0-9\s_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    filename = "_".join(text.split()) + ".pdf"

    if filename == ".pdf":
        filename = "bolum_tanitim_sunumu.pdf"
    return filename
