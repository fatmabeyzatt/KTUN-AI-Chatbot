import base64
from io import BytesIO
import re

from PIL import Image, ImageFilter, ImageOps
import pytesseract


# Keep your exact Tesseract path.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


WHITELIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@.-_"
OCR_CONFIGS = [
    rf"--oem 3 --psm 7 -c tessedit_char_whitelist={WHITELIST}",
    rf"--oem 3 --psm 8 -c tessedit_char_whitelist={WHITELIST}",
    rf"--oem 3 --psm 6 -c tessedit_char_whitelist={WHITELIST}",
]


def _decode_base64_image(image_src):
    if "base64," not in image_src:
        return None
    base64_str = image_src.split("base64,")[1]
    return Image.open(BytesIO(base64.b64decode(base64_str)))


def _build_variants(img):
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)

    upscale = gray.resize((gray.width * 4, gray.height * 4), Image.Resampling.LANCZOS)
    sharpened = upscale.filter(ImageFilter.SHARPEN)

    variants = [upscale, sharpened]
    for threshold in (110, 130, 150, 170, 190):
        bw = sharpened.point(lambda px, t=threshold: 255 if px > t else 0)
        variants.append(bw)
        variants.append(ImageOps.invert(bw))
    return variants


def _normalize_domain(domain):
    d = (domain or "").strip().lower()
    d = d.replace(" ", "")
    d = d.replace("edutr", "edu.tr")
    d = d.replace("edut", "edu.tr")
    d = d.replace("edu.", "edu.tr")
    d = d.replace("ktn.", "ktun.")
    if "ktun" in d or not d:
        return "ktun.edu.tr"
    return d


def _extract_candidate(raw_text):
    text = (raw_text or "").strip().lower()
    text = text.replace("\n", "").replace("\r", "").replace(" ", "")
    if not text:
        return ""

    # First try direct email pattern.
    pattern = r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if matches:
        return matches[0].lower()

    # Then clean noisy OCR characters and keep possible token.
    cleaned = re.sub(r"[^a-z0-9@._-]", "", text)
    return cleaned


def _to_email(candidate):
    if not candidate:
        return ""

    token = candidate.lower().strip()
    token = token.replace("..", ".")

    if "@" in token:
        local, domain = token.split("@", 1)
        local = re.sub(r"[^a-z0-9._-]", "", local)
        if not local:
            return ""
        return f"{local}@{_normalize_domain(domain)}"

    # No @ found: try splitting around ktun marker.
    idx = token.find("ktun")
    if idx != -1:
        local = token[:idx]
        local = re.sub(r"(i|l|g)$", "", local)  # trim common OCR tail noise
        local = re.sub(r"[^a-z0-9._-]", "", local)
        if local:
            return f"{local}@ktun.edu.tr"

    token = re.sub(r"[^a-z0-9._-]", "", token)
    if not token:
        return ""
    return f"{token}@ktun.edu.tr"


def _score_email(email):
    if not email:
        return -1
    score = 0
    if "@" in email:
        score += 5
    if email.endswith("@ktun.edu.tr"):
        score += 5

    local = email.split("@", 1)[0] if "@" in email else email
    if 3 <= len(local) <= 30:
        score += 2

    # Penalize obvious broken sequences.
    if ".." in email or "__" in email:
        score -= 1
    if len(local) < 3:
        score -= 2
    return score


def extract_email_from_base64_image_src(image_src):
    try:
        img = _decode_base64_image(image_src)
        if img is None:
            return "Bilinmiyor"
    except Exception:
        return "Bilinmiyor"

    candidates = set()
    for variant in _build_variants(img):
        for config in OCR_CONFIGS:
            try:
                raw_text = pytesseract.image_to_string(variant, config=config)
            except Exception:
                continue
            candidate = _extract_candidate(raw_text)
            email = _to_email(candidate)
            if email:
                candidates.add(email)

    if not candidates:
        return "Bilinmiyor"

    best = max(candidates, key=_score_email)
    return best
