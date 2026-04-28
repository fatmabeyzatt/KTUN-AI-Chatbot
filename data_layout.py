import glob
import os


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
RAW_DYNAMIC_DIR = os.path.join(RAW_DIR, "dynamic_veri")
RAW_PDF_DIR = os.path.join(RAW_DIR, "pdf")


def ensure_data_directories():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_DYNAMIC_DIR, exist_ok=True)
    os.makedirs(RAW_PDF_DIR, exist_ok=True)


def _unique_sorted(paths):
    normalized = {os.path.abspath(path) for path in paths if os.path.isfile(path)}
    return sorted(normalized)


def discover_structured_files(data_root=DATA_ROOT):
    # Backward-compatible: read both legacy `data/*` and new `data/processed/**/*`.
    top_level_csv = glob.glob(os.path.join(data_root, "*.csv"))
    top_level_json = glob.glob(os.path.join(data_root, "*.json"))
    processed_csv = glob.glob(os.path.join(data_root, "processed", "**", "*.csv"), recursive=True)
    processed_json = glob.glob(os.path.join(data_root, "processed", "**", "*.json"), recursive=True)
    return _unique_sorted(top_level_csv + top_level_json + processed_csv + processed_json)


def discover_pdf_files(data_root=DATA_ROOT):
    pdf_paths = glob.glob(os.path.join(data_root, "raw", "pdf", "**", "*.pdf"), recursive=True)
    return _unique_sorted(pdf_paths)
