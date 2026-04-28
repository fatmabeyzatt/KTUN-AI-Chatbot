import os

from data_layout import DATA_ROOT, PROCESSED_DIR, RAW_DYNAMIC_DIR, RAW_PDF_DIR, ensure_data_directories


# Static / Dynamic folders (same logic as your script, but inside project)
STATIK_KLASOR = PROCESSED_DIR
DINAMIK_KLASOR = RAW_DYNAMIC_DIR
STATIK_PDF_KLASORU = RAW_PDF_DIR

# Output files
PERSONEL_CSV = os.path.join(STATIK_KLASOR, "akademik_personel.csv")
HAKKIMIZDA_CSV = os.path.join(STATIK_KLASOR, "hakkimizda_dikey.csv")
ONEMLI_LINKLER_JSON = os.path.join(STATIK_KLASOR, "bolum_tanitimi_linkleri.json")


def ensure_directories():
    ensure_data_directories()
