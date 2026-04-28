import os
import sys

# Ensure project root is importable even when script is run from another cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.config.driver import create_chrome_driver
from data_pipeline.config.paths import (
    DATA_ROOT,
    HAKKIMIZDA_CSV,
    ONEMLI_LINKLER_JSON,
    PERSONEL_CSV,
    STATIK_PDF_KLASORU,
    ensure_directories,
)
from data_pipeline.crawler.akademik_personel import modul_akademik_personel
from data_pipeline.crawler.bolum_tanitimi import modul_bolum_tanitimi
from data_pipeline.crawler.hakkimizda import modul_hakkimizda


def run_static_pipeline():
    ensure_directories()
    driver = create_chrome_driver()

    try:
        print(f"Static pipeline started. Output root: {DATA_ROOT}")

        modul_akademik_personel(
            driver,
            "https://www.ktun.edu.tr/tr/Birim/AkademikPersonel/?brm=EpIs2PfEpbftz7ni7vE3uw==",
            PERSONEL_CSV,
        )
        modul_hakkimizda(
            driver,
            "https://www.ktun.edu.tr/tr/Birim/Hakkimizda/?brm=CjjgkdJ2kGZdNA6detUMmQ==",
            HAKKIMIZDA_CSV,
        )
        modul_bolum_tanitimi(
            driver,
            "https://www.ktun.edu.tr/tr/Birim/Index/?brm=z3N8SGVqVghhlNlb8qFEpA==",
            STATIK_PDF_KLASORU,
            ONEMLI_LINKLER_JSON,
        )

        print("Static pipeline completed.")
    finally:
        driver.quit()


if __name__ == "__main__":
    run_static_pipeline()
