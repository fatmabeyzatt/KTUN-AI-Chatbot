import json
from urllib.parse import urljoin

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from data_pipeline.normalize.pdf_naming import create_pdf_filename
from data_pipeline.pdf_download.downloader import download_pdf


def modul_bolum_tanitimi(driver, url, statik_pdf_klasoru, onemli_linkler_json):
    print("\n" + "=" * 50 + "\n🚀 MODÜL 3: BÖLÜM TANITIMI BAŞLATILDI\n" + "=" * 50)
    driver.get(url)
    WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.TAG_NAME, "a")))

    indirilen_linkler = set()
    ust_linkler_bilgi = {}

    try:
        tum_linkler = driver.find_elements(By.TAG_NAME, "a")
        for a_tag in tum_linkler:
            href = (a_tag.get_attribute("href") or "").strip()
            if not href or ".pdf" not in href.lower() or href in indirilen_linkler:
                continue

            tam_link = urljoin(url, href)

            try:
                cumle_metni = a_tag.find_element(By.XPATH, "..").text.strip()
            except Exception:
                cumle_metni = ""

            dosya_adi = create_pdf_filename(cumle_metni)

            print(f"📥 İndiriliyor: {dosya_adi}")
            download_pdf(tam_link, statik_pdf_klasoru, dosya_adi)

            ust_linkler_bilgi[dosya_adi] = tam_link
            indirilen_linkler.add(href)

        if ust_linkler_bilgi:
            with open(onemli_linkler_json, "w", encoding="utf-8") as file:
                json.dump(ust_linkler_bilgi, file, ensure_ascii=False, indent=4)

        print("✅ Başarılı: Tüm dosyalar anlamlı isimlerle kaydedildi.")

    except Exception as exc:
        print(f"❌ PDF Modülünde hata: {exc}")
