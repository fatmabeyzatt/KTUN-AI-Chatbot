import csv

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from data_pipeline.normalize.email_ocr import extract_email_from_base64_image_src


def modul_akademik_personel(driver, ana_url, personel_csv):
    print("\n" + "=" * 50 + "\n🚀 MODÜL 1: AKADEMİK PERSONEL BAŞLATILDI\n" + "=" * 50)
    driver.get(ana_url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, 'prsnl=')]"))
        )
        link_elementleri = driver.find_elements(By.XPATH, "//a[contains(@href, 'prsnl=')]")
        dinamik_linkler = sorted(
            set([el.get_attribute("href") for el in link_elementleri if el.get_attribute("href")])
        )
        print(f"✅ Toplam {len(dinamik_linkler)} akademisyen bulundu.")
    except Exception:
        print("❌ Personel linkleri bulunamadı.")
        return

    tum_veri = []
    for i, url in enumerate(dinamik_linkler, start=1):
        driver.get(url)
        isim_unvan, fakulte, bolum, email = "Bilinmiyor", "Bilinmiyor", "Bilinmiyor", "Bilinmiyor"

        try:
            isim = WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, "h6"))).text.strip()
            if isim:
                isim_unvan = isim

            h7_etiketleri = driver.find_element(By.XPATH, "//h6/parent::div").find_elements(By.TAG_NAME, "h7")
            if h7_etiketleri:
                parcalar = h7_etiketleri[0].text.strip().split("\n")
                if len(parcalar) >= 1:
                    fakulte = parcalar[0].strip()
                if len(parcalar) >= 2:
                    bolum = parcalar[1].strip()
        except Exception:
            pass

        try:
            img_element = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.XPATH, "//img[contains(@src, 'data:image/bmp;base64')]"))
            )
            image_src = img_element.get_attribute("src")
            email = extract_email_from_base64_image_src(image_src)
        except Exception:
            pass

        tum_veri.append([isim_unvan, fakulte, bolum, email, url])
        print(f"✅ [{i}/{len(dinamik_linkler)}] {isim_unvan} çekildi.")

    with open(personel_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Isim_Unvan", "Fakulte", "Bolum", "Email", "Profil_URL"])
        writer.writerows(tum_veri)
