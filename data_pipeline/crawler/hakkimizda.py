import csv
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def modul_hakkimizda(driver, url, hakkimizda_csv):
    print("\n" + "=" * 50 + "\n🚀 MODÜL 2: HAKKIMIZDA BAŞLATILDI\n" + "=" * 50)
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    satirlar, genel_metin = [], ""

    try:
        paragraflar = (
            driver.find_element(By.CLASS_NAME, "gdlr-core-pbf-sidebar-content-inner").find_elements(By.TAG_NAME, "p")
        )
    except Exception:
        paragraflar = driver.find_elements(By.TAG_NAME, "p")

    for p_tag in paragraflar:
        metin = p_tag.text.strip()
        if not metin or len(metin) < 3:
            continue
        if ":" in metin:
            bolumler = metin.split(":", 1)
            satirlar.append(["Hakkımızda", url, bolumler[0].strip(), bolumler[1].strip()])
        else:
            genel_metin += " " + metin

    if genel_metin.strip():
        satirlar.append(["Hakkımızda", url, "Genel Metin", genel_metin.strip()])

    with open(hakkimizda_csv, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Sayfa_Adi", "URL", "Baslik", "Icerik"])
        writer.writerows(satirlar)
