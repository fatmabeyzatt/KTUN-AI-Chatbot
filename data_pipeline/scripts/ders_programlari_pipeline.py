import os
import requests
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import csv
import re
import PyPDF2
import sys

# Ensure project root is importable even when script is run from another cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_layout import DATA_ROOT, RAW_PDF_DIR, ensure_data_directories

# -----------------------------
# 1. DINAMIK KLASOR MIMARISI
# -----------------------------
ensure_data_directories()
dinamik_klasor = DATA_ROOT
ders_pdf_klasoru = os.path.join(RAW_PDF_DIR, "ders_programlari")

os.makedirs(ders_pdf_klasoru, exist_ok=True)
csv_dosyasi = os.path.join(dinamik_klasor, "guncel_ders_programlari.csv")

# -----------------------------
# 2. CHROME AYARLARI
# -----------------------------
options = Options()
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

# ==========================================
# DINAMIK MODUL: AKILLI DERS PROGRAMI AVCISI
# ==========================================
def dinamik_ders_programlarini_cek(ana_url):
    print("\n" + "="*60 + "\n🚀 DINAMIK MODUL: GUNCEL DERS PROGRAMLARI BASLATILDI\n" + "="*60)
    driver.get(ana_url)
    time.sleep(3)
    
    detayli_veriler = []
    indirilen_linkler = set()

    def metni_parcala(metin, chunk_size=1400, overlap=180):
        text = (metin or "").strip()
        if not text:
            return [""]
        if len(text) <= chunk_size:
            return [text]
        parcalar = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            parca = text[start:end].strip()
            if parca:
                parcalar.append(parca)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return parcalar or [text]
    
    # --- ASAMA 1: PDF LINKLERINI BUL VE AKILLI FILTREDEN GECIR ---
    try:
        ana_icerik_kutusu = driver.find_element(By.CLASS_NAME, "gdlr-core-pbf-sidebar-content-inner")
    except:
        ana_icerik_kutusu = driver.find_element(By.TAG_NAME, "body")

    tum_linkler = ana_icerik_kutusu.find_elements(By.TAG_NAME, "a")
    ham_pdf_linkleri = []

    for a in tum_linkler:
        href = a.get_attribute("href")
        metin = a.text.strip()
        
        if href and ".pdf" in href.lower() and href not in indirilen_linkler:
            ham_pdf_linkleri.append({"Metin": metin, "URL": href})
            indirilen_linkler.add(href)

    # 🧠 AKILLI DONEM FILTRESI
    tum_yillar = []
    for pdf in ham_pdf_linkleri:
        # Metindeki tum "20XX" formatindaki yillari bul
        yillar = re.findall(r'20\d{2}', pdf["Metin"])
        tum_yillar.extend([int(y) for y in yillar])

    pdf_linkleri = []
    if tum_yillar:
        en_guncel_yil = max(tum_yillar)
        print(f"🎯 Sistem otomatik olarak en guncel egitim yilini tespit etti: {en_guncel_yil}")

        for pdf in ham_pdf_linkleri:
            # Sadece isminde en guncel yil gecenleri VEYA isminde hic yil gecmeyen genel linkleri al
            if str(en_guncel_yil) in pdf["Metin"] or not re.search(r'20\d{2}', pdf["Metin"]):
                pdf_linkleri.append(pdf)
    else:
        # Eger sitede hic yil yazmiyorsa, en ustteki 4 taneyi al
        pdf_linkleri = ham_pdf_linkleri[:4]

    print(f"✅ Zeka Filtresi devrede: Eski yillar elendi, {len(pdf_linkleri)} adet GUNCEL program kaldi.")

    # --- ASAMA 2: PDF'LERI INDIR VE ICINI OKU ---
    for i, pdf_bilgi in enumerate(pdf_linkleri, start=1):
        tam_link = urljoin(ana_url, pdf_bilgi["URL"])
        
        if len(pdf_bilgi["Metin"]) < 5:
            link_basligi = f"Ders_Programi_Belge_{i}"
        else:
            link_basligi = pdf_bilgi["Metin"]

        temiz_isim = link_basligi.lower().replace(" ", "_")
        temiz_isim = temiz_isim.translate(str.maketrans("iuogsc.", "iuogsc_"))
        dosya_adi = f"Program_{i}_{temiz_isim[:30]}.pdf"
        dosya_yolu = os.path.join(ders_pdf_klasoru, dosya_adi)

        pdf_icerik_metni = ""

        try:
            print(f"\n📥 [{i}/{len(pdf_linkleri)}] Indiriliyor ve Okunuyor: {dosya_adi}")
            
            response = requests.get(tam_link)
            with open(dosya_yolu, "wb") as f:
                f.write(response.content)

            with open(dosya_yolu, "rb") as f_pdf:
                pdf_okuyucu = PyPDF2.PdfReader(f_pdf)
                sayfa_sayisi = len(pdf_okuyucu.pages)
                
                for sayfa_no in range(sayfa_sayisi):
                    sayfa = pdf_okuyucu.pages[sayfa_no]
                    okunan_metin = sayfa.extract_text()
                    if okunan_metin:
                        pdf_icerik_metni += okunan_metin + " "

            pdf_icerik_metni = re.sub(r'\s+', ' ', pdf_icerik_metni).strip()
            
            if len(pdf_icerik_metni) < 10:
                pdf_icerik_metni = "UYARI: PDF icerigi tablo tabanli oldugu icin metin olarak okunamadi."
            else:
                print("   ✅ PDF icerigi basariyla metne donusturuldu.")

        except Exception as e:
            print(f"   ❌ PDF islenirken hata olustu: {e}")
            pdf_icerik_metni = f"Hata olustu: {str(e)}"

        eski_donem_linki = "Eski donem programlari ve detaylar icin: " + ana_url

        parcalar = metni_parcala(pdf_icerik_metni)
        for parca_no, parca_metin in enumerate(parcalar, start=1):
            detayli_veriler.append([
                link_basligi,
                parca_metin,
                dosya_adi,
                tam_link,
                eski_donem_linki,
                parca_no
            ])

    # --- ASAMA 3: CSV'YE GUVENLI KAYIT ---
    if detayli_veriler:
        kaydedildi = False
        while not kaydedildi:
            try:
                with open(csv_dosyasi, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(["Program_Basligi", "PDF_Icerik_Metni", "Indirilen_Dosya", "PDF_Link", "Eski_Donem_Yonlendirme", "Icerik_Parca_No"])
                    writer.writerows(detayli_veriler)
                
                print(f"\n🎉 HARIKA! Sadece guncel programlar okunup '{os.path.basename(csv_dosyasi)}' dosyasina kaydedildi.")
                kaydedildi = True
                
            except PermissionError:
                input(f"\n⚠️ DIKKAT: '{os.path.basename(csv_dosyasi)}' su an acik! Kapatip ENTER'a basin...")

# -----------------------------
# CALISTIRICI
# -----------------------------
try:
    print("Sistem Baslatiliyor...")
    ders_programlari_linki = "https://www.ktun.edu.tr/tr/Birim/Index/?brm=3ptUuE+8tZOrPrMNwm4Alw==" 
    dinamik_ders_programlarini_cek(ders_programlari_linki)
finally:
    driver.quit()
