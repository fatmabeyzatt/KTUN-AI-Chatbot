import os
import requests
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import csv
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import re
import sys

# Ensure project root is importable even when script is run from another cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_layout import DATA_ROOT, RAW_PDF_DIR, ensure_data_directories

# -----------------------------
# ⚙️ TESSERACT YOLU (Bilgisayarındaki yolu kontrol et)
# -----------------------------
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
SYSTEM_TESSDATA_DIR = os.path.join(os.path.dirname(TESSERACT_CMD), "tessdata")
USER_TESSDATA_DIR = os.path.join(os.path.expanduser("~"), ".tesseract", "tessdata")
TESSDATA_DIR = (
    SYSTEM_TESSDATA_DIR
    if os.path.exists(os.path.join(SYSTEM_TESSDATA_DIR, "tur.traineddata"))
    else USER_TESSDATA_DIR
)
TUR_TRAINEDDATA = os.path.join(TESSDATA_DIR, "tur.traineddata")
TUR_DOWNLOAD_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/tur.traineddata"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# -----------------------------
# 1️⃣ DİNAMİK KLASÖR MİMARİSİ
# -----------------------------
ensure_data_directories()
dinamik_klasor = DATA_ROOT
dinamik_pdf_klasoru = os.path.join(RAW_PDF_DIR, "duyurular")
os.makedirs(dinamik_pdf_klasoru, exist_ok=True)
csv_dosyasi = os.path.join(DATA_ROOT, "guncel_duyurular.csv")

# -----------------------------
# 2️⃣ CHROME AYARLARI
# -----------------------------
options = Options()
# options.add_argument("--headless") # Botu sunucuya kurduğunda arka planda çalışması için bunu açabilirsin
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)


def turkce_dil_dosyasini_hazirla():
    if os.path.exists(TUR_TRAINEDDATA):
        return True

    print("⚠️ tur.traineddata bulunamadi, otomatik indirilmeye calisiliyor...")
    try:
        os.makedirs(TESSDATA_DIR, exist_ok=True)
        response = requests.get(TUR_DOWNLOAD_URL, timeout=60)
        response.raise_for_status()
        with open(TUR_TRAINEDDATA, "wb") as f:
            f.write(response.content)
        print("✅ tur.traineddata indirildi.")
        return True
    except Exception as e:
        print(f"⚠️ tur.traineddata indirilemedi: {e}")
        print("⚠️ OCR su an eng fallback ile devam edecek.")
        return False


OCR_LANG = "tur" if turkce_dil_dosyasini_hazirla() else "eng"


AY_ADLARI = {
    "OCAK",
    "SUBAT",
    "ŞUBAT",
    "MART",
    "NISAN",
    "MAYIS",
    "HAZIRAN",
    "TEMMUZ",
    "AGUSTOS",
    "AĞUSTOS",
    "EYLUL",
    "EYLÜL",
    "EKIM",
    "EKİM",
    "KASIM",
    "ARALIK",
}


def temizle_detayli_icerik(ham_metin, baslik):
    satirlar = [s.strip() for s in ham_metin.splitlines() if s.strip()]
    temiz = []
    baslik_norm = (baslik or "").strip().lower()

    for satir in satirlar:
        satir_norm = satir.lower()
        satir_buyuk = satir.upper()

        # Duyuru kartindan gelen gereksiz tekrar satirlari
        if satir_norm == baslik_norm:
            continue
        if satir_buyuk == "KTUN":
            continue
        if re.fullmatch(r"\d{1,2}", satir):
            continue
        if satir_buyuk in AY_ADLARI:
            continue

        # Tekrarlayan satirlari bir kez tut
        if temiz and temiz[-1].lower() == satir_norm:
            continue
        temiz.append(satir)

    return "\n".join(temiz).strip()


# ==========================================
# 🧩 DİNAMİK MODÜL: KAPSAMLI DUYURU AVCISI
# ==========================================
def dinamik_duyurulari_cek(ana_url):
    print("\n" + "="*60 + "\n🚀 DİNAMİK MODÜL: DUYURULAR BAŞLATILDI\n" + "="*60)
    driver.get(ana_url)
    time.sleep(3)
    
    duyuru_listesi = []

    # --- AŞAMA 1: DUYURU LİSTESİNİ TOPLA ---
    try:
        tablo = driver.find_element(By.CLASS_NAME, "table")
        satirlar = tablo.find_elements(By.XPATH, ".//tbody/tr")
        for satir in satirlar:
            sutunlar = satir.find_elements(By.TAG_NAME, "td")
            if len(sutunlar) >= 3:
                baslik = sutunlar[0].text.strip()
                tarih = sutunlar[1].text.strip()
                link = sutunlar[2].find_element(By.TAG_NAME, "a").get_attribute("href")
                duyuru_listesi.append({"Tarih": tarih, "Baslik": baslik, "Link": link})
    except Exception as e:
        print(f"❌ Duyuru tablosu okunamadı: {e}")
        return

    detayli_veriler = []
    
    # --- AŞAMA 2: DUYURU DETAYLARINA GİR VE SÖMÜR ---
    for i, duyuru in enumerate(duyuru_listesi, start=1):
        print(f"\n🔍 [{i}/{len(duyuru_listesi)}] İnceleniyor: {duyuru['Baslik']}")
        driver.get(duyuru['Link'])
        time.sleep(2)
        
        icerik_metni = ""
        ek_linkler = []
        indirilen_pdfler = []
        
        # Sadece ana metin kutusunu hedefle
        try:
            ana_icerik_kutusu = driver.find_element(By.CLASS_NAME, "gdlr-core-pbf-sidebar-content-inner")
        except:
            ana_icerik_kutusu = driver.find_element(By.TAG_NAME, "body")

        # 2.1 - ANA METNİ ÜTÜLE (Temizle)
        try:
            ham_metin = ana_icerik_kutusu.text.strip()
            icerik_metni = re.sub(r'\n{2,}', '\n', ham_metin) # Fazla boşlukları tek satıra indir
            icerik_metni = temizle_detayli_icerik(icerik_metni, duyuru['Baslik'])
        except: pass

        # 2.2 - GELİŞMİŞ GÖRSEL OKUMA (SADECE TÜRKÇE VE BLOK OKUMA)
        try:
            resimler = ana_icerik_kutusu.find_elements(By.TAG_NAME, "img")
            for resim in resimler:
                img_src = resim.get_attribute("src")
                
                if img_src and "logo" not in img_src.lower() and "icon" not in img_src.lower():
                    try:
                        print(f"   🖼️ Afiş tespit edildi, gelişmiş Türkçe OCR ile okunuyor...")
                        response = requests.get(img_src)
                        img = Image.open(BytesIO(response.content))
                        
                        # Görüntü Ön İşleme (Preprocessing)
                        img = img.resize((img.width * 3, img.height * 3), Image.Resampling.LANCZOS)
                        img = img.convert('L')
                        img = ImageEnhance.Contrast(img).enhance(2.5)
                        img = img.filter(ImageFilter.SHARPEN)
                        
                        # Tesseract'i hazir dil ile blok modunda calistir
                        ocr_config = r'--oem 3 --psm 4'
                        okunan_yazi = pytesseract.image_to_string(img, lang=OCR_LANG, config=ocr_config).strip()
                        
                        if len(okunan_yazi) > 5:
                            # Sadece harf ve rakamları bırak, saçma sembolleri yok et
                            okunan_yazi = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ0-9\s.,:!?/-]', '', okunan_yazi)
                            temiz_ocr = re.sub(r'\s+', ' ', okunan_yazi)
                            
                            icerik_metni += f"\n[GÖRSEL İÇERİĞİ]: {temiz_ocr}"
                            print("   ✅ Görseldeki yazı başarıyla metne eklendi.")
                    except Exception as e:
                        print(f"   ⚠️ Görsel atlandı: {e}")
        except: pass

        # 2.3 - PDF VE LİNK AVCISI
        try:
            tum_linkler = ana_icerik_kutusu.find_elements(By.TAG_NAME, "a")
            for a in tum_linkler:
                href = a.get_attribute("href")
                link_metni = a.text.strip()
                
                if not href or href == duyuru['Link']: continue
                    
                if ".pdf" in href.lower():
                    tam_link = urljoin(duyuru['Link'], href)
                    temiz_isim = link_metni.lower().replace(" ", "_") if len(link_metni) > 3 else "belge"
                    temiz_isim = temiz_isim.translate(str.maketrans("ıüğoşç.", "iugosc_"))
                    dosya_adi = f"Duyuru_{i}_{temiz_isim[:20]}.pdf" 
                    dosya_yolu = os.path.join(dinamik_pdf_klasoru, dosya_adi)
                    
                    if not os.path.exists(dosya_yolu):
                        print(f"   📥 PDF İndiriliyor: {dosya_adi}")
                        with open(dosya_yolu, "wb") as f:
                            f.write(requests.get(tam_link).content)
                    
                    indirilen_pdfler.append(dosya_adi)
                
                elif len(link_metni) > 3 and "ktun.edu.tr" not in href and "http" in href:
                    ek_linkler.append(f"{link_metni}: {href}")
        except: pass
        
        detayli_veriler.append([
            duyuru['Tarih'],
            duyuru['Baslik'],
            icerik_metni.strip(),
            " | ".join(ek_linkler) if ek_linkler else "-",
            " | ".join(indirilen_pdfler) if indirilen_pdfler else "-",
            duyuru['Link']
        ])

    # --- AŞAMA 3: CSV'YE GÜVENLİ KAYIT ---
    if detayli_veriler:
        kaydedildi = False
        while not kaydedildi:
            try:
                with open(csv_dosyasi, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(["Tarih", "Baslik", "Detayli_Icerik", "Ek_Linkler", "Indirilen_PDFler", "Sayfa_URL"])
                    writer.writerows(detayli_veriler)
                
                print(f"\n🎉 MÜKEMMEL! Tüm duyurular en temiz haliyle CSV'ye kaydedildi.")
                kaydedildi = True
                
            except PermissionError:
                input(f"\n⚠️ DİKKAT: '{os.path.basename(csv_dosyasi)}' şu an arka planda (Excel vb.) açık!\nLütfen dosyayı kapatın ve kaydetmek için klavyeden ENTER'a basın...")

# -----------------------------
# ⚙️ ÇALIŞTIRICI BLOĞU
# -----------------------------
try:
    print(f"🛠️ Sistem Başlatılıyor... Hedef Klasör: {dinamik_klasor}")
    dinamik_duyurulari_cek("https://www.ktun.edu.tr/tr/Birim/Duyurular/?brm=lU1aMXHXaICnOoQ8oGSy8g==")
finally:
    driver.quit()
