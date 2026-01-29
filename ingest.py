import os
import sys

# SQLite fix for Docker/Linux if needed
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import shutil
import glob
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings

# --- AYARLAR ---
DATA_KLASORU = "data"           # CSV'lerin olduğu klasör
VECTOR_DB_KLASORU = "chroma_db" # Veritabanının kaydedileceği yer

def create_pipeline():
    print("--- YAPAY ZEKA MÜHENDİSLİĞİ VERİ YÜKLEME MODU ---")
    
    # 1. TEMİZLİK: Eski 'ktun rag' verilerini temizleyelim
    if os.path.exists(VECTOR_DB_KLASORU):
        print(f"🧹 Eski veritabanı tespit edildi ve siliniyor (Temiz Kurulum)...")
        try:
            # Klasörün içini boşalt, klasörün kendisini silme (Docker mount hatasını önlemek için)
            for filename in os.listdir(VECTOR_DB_KLASORU):
                file_path = os.path.join(VECTOR_DB_KLASORU, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"   ⚠️ HATA: {file_path} silinemedi: {e}")
            print("   -> Temizlik başarılı.")
        except Exception as e:
            print(f"   ⚠️ HATA: Klasör temizlenemedi. ({e})")
            return

    # 2. DOSYALARI BUL
    csv_yollari = glob.glob(os.path.join(DATA_KLASORU, "*.csv"))
    
    if not csv_yollari:
        print(f"❌ HATA: '{DATA_KLASORU}' klasöründe hiç CSV dosyası yok!")
        return

    print(f"📂 İşlenecek Dosya Sayısı: {len(csv_yollari)}")
    
    tum_dokumanlar = []
    
    # 3. YÜKLEME DÖNGÜSÜ
    for dosya in csv_yollari:
        dosya_adi = os.path.basename(dosya)
        print(f"   Reading -> {dosya_adi} ... ", end="")
        
        try:
            # DİKKAT: Senin yeni dosyalarında link sütunu "KAYNAK LİNK" olarak geçiyor.
            # CSVLoader, diğer tüm sütunları (Tarih, Başlık, İçerik) otomatik olarak metne ekler.
            loader = CSVLoader(
                file_path=dosya, 
                encoding="utf-8", 
                source_column="KAYNAK LİNK" 
            )
            veri = loader.load()
            tum_dokumanlar.extend(veri)
            print(f"✅ (Eklenen belge: {len(veri)})")
        except Exception as e:
            print(f"\n   ❌ HATA: Dosya okunamadı. 'KAYNAK LİNK' sütunu var mı? Hata: {e}")

    print(f"\n📊 Toplam Veri Havuzu: {len(tum_dokumanlar)} parça")

    if len(tum_dokumanlar) == 0:
        print("Yüklenecek veri bulunamadı.")
        return

    # 4. PARÇALAMA (CHUNKING)
    # Ders programları ve tablolar olduğu için chunk size'ı biraz geniş tutuyoruz
    print("✂️  Metinler optimize ediliyor...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(tum_dokumanlar)
    
    # 5. EMBEDDING VE KAYIT
    print(f"🧠 Yapay Zeka Modeli (MiniLM) çalışıyor... Veritabanı oluşturuluyor...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Persistent client ayarları
    print(f"   -> Persistent Client oluşturuluyor: {os.path.abspath(VECTOR_DB_KLASORU)}")
    client_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )

    # Chroma.from_documents içerisinde client_settings parametresi kullanarak oluştur.
    Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_KLASORU,
        collection_name="ktun_rag",
        client_settings=client_settings
    )
    
    print(f"\n🎉 SİSTEM HAZIR! Tüm yapay zeka mühendisliği verileri yüklendi.")
    print(f"Debug: {VECTOR_DB_KLASORU} içeriği: {os.listdir(VECTOR_DB_KLASORU)}")

if __name__ == "__main__":
    create_pipeline()
