import os
import sys
import shutil
import glob

# Kütüphane yollarını garantiye al
sys.path.append(os.getcwd())

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- AYARLAR ---
DATA_KLASORU = "data"           # CSV'lerin olduğu klasör
VECTOR_DB_KLASORU = "chroma_db" # Veritabanının kaydedileceği yer

def create_pipeline():
    print("--- YAPAY ZEKA MÜHENDİSLİĞİ VERİ YÜKLEME MODU ---")
    
    # 1. TEMİZLİK: Eski 'ktun rag' verilerini temizleyelim
    if os.path.exists(VECTOR_DB_KLASORU):
        print(f"🧹 Eski veritabanı tespit edildi ve siliniyor (Temiz Kurulum)...")
        try:
            shutil.rmtree(VECTOR_DB_KLASORU)
            print("   -> Temizlik başarılı.")
        except Exception as e:
            print(f"   ⚠️ HATA: Klasör silinemedi. Chatbot veya terminal açık olabilir mi? ({e})")
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

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_KLASORU
    )
    
    print(f"\n🎉 SİSTEM HAZIR! Tüm yapay zeka mühendisliği verileri yüklendi.")

if __name__ == "__main__":
    create_pipeline()