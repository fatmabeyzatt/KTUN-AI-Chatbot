__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings

# Eski DB'yi temizle
VECTOR_DB_KLASORU = "chroma_db"
if os.path.exists(VECTOR_DB_KLASORU):
    print(f"🧹 Eski veritabanı temizleniyor...")
    for filename in os.listdir(VECTOR_DB_KLASORU):
        file_path = os.path.join(VECTOR_DB_KLASORU, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"   ⚠️ {file_path} silinemedi: {e}")

print("\n📂 Optimize edilmiş veri yükleme başlıyor...")

# 1. Veriyi Yükle
df = pd.read_csv('data/bilgisayar_statik_veriler.csv')

# 2. Embedding Modelini Hazırla
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. Metin Parçalayıcı
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

documents = []

print("✂️ Veriler optimize ediliyor...")

for index, row in df.iterrows():
    modul = str(row['MODÜL'])
    konu = str(row['KONU / DOSYA ADI'])
    icerik = str(row['İÇERİK (METİN / TABLO)'])
    link = str(row['KAYNAK LİNK'])
    
    # ÖZEL İŞLEM: "BÖLÜM DERSLERİ" ve "DÖNEM" içeren satırları ayrıştır
    if "BÖLÜM DERSLERİ" in modul and "DÖNEM" in konu:
        # İçerik zaten satırlara ayrılmış olabilir
        donem = konu  # Örn: "DÖNEM 3"
        
        # İçeriği satırlara ayır - her ders bir satır
        lines = [line.strip() for line in icerik.strip().split('\n') if line.strip()]
        
        for line in lines:
            # Başlık satırını atla
            if "Ders Kodu" in line or "Dersin Koordinatörü" in line or len(line) < 10:
                continue
            
            # Pipe karakteri ile ayrılmış ders satırlarını işle
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    ders_kodu = parts[0]
                    ders_adi = parts[1]
                    
                    # Her ders için özelleştirilmiş doküman
                    content = f"Bilgisayar Mühendisliği {donem}\n"
                    content += f"Ders Kodu: {ders_kodu}\n"
                    content += f"Ders Adı: {ders_adi}\n"
                    if len(parts) >= 3 and parts[2]:
                        content += f"AKTS: {parts[2]}\n"
                    if len(parts) >= 4 and parts[3]:
                        content += f"Koordinatör: {parts[3]}\n"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": link,
                            "modul": modul,
                            "donem": donem,
                            "ders_kodu": ders_kodu,
                            "ders_adi": ders_adi,
                            "tip": "ders_bilgisi"
                        }
                    )
                    documents.append(doc)
        
        # Ayrıca tüm tabloyu da ekle (genel sorular için)
        header = f"Modül: {modul} | {konu}\n"
        full_content = header + icerik
        doc = Document(
            page_content=full_content,
            metadata={"source": link, "modul": modul, "konu": konu, "tip": "tablo"}
        )
        documents.append(doc)
        
    else:
        # Diğer içerikler için normal işlem
        header = f"Modül: {modul} | Konu: {konu}\n"
        
        if len(icerik) > 800:
            chunks = text_splitter.split_text(icerik)
            for i, chunk in enumerate(chunks):
                full_content = header + f"(Kısım {i+1})\n" + chunk
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "source": link,
                        "modul": modul,
                        "konu": konu,
                        "chunk": i
                    }
                )
                documents.append(doc)
        else:
            full_content = header + icerik
            doc = Document(
                page_content=full_content,
                metadata={"source": link, "modul": modul, "konu": konu}
            )
            documents.append(doc)

print(f"\n📊 Toplam {len(documents)} optimize edilmiş parça oluşturuldu.")
print("🧠 ChromaDB'ye kaydediliyor...")

# 4. ChromaDB'ye Kaydet
client_settings = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=VECTOR_DB_KLASORU,
    collection_name="ktun_rag",
    client_settings=client_settings
)

print(f"\n🎉 Başarılı! {len(documents)} doküman optimize edilmiş şekilde kaydedildi.")
print(f"Debug: {VECTOR_DB_KLASORU} içeriği: {os.listdir(VECTOR_DB_KLASORU)}")