import os
import sys

# Kütüphane yollarını garantiye al
sys.path.append(os.getcwd())

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- AYARLAR ---
VECTOR_DB_KLASORU = "chroma_db"

def test_database():
    print("--- YAPAY ZEKA BÖLÜMÜ TEST MODU ---")
    
    if not os.path.exists(VECTOR_DB_KLASORU):
        print(f"HATA: '{VECTOR_DB_KLASORU}' klasörü bulunamadı!")
        return

    print("1. Model ve Veritabanı Yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_KLASORU, 
        embedding_function=embedding_model
    )

    print("✅ Veritabanı Bağlandı! (Çıkmak için 'q' yazın)\n")

    while True:
        soru = input("\n🔎 SORU SOR (Örn: 'Bölüm başkanı kim?'): ")
        if soru.lower() == 'q': break
        
        # k=3: En alakalı 3 belgeyi getir
        results = vectorstore.similarity_search_with_score(soru, k=3)

        print(f"\n--- '{soru}' İÇİN SONUÇLAR ---")
        
        if len(results) == 0:
            print("❌ Hiçbir sonuç bulunamadı.")
        else:
            for i, (doc, score) in enumerate(results):
                # Metadata'dan kaynak linkini çekiyoruz
                kaynak = doc.metadata.get("source", "Bilinmiyor")
                
                # Dosya adını da gösterelim ki hangi CSV'den geldiğini anlayalım
                # (CSVLoader dosya yolunu da metadata'ya ekler)
                dosya_yolu = doc.metadata.get("source", "") 
                # Eğer source link ise dosya adını row veya başka yerden bulamayabiliriz, 
                # ama içerikten anlayacağız.
                
                print(f"\n📄 [SONUÇ {i+1}] (Alaka Skoru: {score:.3f})")
                print(f"🔗 Kaynak: {kaynak}")
                # İçeriğin boşluklarını temizleyip ilk 300 karakteri göster
                icerik_ozeti = doc.page_content.replace("\n", " ")[:350]
                print(f"📝 İçerik: {icerik_ozeti}...") 
                print("-" * 40)

if __name__ == "__main__":
    test_database()