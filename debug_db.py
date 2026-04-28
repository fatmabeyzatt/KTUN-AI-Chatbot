import chromadb
from langchain_chroma import Chroma
from chromadb.config import Settings
from embedding_config import create_embedding_model

# Telemetry fix
client_settings = Settings(
    anonymized_telemetry=False,
    allow_reset=True
)

embedding_model = create_embedding_model()

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    client_settings=client_settings
)

print("--- DB İnceleme ---")
# Basit bir sorgu yapalım
query = "Ayrık Matematik"
print(f"Sorgu: {query}")
docs = vectorstore.similarity_search(query, k=5)

if not docs:
    print("❌ Hiçbir doküman bulunamadı.")
else:
    print(f"✅ {len(docs)} doküman bulundu:")
    for i, doc in enumerate(docs):
        print(f"\n[{i+1}] Kaynak: {doc.metadata.get('source', 'Bilinmiyor')}")
        print(f"İçerik önizleme: {doc.page_content[:300]}...")

print("\n--- Tüm Koleksiyon İstatistikleri ---")
try:
    # collection count
    print(f"Koleksiyondaki toplam eleman: {vectorstore._collection.count()}")
    
    # "Ayrık" kelimesini içeren içerikleri manuel bulabilir miyiz?
    # Not: Chroma API üzerinden doğrudan 'where_document' benzeri basit text match yok, 
    # ama get() ile tüm veriyi çekip bakabiliriz (veri az ise).
    data = vectorstore._collection.get(include=['documents', 'metadatas'])
    documents = data['documents']
    count_keyword = 0
    for doc in documents:
        if "Ayrık Matematik" in doc:
            count_keyword += 1
            print(f" -> Bulundu: {doc[:100]}...")
            
    print(f"\n'Ayrık Matematik' geçen belge sayısı (exact match): {count_keyword}")

except Exception as e:
    print(f"İstatistik hatası: {e}")
