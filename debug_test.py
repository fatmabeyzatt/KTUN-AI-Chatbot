__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from embedding_config import create_embedding_model
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# 1. Yapılandırma
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "ktun_rag"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

print(f"\n--- DEBUG BAŞLATILDI ---")
print(f"Koleksiyon: {COLLECTION_NAME}")

# 2. Embedding Modelini Yükle (Cache kullanması için aynı ismi veriyoruz)
embeddings = create_embedding_model()

# 3. Veritabanına Bağlan
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# 4. Test Sorgusu: Veritabanından veri çekme testi
test_sorgusu = "Konya Teknik Üniversitesi hakkında ne biliyorsun?"
print(f"\n[1] Veritabanında arama yapılıyor: '{test_sorgusu}'")

docs = vectorstore.similarity_search(test_sorgusu, k=3)

if len(docs) > 0:
    print(f"✅ {len(docs)} döküman bulundu.")
    for i, doc in enumerate(docs):
        print(f"\n--- Bulunan Parça {i+1} (İlk 100 karakter) ---")
        print(doc.page_content[:150] + "...")
else:
    print("❌ HATA: Hiç döküman bulunamadı! Veritabanı yolu veya koleksiyon ismi yanlış olabilir.")
    sys.exit()

# 5. Model Bağlantı Testi (Qwen 3 4B)
print(f"\n[2] Qwen 3 4B modeline bağlanılıyor ({OLLAMA_URL})...")
llm = ChatOllama(model="qwen3:4b", base_url=OLLAMA_URL, temperature=0.1)

try:
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Aşağıdaki bilgileri kullanarak soruyu cevapla:\n\nBİLGİ:\n{context}\n\nSORU: {test_sorgusu}"
    
    response = llm.invoke(prompt)
    print(f"\n--- MODEL CEVABI ---")
    print(response.content)
    print(f"\n--- TEST BAŞARIYLA TAMAMLANDI ---")
except Exception as e:
    print(f"❌ MODEL HATASI: {e}")
