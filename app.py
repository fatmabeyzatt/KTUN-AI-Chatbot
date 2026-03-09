__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Çevre değişkeninden Ollama URL'sini al
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 2. Embedding Modelini Yükle
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. Veritabanına Bağlan
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="ktun_rag"
)

# 4. LLM Yapılandırması
llm = ChatOllama(
    model="qwen3:4b",
    base_url=OLLAMA_URL,
    temperature=0.1  # Daha deterministik, bağlama sadık cevaplar için düşürüldü
)

# 5. RAG Zinciri - Çok Katı Grounding
system_prompt = (
    "Sen KONYA TEKNİK ÜNİVERSİTESİ (KTÜN) Bilgisayar Mühendisliği Bölümü asistanısın.\n\n"
    "KRİTİK KURALLAR - ASLA ATLAMA:\n"
    "1. SADECE aşağıdaki metinde DOĞRUDAN YAZAN bilgileri kullan\n"
    "2. Kendi bilgin = YASAK! Tahmin = YASAK! Yorum = YASAK!\n"
    "3. Başka üniversite (Bilkent/ODTÜ/Boğaziçi/vs) = YASAK!\n"
    "4. Türkiye genel bilgisi = YASAK! Sadece KTÜN!\n"
    "5. Eğer aşağıdaki metinde GÖREMEZSEN 'Bu bilgiye sahip değilim' yaz\n"
    "ŞİMDİ AŞAĞIDAKİ METNE BAK, BAŞKA YERDEKİ BİLGİNİ KULLANMA!\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "BAĞLAM METNİ (SADECE BURADAN CEVAP VER):\n\n"
    "{context}\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "CEVABINI BU METİNDEN VER. BAŞKA BİLGİ KULLANMA!"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}  # Daha fazla doküman çekerek daha iyi sonuç al
    ), 
    qa_chain
)

if __name__ == "__main__":
    print("Asistan çalışıyor... (Çıkmak için 'exit' yazın)")
    sys.stdout.flush()
    
    while True:
        try:
            sys.stdout.write("\nSoru: ")
            sys.stdout.flush()
            query = sys.stdin.readline().strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']: 
                break
            
            # Debug: Hangi dokümanların geldiğini gör
            print("\n🔍 Alakalı dokümanlar aranıyor...")
            relevant_docs = vectorstore.similarity_search(query, k=20)
            
            # DEBUG: İlk 3 dokümanı göster
            if relevant_docs:
                print(f"\n✅ {len(relevant_docs)} doküman bulundu. İlk 3 önizleme:")
                sys.stdout.flush()
                for i, doc in enumerate(relevant_docs[:3]):
                    print(f"\n--- Doküman {i+1} ---")
                    print(doc.page_content[:300] + "...")
                    sys.stdout.flush()
            else:
                print("\n⚠️ Hiç doküman bulunamadı!")
                sys.stdout.flush()
        except (EOFError, KeyboardInterrupt):
            print("\n\nÇıkış yapılıyor...")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")
            continue
        # for i, doc in enumerate(relevant_docs):
        #     print(f"\n--- Doküman {i+1} ---")
        #     print(doc.page_content[:200] + "...")
            
        print("🤖 Cevap üretiliyor...")
        sys.stdout.flush()
        response = rag_chain.invoke({"input": query})
        
        # Cevap doğrulama: Halüsinasyon kontrolü
        answer = response['answer']
        context_text = " ".join([doc.page_content for doc in response['context']])
        
        # Şüpheli kelimeler kontrolü (başka üniversite isimleri + genel Türkiye ifadeleri)
        suspicious_words = [
            "bilkent", "odtü", "boğaziçi", "hacettepe", "ankara üniv", "istanbul üniv",
            "türkiye'de", "türkiyede", "üniversitelerde", "genellikle", "örneğin",
            "bazı üniversite", "bilim ve teknoloji üniversitesi"
        ]
        
        is_suspicious = any(word in answer.lower() for word in suspicious_words)
        
        # Eğer cevap uzunsa ve bağlamda olmayan bilgi veriyorsa engelle
        if is_suspicious or (len(answer) > 300 and "bu bilgi" not in answer.lower()):
            print(f"\nCevap: Bu bilgiye sahip değilim.")
            print(f"\n⚠️ Uyarı: Sistem KTÜN dışı bilgi üretmeye çalıştı. Cevap engellendi.")
        else:
            print(f"\nCevap: {answer}")
        
        print(f"\nKaynaklar: {list(set([doc.metadata.get('source', 'Bilinmiyor') for doc in response['context']]))}")
        sys.stdout.flush()