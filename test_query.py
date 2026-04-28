__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from embedding_config import create_embedding_model
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Yapılandırma
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding Model
embedding_model = create_embedding_model()

# Veritabanı
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="ktun_rag"
)

# LLM
llm = ChatOllama(
    model="qwen3:4b",
    base_url=OLLAMA_URL,
    temperature=0.3
)

# RAG Chain
system_prompt = (
    "Sen Konya Teknik Üniversitesi Bilgisayar Mühendisliği ve Yapay Zeka Bölümleri için bir asistansın. "
    "Aşağıdaki BAĞLAM BİLGİLERİNİ dikkatlice oku ve kullan. "
    "Ders kodları, ders adları, koordinatörler, hocalar, AKTS bilgileri gibi detayları doğrudan bul. "
    "Cevaplarını her zaman TÜRKÇE olarak ver. "
    "Eğer bilgi bağlamda varsa, DOĞRUDAN o bilgiyi kullan. "
    "Eğer bilgi bağlamda yoksa, 'Bu bilgiye sahip değilim' de."
    "\n\nBAĞLAM BİLGİLERİ:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    ), 
    qa_chain
)

# Test Soruları
test_questions = [
    "Ayrık Matematik dersinin ders kodu nedir ve dersin koordinatörü kimdir?",
    "Bilgisayar Mühendisliği bölüm başkanı kimdir?",
    "Veri Yapıları dersinin koordinatörü kimdir?"
]

print("=" * 60)
print("TEST BAŞLADI")
print("=" * 60)

for i, query in enumerate(test_questions, 1):
    print(f"\n\n{'='*60}")
    print(f"SORU {i}: {query}")
    print("=" * 60)
    
    # Alakalı dokümanları bul
    print("\n🔍 Dokümanlar aranıyor...")
    relevant_docs = vectorstore.similarity_search(query, k=20)
    
    if relevant_docs:
        print(f"✅ {len(relevant_docs)} doküman bulundu.")
        print("\nİlk 2 doküman önizleme:")
        for j, doc in enumerate(relevant_docs[:2], 1):
            print(f"\n--- Doküman {j} ---")
            print(doc.page_content[:300] + "...")
    
    # Cevap üret
    print("\n🤖 Cevap üretiliyor...")
    try:
        response = rag_chain.invoke({"input": query})
        print(f"\n✅ CEVAP:\n{response['answer']}")
        
        sources = list(set([doc.metadata.get('source', 'Bilinmiyor') 
                          for doc in response['context']]))
        print(f"\n📚 Kaynaklar ({len(sources)}):")
        for source in sources[:3]:
            print(f"  - {source}")
    except Exception as e:
        print(f"\n❌ HATA: {e}")

print("\n\n" + "=" * 60)
print("TEST TAMAMLANDI")
print("=" * 60)
