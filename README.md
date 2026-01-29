# KTUN-AI-Chatbot

## Ollama ve Model Ayarları
Ollama üzerinden Qwen3 4B modelinin kurulu olduğuna emin olun. Uygulama, Ollama modellerine URL üzerinden istek gönderir; erişim şu şekilde yapılır:

```python
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(
    model="qwen3:4b",
    base_url=OLLAMA_URL,
    temperature=0.1
)
```

## Veritabanı Bağlantısı
`collection_name` değeri hata alınmaması için `ktun_rag` olarak kalmalıdır:

```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="ktun_rag"
)
```

## Docker Notu
Docker ayağa kaldırılırken cache/no-cache satırlarını ihtiyacınıza göre düzenleyin.

## Notlar
- Tüm test kodları da dahil çalışmanın tamamı mevcut.
- `chroma_db` local olduğu için kodla tekrar oluşturulması tavsiye edilir.
- Docker’da modeli 1 kere yükleyip çalıştırıp `attach` ile modele CMD veya terminal üzerinden erişebilirsiniz.
- `requirements.txt` içinde versiyon uyuşmazlığı çıkmaması için dikkat edilmeli, hata alınırsa ilk buraya bakılmalı.
