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

## Embedding Modeli (Offline / Local Path)
Embedding yuklemesi `embedding_config.py` uzerinden merkezi olarak yonetilir.

Desteklenen ortam degiskenleri:
- `EMBEDDING_MODEL_NAME` (varsayilan: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- `EMBEDDING_MODEL_PATH` (local model klasoru; varsa repo adina gore onceliklidir)
- `EMBEDDING_OFFLINE=true` (sadece local/cache dosyalarini kullanir)
- `HF_HOME` (HuggingFace cache klasoru)

Ornek:
```powershell
$env:EMBEDDING_MODEL_PATH="C:\models\paraphrase-multilingual-MiniLM-L12-v2"
$env:EMBEDDING_OFFLINE="true"
python app.py
```

## Terminal Cikti Modu
`app.py` varsayilan olarak sadece cevabi yazdirir.

Opsiyonel ortam degiskenleri:
- `ANSWER_ONLY=true|false` (varsayilan: `true`)
- `SHOW_SOURCES=true|false` (varsayilan: `false`)
- `STRUCTURED_ONLY=true|false` (varsayilan: `false`, acilinca sadece structured DB cevaplar)

## Ingest (Incremental)
`ingest.py` artik tum vektoru her calismada silmez.

- `data/*.csv|json` ve `data/processed/**/*` recursive okunur.
- `data/raw/pdf/**/*` altindaki PDF'ler parse edilip indexlenir.
- Degisen/yeni kaynaklar yeniden embed edilir, silinen kaynaklar koleksiyondan temizlenir.
- Durum dosyasi: `chroma_db/.ingest_state.json`

Not: `SHOW_PROGRESS` env degiskeni artik dokuman onizlemesi basmaz; uygulama sade cikti verir.

