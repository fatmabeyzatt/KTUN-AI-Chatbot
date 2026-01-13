# 1. Python 3.10 kullan
FROM python:3.10-slim

# 2. Linux sistem güncellemelerini yap ve derleme araçlarını kur (Hata almamak için)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Çalışma klasörünü ayarla
WORKDIR /app

# 4. Kütüphaneleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tüm kodları kopyala
COPY . .

# 6. SQLite düzeltmesi (ChromaDB için gerekebiliyor)
ENV LD_LIBRARY_PATH=/usr/local/lib

# 7. Başlatma komutu (JSON formatı warning uyarısını da giderir)
CMD ["sh", "-c", "python ingest.py && python test_rag.py"]