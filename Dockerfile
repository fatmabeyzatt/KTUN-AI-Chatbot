FROM python:3.10-slim

# Sistem bağımlılıklarını yükle (HuggingFace modelleri için gerekli olabilir)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Adım: Ağır kütüphaneleri baştan yükle (Layer Caching için)
# Bu komut bir kez çalışır ve cache'e alınır. requirements.txt değişse bile burası tekrar çalışmaz.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    sentence-transformers==3.2.1 \
    langchain-huggingface==0.0.3

# 2. Adım: Gereksinim dosyasını kopyala ve kur (Zaten yüklü olanları atlar)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Python unbuffered mod (stdout/stderr hemen flush edilir)
ENV PYTHONUNBUFFERED=1

# Uygulamayı çalıştır
CMD ["python", "-u", "app.py"]
